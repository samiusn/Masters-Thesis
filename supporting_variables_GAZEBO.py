# -*- coding: utf-8 -*-
"""
Created on Sun May 22 16:35:59 2022

@author: Syed Sami Al Haq
"""
import casadi as ca
import numpy as np
import queue
import rospy
import matplotlib.pyplot as plt
import pymap3d as pm
from casadi import sin, cos, pi
from variables import *
from std_msgs.msg import Float32
from time import time, ctime
from casadi import sin, cos, pi
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped
from tf.transformations import euler_from_quaternion


Q_x = 100
Q_y = 100
Q_theta = 100
Q_u = 1
Q_v = 0
Q_r = 0

R1 = 0.01
R2 = 0.01

step_horizon = 1  # time between steps in seconds?? 0.1 Try diefferent step size.
N = 10              # number of look ahead steps?? May increase to 2 sec horizon
sim_time = 1700      # simulation time

# specs
x_init = 0
y_init = 0
theta_init = pi*3/2
u_init = 0
v_init = 0
r_init = 0

x_target = ultimate_x_target = 11
y_target = ultimate_y_target = 60
theta_target = pi/2
u_target = 1
v_target = 1
r_target = 1

v_max = 250
v_min = -100

state_init = ca.DM([x_init, y_init, theta_init, u_init, v_init, r_init])        # initial state
state_target = ca.DM([x_target, y_target, theta_target, u_target, v_target, r_target])  # target state

#Obstacle
obstacle_axis = [6.5,23]
obstacle_clearance = 5
obstacle_stationary_trajectory = [[6.5,23]]
obstacle_moving_trajectory = [[30,10],[20,20],[10,30],[0,40]]
obstacleFlag = True
movingTargetCounter = -1

#Plot
graphLimitPos = 70
graphLimitNeg = -10
ax = plt.gca()
ax.set_xlim((graphLimitNeg,graphLimitPos))
ax.set_ylim((graphLimitNeg,graphLimitPos))
ax.set_aspect(1)

#ROS
latitude_ori = 0
longitude_ori = 0
altitude_ori = 0

init_counter = True

#Loop variables
mpc_iter = 0

m = 250      #Mass of USV
Iz = 495      #Value did not find in code gussed and set value
Xu = 53.1     #Value from code
Yv = 40
Nr = 400
Xdotu = 0#-0.08  #Value from code
Ydotv = -0#50   #Value from code
Ndotr = 0   #Value from code
dp = 2.4      #Value from code

# state symbolic variables
x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')
u = ca.SX.sym('u')
v = ca.SX.sym('v')
r = ca.SX.sym('r')

states = ca.vertcat(
    x,
    y,
    theta,
    u,
    v,
    r
)
n_states = states.numel()

# control symbolic variables
XP1 = ca.SX.sym('XP1')
XP2 = ca.SX.sym('XP2')

controls = ca.vertcat(
    XP1,
    XP2
)
n_controls = controls.numel()

# matrix containing all states over all time steps +1 (each column is a state vector)
X = ca.SX.sym('X', n_states, N + 1)

# matrix containing all control actions over all time steps (each column is an action vector)
U = ca.SX.sym('U', n_controls, N)

# coloumn vector for storing initial state and target state
P = ca.SX.sym('P', n_states + n_states)

# state weights matrix (Q_X, Q_Y, Q_THETA)
Q = ca.diagcat(Q_x, Q_y, Q_theta, Q_u, Q_v, Q_r)

# controls weights matrix
R = ca.diagcat(R1, R2)

lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
##########
lbx[0: n_states*(N+1): n_states] = -1000     # X lower bound
lbx[1: n_states*(N+1): n_states] = -1000     # Y lower bound
lbx[2: n_states*(N+1): n_states] = -1000     # theta lower bound
lbx[3: n_states*(N+1): n_states] = -30        # u lower bound
lbx[4: n_states*(N+1): n_states] = -30        # v lower bound
lbx[5: n_states*(N+1): n_states] = -10     # r lower bound

ubx[0: n_states*(N+1): n_states] = 1000      # X upper bound
ubx[1: n_states*(N+1): n_states] = 1000      # Y upper bound
ubx[2: n_states*(N+1): n_states] = 1000      # theta upper bound
ubx[3: n_states*(N+1): n_states] = 30         # u upper bound
ubx[4: n_states*(N+1): n_states] = 30        # v upper bound
ubx[5: n_states*(N+1): n_states] = 10      # r upper bound

lbx[n_states*(N+1):] = v_min                   # tau lower bound
ubx[n_states*(N+1):] = v_max                   # tau upper bound


args = {
    'lbg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints lower bound
    'ubg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints upper bound
    'lbx': lbx,
    'ubx': ubx
}

RHS = ca.vertcat(u*cos(theta)- v*sin(theta),\
                 u*sin(theta)+ v*cos(theta),\
                 r,\
                (-Xu*u + (XP1+XP2) + (m-Ydotv)*v*r)/(m-Xdotu),\
                (-Yv*v - (m-Xdotu)*u*r)/(m-Ydotv),\
                ((-Nr*r) + ((XP1-XP2)*dp) + (-Xdotu+Ydotv)*u*v)/(Iz-Ndotr))
    
# maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
f = ca.Function('f', [states, controls], [RHS])


cost_fn = 0  # cost function
g = X[:, 0] - P[:n_states]  # constraints in the equation


# runge kutta
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    cost_fn = cost_fn \
        + (st - P[n_states:]).T @ Q @ (st - P[n_states:]) \
        + con.T @ R @ con
    st_next = X[:, k+1]
    k1 = f(st, con)
    k2 = f(st + step_horizon/2*k1, con)
    k3 = f(st + step_horizon/2*k2, con)
    k4 = f(st + step_horizon * k3, con)
    st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    g = ca.vertcat(g, st_next - st_next_RK4)

OPT_variables = ca.vertcat(
    X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
    U.reshape((-1, 1))
)

nlp_prob = {
    'f': cost_fn,
    'x': OPT_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
        'max_iter': 10000, #Correction Try out first
        'print_level': 0,
        'acceptable_tol': 1e-4,
        'acceptable_obj_change_tol': 1e-2
    },
    'print_time': 0
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

mpc_iter = 0
u0 = ca.DM.zeros((n_controls, N))          # initial control
X0 = ca.repmat(state_init, 1, N+1)         # initial state full
t0 = 0
t = ca.DM(t0)

def DM2Arr(dm):
    return np.array(dm.full())

cat_states = DM2Arr(X0)
cat_controls = DM2Arr(u0[:, 0])
cont_XP1 = DM2Arr(u0[0, 0])
cont_XP2 = DM2Arr(u0[1, 0])
times = np.array([[0]])


def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0