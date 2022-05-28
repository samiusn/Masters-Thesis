# -*- coding: utf-8 -*-
"""
Created on Sun May 22 17:07:16 2022

@author: Syed Sami Al Haq
"""
from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
import matplotlib.pyplot as plt
from variables import *

def gpsCallback(data):
    # print(data.latitude, data.longitude)
    global x_init, y_init, z_init,\
         init_counter,latitude_ori, longitude_ori, altitude_ori
    
    latitude = data.latitude
    longitude = data.longitude
    altitude = data.altitude
    #To capture initial lat long for ENU reference
    #Executes only once at beganing
    if init_counter:
        latitude_ori = latitude
        longitude_ori = longitude
        altitude_ori = altitude
        init_counter = False

    x_init_temp, y_init_temp, z_init_temp = pm.geodetic2enu(latitude, longitude,\
                            altitude, latitude_ori, longitude_ori, altitude_ori)
    x_init = x_init_temp
    y_init = y_init_temp
    z_init = z_init_temp

def imuCallback(data):
    global r_init, theta_init
    (roll,pitch,yaw) = euler_from_quaternion([data.orientation.x,\
        data.orientation.y,data.orientation.z,data.orientation.w])
    theta_init = yaw
    # print(theta_init_copy)
    r_init_temp = data.angular_velocity.z
    r_init = r_init_temp

def fixVelCallback(data):
    # print(data)
    global u_init, v_init
    u_init_temp = data.vector.x
    v_init_temp = data.vector.y
    u_init = u_init_temp
    v_init = v_init_temp

def isTargetReached():
    global mpc_iter, step_horizon, sim_time, state_init,\
        state_target, obstacleFlag
    
    if (mpc_iter * step_horizon > sim_time):
        return True
    
    print(ca.norm_2(state_init[0:3] - ca.DM([ultimate_x_target, \
        ultimate_y_target, theta_target])))
    print(state_target[0], state_target[1])
    
    if (ca.norm_2(state_init[0:2] - state_target[0:2]) < 5):
        if obstacleFlag:
            obstacleFlag = False
        state_target[0] = ultimate_x_target
        state_target[1] = ultimate_y_target
        if (ca.norm_2(state_init[0:3] - state_target[0:3]) < 0.01):
            return True
    
    
    return False

def chaseTarget():
    global args, state_init, state_target, n_states,\
        n_controls, N, f, solver, nlp_prob, mpc_iter, u0,\
        X0, t0, t, cat_states, cat_controls, cont_XP1,\
        cont_XP2, times, x_init, y_init, theta_init,\
        u_init, v_init, r_init

    t1 = time()

    state_init[0,0] = x_init
    state_init[1,0] = y_init
    state_init[2,0] = theta_init
    state_init[3,0] = u_init
    state_init[4,0] = v_init
    state_init[5,0] = r_init
    print(state_init)

    args['p'] = ca.vertcat(
        state_init,    # current state
        state_target   # target state
    )
    # optimization variable current state
    # args['x0'] = ca.vertcat(
    #     ca.reshape(X0, n_states*(N+1), 1),
    #     ca.reshape(u0, n_controls*N, 1)
    # )
    
    args['x0'] = np.zeros(n_states*(N+1) + n_controls*N)
    # print(args['p'])
    
    sol = solver(
        x0=args['x0'],
        lbx=args['lbx'],
        ubx=args['ubx'],
        lbg=args['lbg'],
        ubg=args['ubg'],
        p=args['p']
    )
    
    #Extract and seperate control and signal data from solver output
    u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
    X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)

    # print(u)
    #Store state and control for ploting
    cat_states = np.dstack((
        cat_states,
        DM2Arr(X0)
    ))
    cat_controls = np.vstack((
        cat_controls,
        DM2Arr(u[:, 0])
    ))
    
    cont_XP1 = DM2Arr(u[0, 0])
    cont_XP2 = DM2Arr(u[1, 0])
    
    #Scaling control to 1 to -1
    if cont_XP1 < 0 :
        cont_XP1 = DM2Arr(u[0, 0])/100
    else :
        cont_XP1 = DM2Arr(u[0, 0])/250

    if cont_XP2 < 0 :
        cont_XP2 = DM2Arr(u[1, 0])/100
    else :
        cont_XP2 = DM2Arr(u[1, 0])/250
    
    print(cont_XP1, cont_XP2)

    #Publish to ROS
    if not rospy.is_shutdown():
        pub_r.publish(cont_XP1)
        pub_l.publish(cont_XP2)


    # print(DM2Arr(X0)[0,0], DM2Arr(X0)[1,0], DM2Arr(X0)[2,0])
    # cont_XP1 = 
    # cont_XP1 = np.vstack((cont_XP1,DM2Arr(u[0, 0])))
    # cont_XP2 = np.vstack((cont_XP2,DM2Arr(u[1, 0])))
    
    t = np.vstack((
        t,
        t0
    ))

    #Mathematical iteration muted for GAZEBO
    # t0, state_init, u0 = shift_timestep(step_horizon, t0, state_init, u, f)

    # print(X0)
    X0 = ca.horzcat(
        X0[:, 1:],
        ca.reshape(X0[:, -1], -1, 1)
    )

    # xx ...
    t2 = time()
    print(mpc_iter)
    # print(t2-t1)
    times = np.vstack((
        times,
        t2-t1
    ))

    mpc_iter = mpc_iter + 1

def isObstacleDetected():
    global obstacleFlag, state_init, state_target
    
    #External code to detec next obstacle
    
    return obstacleFlag

def isObstacleStationary():
    return True

def updateTarget():
    global state_target
    
    state_target[0] = obstacle_axis[0] + obstacle_clearance
    state_target[1] = obstacle_axis[1] - 3
    # state_target[0,2] = 100

def updateMovingTarget(heading, velocity, location,):
    global state_target, mpc_iter, movingTargetCounter, mpc_iter

    if mpc_iter%33 == 0:
        if movingTargetCounter < 2:
            movingTargetCounter = movingTargetCounter + 1
    if isObstacleOnRight():
        state_target[0] = obstacle_moving_trajectory[movingTargetCounter][0]
        state_target[1] = obstacle_moving_trajectory[movingTargetCounter][1]
    else:
        state_target[0] = ultimate_x_target
        state_target[1] = ultimate_y_target

#Input will come from visual sensors
#Dummy values used now.
#Input : None
def isObstacleOnRight():
    global mpc_iter
    
    if mpc_iter < 30:
        return True
    else :
        False

#Draws obstacle is grap
#Input : trajectory in array
def drawObstacle(trajectory):
    global ax, p, obstacle_axis, obstacle_clearance
    
    for axis in trajectory:
        ax.plot((axis[0]), (axis[1]), 'o', alpha=1, color='y')
        p = plt.Circle(( axis[0] , axis[1] ), obstacle_clearance 
                       ,color='r', alpha=1, fill=False)  
        ax.add_artist(p)
        
    # ax.plot(trajectory[0][0], trajectory[0][1], 'o', color='r')
    # p = plt.Circle((trajectory[3][0] , trajectory[3][1] ), obstacle_clearance 
    #                ,color='r', fill=False)  
    ax.add_artist(p)

rospy.Subscriber("/wamv/sensors/gps/gps/fix", NavSatFix, gpsCallback)
rospy.Subscriber("/wamv/sensors/imu/imu/data", Imu, imuCallback)
rospy.Subscriber("/wamv/sensors/gps/gps/fix_velocity", Vector3Stamped, fixVelCallback)
pub_r = rospy.Publisher('/wamv/thrusters/right_thrust_cmd', Float32, queue_size = 10)
pub_l = rospy.Publisher('/wamv/thrusters/left_thrust_cmd', Float32, queue_size = 10)

rospy.init_node('publisher_thruster_right', anonymous=True)
rate = rospy.Rate(10)

##############################################################################
############################ Main loop starts here ###########################
##############################################################################
while True:
    #Check if target reached
    if isTargetReached():
        break
    if isObstacleDetected():
        if isObstacleStationary():
            updateTarget()
        else:
            updateMovingTarget(0,0,0)
    chaseTarget()


################################### Plotting ##################################
ax.plot(cat_states[0, 0], cat_states[1, 0])

#Print Obstacle
if isObstacleStationary():
    drawObstacle(obstacle_stationary_trajectory)
else :
    drawObstacle(obstacle_moving_trajectory)
    
#Plot Start
ax.plot(0,0, 'o', color='k')
#Plot End
ax.plot(ultimate_x_target, ultimate_y_target, 'o', color='r')
plt.show()










