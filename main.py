"""
Running the joint controller with an inverse kinematics path planner
for a Mujoco simulation. The path planning system will generate
a trajectory in joint space that moves the end effector in a straight line
to the target, which changes every n time steps.
"""
import glfw
from abr_control.arms.mujoco_config import MujocoConfig as arm
from abr_control.interfaces.mujoco import Mujoco
from robot_control import UR5Control

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# initialize our robot config for the jaco2
robot_config = arm("ur5e.xml", folder='./ur5', use_sim_state=False)

# create our path planner
n_timesteps = 2000

# create our interface
dt = 0.002
interface = Mujoco(robot_config, dt=dt)
interface.connect()
interface.send_target_angles(robot_config.START_ANGLES)
flag = True
target = None
ur5_control = UR5Control(interface, robot_config, n_timesteps)

maximum_range = 20  # 传感器最大量程，根据实际情况进行调整


def sensor_callback(model, data):
    sensor_data = []  # 初始化传感器数据

    # global sensor_data
    for i in range(16):
        sensor_name = f'touch{i}'
        data_value = data.sensor(sensor_name).data[1].copy()
        sensor_data.append(data_value)
    sensor_data = np.array(sensor_data).reshape((4, 4))


# 设置传感器回调
# mujoco.set_mjcb_control(sensor_callback)

# 创建动画

try:
    print("\nSimulation starting...")
    print("Click to move the target.\n")

    count = 0
    while not glfw.window_should_close(interface.viewer.window):

        if flag:
            print(robot_config.model.joint("slide_joint").damping[0])
            ur5_control.move_ee2(pos=[-0.135,0.495,0.16])
            ur5_control.close_gripper(0.7)   # upper:0.796
            ur5_control.move_ee1(detal=[0, 0, 0.1])

            ur5_control.close_gripper(0.0)   # upper:0.796
            ur5_control.move_ee2(pos=[-0.135,0.495,0.16])
            ur5_control.close_gripper(0.7)   # upper:0.796

            ur5_control.joint_control(detal_array=np.array([0, 0, 0, 0, 0, np.pi]))
            robot_config.model.joint("slide_joint").damping[0] = 1
            print(robot_config.model.joint("slide_joint").damping[0])
            ur5_control.move_ee1(detal=[0, 0, 0.177])
            #
            flag = False

        interface.viewer.render()


finally:
    # stop and reset the simulation
    interface.disconnect()

    print("Simulation terminated...")
