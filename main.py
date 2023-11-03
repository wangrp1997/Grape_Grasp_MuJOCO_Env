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
import multiprocessing


# initialize our robot config for the jaco2
robot_config = arm("ur5e.xml", folder='./ur5', use_sim_state=False)

# create our path planner
n_timesteps = 2000



maximum_range = 0.001  # 传感器最大量程，根据实际情况进行调整
plt.figure()
sensor_data = np.zeros((4, 4))

# create our interface
dt = 0.002
interface = Mujoco(robot_config, dt=dt)
interface.connect()
interface.send_target_angles(robot_config.START_ANGLES)
flag = True
target = None
ur5_control = UR5Control(interface, robot_config, n_timesteps)

# 初始化计数器和采样周期
counter = 0
sampling_period = 10  # 采样周期，例如每10个回调周期采样一次

# 初始化传感器数据
sensor_data = np.zeros((4, 4))

# 创建一个Matplotlib图形对象
img = plt.imshow(sensor_data, cmap='viridis', interpolation='nearest')
# # 添加颜色条
plt.colorbar()
# 设置图形标题
plt.title('Sensor Data Visualization')

# 更新传感器数据的回调函数
# 更新传感器数据的回调函数
def sensor_callback(model, data):
    global sensor_data
    global sensor_queue
    for i in range(16):
        sensor_name = f'touch{i}'
        data_value = data.sensor(sensor_name).data[1].copy()
        sensor_data[i // 4, i % 4] = data_value
    sensor_queue.put(sensor_data)  # 通过队列传递传感器数据

# 设置传感器回调
sensor_queue = multiprocessing.Queue()
mujoco.set_mjcb_control(sensor_callback)

# 定义一个函数用于绘制图形的进程
def plot_data(sensor_queue):
    while True:
        sensor_data = sensor_queue.get()  # 从队列获取传感器数据
        img.set_data(sensor_data)
        img.autoscale()
        plt.pause(0.01)

plot_process = multiprocessing.Process(target=plot_data, args=(sensor_queue,))


try:
    print("\nSimulation starting...")
    print("Click to move the target.\n")

    # 启动图形绘制进程
    plot_process.start()

    while not glfw.window_should_close(interface.viewer.window):

        if flag:
            print(robot_config.model.joint("slide_joint").damping[0])
            ur5_control.move_ee2(pos=[-0.135,0.495,0.16])
            ur5_control.close_gripper(0.7)   # upper:0.796
            ur5_control.move_ee2(pos=[-0.135, 0.495, 0.26])

            ur5_control.close_gripper(0.0)   # upper:0.796
            ur5_control.move_ee2(pos=[-0.135,0.495,0.16])
            ur5_control.close_gripper(0.7)   # upper:0.796

            ur5_control.joint_control(detal_array=np.array([0, 0, 0, 0, 0, np.pi]))
            robot_config.model.joint("slide_joint").damping[0] = 1
            print(robot_config.model.joint("slide_joint").damping[0])
            ur5_control.move_ee2(pos=[-0.135, 0.495, 0.26])
            #
            flag = False

        interface.viewer.render()


finally:
    # stop and reset the simulation
    interface.disconnect()
    plot_process.terminate()
    print("Simulation terminated...")
