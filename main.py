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
from OpenGL.GL import *
from OpenGL.GLUT import *
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

# initialize our robot config for the jaco2
robot_config = arm("ur5e.xml", folder='./ur5', use_sim_state=False)

maximum_range = 0.001  # 传感器最大量程，根据实际情况进行调整
sensor_data = np.zeros((4, 4))
# 创建一个图形窗口并设置子图
plt.figure()
plt.suptitle('Sensor Data Visualization')
plt.axis('on')  # 隐藏坐标轴
sensor_subplot = plt.subplot(111)

# 初始化传感器数据和颜色范围
np.zeros((4, 4))
vmin, vmax = 0, 4

# 创建初始的colorbar
im = sensor_subplot.imshow(sensor_data, cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, 4, 0, 4])  # 设置x轴和y轴范围
# sensor_subplot.set_title('Sensor Data')

cbar = plt.colorbar(im, ax=sensor_subplot)
cbar.set_label('Sensor Value')  # 颜色条标题


# create our path planner
n_timesteps = 2000
# create our interface
dt = 0.002
interface = Mujoco(robot_config, dt=dt, im=im)
interface.connect()
interface.send_target_angles(robot_config.START_ANGLES)
flag = True
target = None
ur5_control = UR5Control(interface, robot_config, n_timesteps)



try:
    print("\nSimulation starting...")
    print("Click to move the target.\n")


    while not glfw.window_should_close(interface.viewer.window):

        if flag:
            # print(robot_config.model.joint("slide_joint").damping[0])
            ur5_control.move_ee2(pos=[-0.135, 0.495, 0.16])
            ur5_control.close_gripper(0.7)   # upper:0.796
            ur5_control.move_ee2(pos=[-0.135, 0.495, 0.26])
            print(f'Current grasp Z position: {robot_config.data.body("grape_0").xpos[2]}')

            while robot_config.data.body("grape_0").xpos[2] < 0.24:
                ur5_control.close_gripper(0.0)   # upper:0.796
                ur5_control.move_ee2(pos=[-0.135, 0.495, 0.16])
                ur5_control.close_gripper(0.7)   # upper:0.796

                print(f"Current hinge joint position: {robot_config.data.joint('hinge_joint').qpos}")
                hinge_joint0 = robot_config.data.joint('hinge_joint').qpos[0]

                ur5_control.joint_control(detal_array=np.array([0, 0, 0, 0, 0, np.pi]))
                print(f"Current hinge joint position: {robot_config.data.joint('hinge_joint').qpos}")
                hinge_joint1 = robot_config.data.joint('hinge_joint').qpos[0]

                if abs(hinge_joint1 - hinge_joint0) < 0.6:
                    robot_config.model.actuator("fingers_actuator").forcerange[1] += 4

                robot_config.model.joint("slide_joint").damping[0] -= abs(hinge_joint1 - hinge_joint0)*50
                # print(robot_config.model.joint("slide_joint").damping[0])
                ur5_control.move_ee2(pos=[-0.135, 0.495, 0.26])
                print(f'Current grasp Z positon: {robot_config.data.body("grape_0").xpos[2]}')

            #
            flag = False

        # 在窗口中绘制传感器数据
        interface.viewer.render()


finally:
    # stop and reset the simulation
    interface.disconnect()
    # plot_process.terminate()
    print("Simulation terminated...")


