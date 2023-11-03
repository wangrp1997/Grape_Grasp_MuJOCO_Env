import math
import time

from abr_control.utils import transformations
from abr_control.controllers import path_planners
import numpy as np
import mujoco
class UR5Control:
    def __init__(self, interface, robot_config, n_timesteps):
        self.path_planner = path_planners.InverseKinematics(robot_config)
        self.interface = interface
        self.robot_config = robot_config
        self.n_timesteps = n_timesteps
        self.tolerance = 0.01
        self.gripper_ctrl_LIMIT = self.robot_config.model.actuator("fingers_actuator").ctrlrange[1]


    def move_ee1(self,detal):
        feedback = self.interface.get_feedback()
        Tx = self.robot_config.Tx("EE", q=feedback["q"], object_type="body")
        target_xyz = np.array([Tx[0]+detal[0], Tx[1]+detal[1], Tx[2]+detal[2]])
        R = self.robot_config.R("EE", q=feedback["q"])
        target_orientation = transformations.euler_from_matrix(R, "sxyz")
        # print(target_orientation)
        # update the position of the target
        # self.interface.set_mocap_xyz("target", target_xyz)
        # can use 3 different methods to calculate inverse kinematics
        # see inverse_kinematics.py file for details
        self.path_planner.generate_path(
            position=feedback["q"],
            target_position=np.hstack([target_xyz, target_orientation]),
            method=2,
            dt=0.001,
            n_timesteps=self.n_timesteps,
            plot=False, )

        # returns desired [position, velocity]
        while self.path_planner.n < self.path_planner.n_timesteps:
            target = self.path_planner.next()[0]
            # use position control
            self.interface.set_target_ctrl(target[: self.robot_config.N_JOINTS])
            self.interface.viewer.render()

    def move_ee2(self, pos):
        feedback = self.interface.get_feedback()
        target_xyz = np.array([pos[0], pos[1], pos[2]])
        R = self.robot_config.R("EE", q=feedback["q"])
        target_orientation = transformations.euler_from_matrix(R, "sxyz")
        # update the position of the target
        # self.interface.set_mocap_xyz("target", target_xyz)
        # can use 3 different methods to calculate inverse kinematics
        # see inverse_kinematics.py file for details
        self.path_planner.generate_path(
            position=feedback["q"],
            target_position=np.hstack([target_xyz, target_orientation]),
            method=2,
            dt=0.001,
            n_timesteps=self.n_timesteps,
            plot=False, )

        # returns desired [position, velocity]
        while self.path_planner.n < self.path_planner.n_timesteps:
            target = self.path_planner.next()[0]
            # use position control
            self.interface.set_target_ctrl(target[: self.robot_config.N_JOINTS])
            self.interface.viewer.render()

    def joint_control(self, detal_array):
        current_q = self.interface.get_feedback()['q'].copy()
        new_q = current_q + detal_array
        while 1:
            self.interface.set_target_ctrl(new_q)
            self.interface.viewer.render()
            error = abs(self.interface.get_feedback()['q'].copy() - new_q)
            # print(max(abs(error)))
            if max(error) < 0.011:
                break

    def close_gripper(self, current_target_joint_values):
        # self.model.actuator("fingers_actuator").forcerange[1] += 1
        # self.robot_config.data.qpos[6] = 1
        # 完全松开 [0. 0. 0. 0. 0. 0. 0. 0.]
        # 完全闭合 [0.79642269  0.0043969   0.79481524 - 0.78996885  0.79642246  0.00437121
        #  0.79478977 - 0.78995321]
        target_ctrl = math.ceil(current_target_joint_values*self.gripper_ctrl_LIMIT/0.796)
        prev_time = time.time()
        while 1:
            current_joint_values = self.robot_config.data.qpos[6]
            deltas = abs(current_target_joint_values - current_joint_values)
            if deltas < self.tolerance or time.time() - prev_time > 2:
                self.robot_config.data.ctrl[6] = math.ceil(current_joint_values*self.gripper_ctrl_LIMIT/0.796)
                mujoco.mj_forward(self.robot_config.model, self.robot_config.data)
                self.interface.viewer.render()
                break
            # self.robot_config.data.actuator("fingers_actuator").ctrl += 1
            self.robot_config.data.ctrl[6] = target_ctrl

            mujoco.mj_forward(self.robot_config.model, self.robot_config.data)
            self.interface.viewer.render()
