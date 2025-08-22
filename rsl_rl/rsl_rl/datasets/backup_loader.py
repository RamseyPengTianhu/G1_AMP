import os
import glob
import json
import joblib
import pickle



import logging

import torch
import numpy as np
from pybullet_utils import transformations

from rsl_rl.utils import utils
from rsl_rl.datasets import pose3d
from rsl_rl.datasets import motion_util

from rsl_rl.utils.motion_lib.skeleton import SkeletonTree
from rsl_rl.utils.motion_lib.motion_lib_robot import MotionLibRobot




class AMPLoader:

    def set_data_index(self):
        """Set data index for the motion data."""
        # Constants for indexing into the motion data - specific to 36-value format
        # 3 + 4 + num_dofs + 3 + 3 + num_dofs = 55

        if self.datatype == "Joint":
            self.JOINT_POS_SIZE = len(self.selected_joint_indices) if self.selected_joint_indices else 29
        else:
            self.JOINT_POS_SIZE = 10 * 3 # 10 joints, each with 3 values (x, y, z)

        self.JOINT_VEL_SIZE = self.JOINT_POS_SIZE

        # Sizes of each component
        self.POS_SIZE = 3
        self.ROT_SIZE = 4
    
        # Derived velocities - these will be computed
        self.LINEAR_VEL_SIZE = 3
        self.ANGULAR_VEL_SIZE = 3

        # self.KEY_POINT_POS_SIZE = 3 * 4
        # self.KEY_POINT_QUAT_SIZE = 4 * 4


        self.KEY_POINT_POS_SIZE = 3 * 30
        self.KEY_POINT_QUAT_SIZE = 0

        self.ROOT_POS_START_IDX = 0
        self.ROOT_POS_END_IDX = self.ROOT_POS_START_IDX + self.POS_SIZE # 3

        self.ROOT_ROT_START_IDX = self.ROOT_POS_END_IDX
        self.ROOT_ROT_END_IDX = self.ROOT_ROT_START_IDX + self.ROT_SIZE # 7

        self.JOINT_POS_START_IDX = self.ROOT_ROT_END_IDX
        self.JOINT_POS_END_IDX = self.JOINT_POS_START_IDX + self.JOINT_POS_SIZE # 7 + num_dofs

        self.LINEAR_VEL_START_IDX = self.JOINT_POS_END_IDX
        self.LINEAR_VEL_END_IDX = self.LINEAR_VEL_START_IDX + self.LINEAR_VEL_SIZE # 10 + num_dofs

        self.ANGULAR_VEL_START_IDX = self.LINEAR_VEL_END_IDX
        self.ANGULAR_VEL_END_IDX = self.ANGULAR_VEL_START_IDX + self.ANGULAR_VEL_SIZE # 13 + num_dofs

        self.JOINT_VEL_START_IDX = self.ANGULAR_VEL_END_IDX
        self.JOINT_VEL_END_IDX = self.JOINT_VEL_START_IDX + self.JOINT_VEL_SIZE # 13 + 2 * num_dofs

        self.KEY_POINT_POS_START_IDX = self.JOINT_VEL_END_IDX
        self.KEY_POINT_POS_END_IDX = self.KEY_POINT_POS_START_IDX + self.KEY_POINT_POS_SIZE
        print(f"Key point position indices: {self.KEY_POINT_POS_START_IDX} to {self.KEY_POINT_POS_END_IDX}")

        self.KEY_POINT_QUAT_START_IDX = self.KEY_POINT_POS_END_IDX
        self.KEY_POINT_QUAT_END_IDX = self.KEY_POINT_QUAT_START_IDX + self.KEY_POINT_QUAT_SIZE
        print(f"Key point quaternion indices: {self.KEY_POINT_QUAT_START_IDX} to {self.KEY_POINT_QUAT_END_IDX}")

    def __init__(
            self,
            device,
            time_between_frames,
            data_dir='',
            preload_transitions=False,
            num_preload_transitions=1000000,
            motion_files=glob.glob('datasets/motion_files2/*'),
            selected_joint_indices=None,  # 新增参数：你想要保留的关节索引列表
            datatype='Joint',
            ):
        """Expert dataset provides AMP observations from motion dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.selected_joint_indices = selected_joint_indices
        self.num_dofs = len(selected_joint_indices)
        self.datatype = datatype
        self.set_data_index()
        
        # Values to store for each trajectory
        self.trajectories = []
        self.extended_traj = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []
        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split('.')[0])
            
            motion_data = None
            key_pos_data = None
            frame_duration = 1.0 / 30.0
            motion_weight = 1.0

            if motion_file.endswith('.pkl'):
                print(f"Loading motion from pickle: {motion_file}")
                with open(motion_file, "rb") as f:
                    data = joblib.load(f)
                    # print('data:',data)
                    print("Available keys:", list(data.keys()))
                    print('dof_pos.shape:', data["dof_pos"].shape)
                    print('root_pos.shape:', data["root_pos"].shape)
                    print('root_rot.shape:', data["root_rot"].shape)
                    print('keypoints.shape:', data["local_body_pos"].shape)
                    print('link_body_list:', data["link_body_list"])

                    if "root_pos" in data and "root_rot" in data and "dof_pos" in data:
                        # ====== Your Robot Motion Format ======
                        root_pos = data["root_pos"]
                        # root_rot = data["root_rot"][:, [3, 0, 1, 2]]  # xyzw -> wxyz
                        root_rot = data["root_rot"]  # xyzw -> wxyz
                        dof_pos = data["dof_pos"]
                        keypoints = data["local_body_pos"]  # ✅ 你需要加这一句，或其他类似名字（要确保存在）
                        link_body_list = data["link_body_list"]

                        excluded_links = ["left_toe_link", "right_toe_link", "head_mocap", "imu_in_torso", 'right_rubber_hand', 'left_rubber_hand', 'pelvis_contour_link', 'head_link']

                        # 找出要保留的索引
                        included_indices = [i for i, name in enumerate(link_body_list) if name not in excluded_links]
                        filtered_keypoints = keypoints[:, included_indices, :]
                        filtered_link_body_list = [link_body_list[i] for i in included_indices]

                        # ==== Step 3: replace
                        keypoints = filtered_keypoints
                        keypoints = keypoints.reshape(keypoints.shape[0], -1)

                        link_body_list = filtered_link_body_list

                        fps = data.get("fps", 30.0)
                        T = len(root_pos)
                        motion_data = []
                        root_data = []
                        for t in range(T):
                            frame = []
                            # root_pos_frame = []
                            # root_rot_frame
                            frame.extend(root_pos[t])
                            frame.extend(root_rot[t])
                            for j in self.selected_joint_indices:
                                frame.append(dof_pos[t][j])
                            root_data.append(frame)
                            motion_data.append(frame)
                            
                        root_data = np.array(motion_data)
                        motion_data = np.array(motion_data)
                        frame_duration = 1.0 / fps
                        motion_weight = 1.0
                        key_pos_data = np.zeros((T, self.KEY_POINT_POS_SIZE))
                    else:
                        raise ValueError(f"Unsupported .pkl format: {motion_file}")

            elif motion_file.endswith('.json'):
                with open(motion_file, "r") as f:
                    try:
                        motion_json = json.load(f)
                        motion_data = np.array(motion_json["Frames"])
                        key_pos_data = np.array(motion_json.get("Keypoints", np.zeros((len(motion_data), self.KEY_POINT_POS_SIZE + self.KEY_POINT_QUAT_SIZE))))
                        frame_duration = float(motion_json.get("FrameDuration", 1.0/30.0))
                        motion_weight = float(motion_json.get("MotionWeight", 1.0))
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid JSON motion file: {motion_file}")

            elif motion_file.endswith('.csv'):
                with open(motion_file, "r") as f:
                    motion_data = []
                    key_pos_data = []
                    for line in f:
                        values = [float(x) for x in line.strip().split(',')]
                        if len(values) == 36 + self.KEY_POINT_POS_SIZE + self.KEY_POINT_QUAT_SIZE:
                            motion_data.append(values[:7])
                            for j in self.selected_joint_indices:
                                motion_data[-1].append(values[j + 7])
                            key_pos_data.append(values[36:])
                    motion_data = np.array(motion_data)
                    key_pos_data = np.array(key_pos_data)
                    frame_duration = 1.0 / 30.0
                    motion_weight = 1.0

            elif motion_file.endswith('.txt'):
                with open(motion_file, "r") as f:
                    next(f)  # skip header
                    lines = f.readlines()
                    motion_data = []
                    for line in lines:
                        values = [float(x) for x in line.strip().split(',')]
                        motion_data.append(values)
                    motion_data = np.array(motion_data)
                    key_pos_data = np.zeros((motion_data.shape[0], self.KEY_POINT_POS_SIZE + self.KEY_POINT_QUAT_SIZE))
                    frame_duration = 1.0 / 30.0
                    motion_weight = 1.0

            else:
                raise ValueError(f"Unsupported motion file format: {motion_file}")

            print('self.datatype:',self.datatype)

            # Dispatch to processing function
            # if self.datatype == "Joint":
            #     print('joint')
            #     self.get_joint_data(motion_data, key_pos_data, device, frame_duration, motion_weight, motion_file, i)
            # elif self.datatype == "Cartesian":
            #     print('Cartesian')

            #     self.get_cartesian_data(motion_data, key_pos_data, device, frame_duration, motion_weight, motion_file, i)
            #     self.get_joint_data(motion_data, key_pos_data, device, frame_duration, motion_weight, motion_file, i)
            print(f"[DEBUG] Adding motion file: {motion_file}")
            print(f"[DEBUG] motion_data.shape = {motion_data.shape}")
            print(f"[DEBUG] key_pos_data.shape = {keypoints.shape}")

            before_len = len(self.trajectory_weights)
            self.get_combined_motion_data(
                motion_data=motion_data,
                key_pos_data=keypoints,
                device=device,
                frame_duration=frame_duration,
                motion_weight=motion_weight,
                motion_file=motion_file,
                i=i
            )

            after_len = len(self.trajectory_weights)
            if after_len == before_len:
                print(f"[WARNING] motion file {motion_file} was ignored by get_combined_motion_data()")


        print('self.datatype:',self.datatype)
        print('self.trajectory_weights:',self.trajectory_weights)
        # Handle empty trajectory case
        if not self.trajectory_weights:
            raise ValueError("No valid motion files were loaded")
        
        # 以时间长度为权重进行归一化
        for i in range(len(self.trajectory_weights)):
            if self.trajectory_lens[i] <= 0:
                raise ValueError(f"Trajectory {self.trajectory_names[i]} has non-positive length: {self.trajectory_lens[i]}")
            self.trajectory_weights[i] *= self.trajectory_lens[i]
            
        # Trajectory weights are used to sample some trajectories more than others
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions
        self.preload_transitions = preload_transitions # True
        if self.preload_transitions:
            print(f'Preloading {num_preload_transitions} transitions')
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print('self.preloaded_s.shape:',self.preloaded_s.shape)
            print('self.preloaded_s_next.shape:',self.preloaded_s_next.shape)
            
            print(f'Finished preloading')

        self.all_trajectories_full = torch.vstack(self.trajectories_full) if self.trajectories_full else torch.tensor([])
        print(f'trajectories shape: {self.trajectories[0].shape}')
        print(f'trajectories_full shape: {self.trajectories_full[0].shape}')
        print(f'all_trajectories_full shape: {self.all_trajectories_full.shape}')
    

    def get_joint_data(self, motion_data, key_pos_data, device, frame_duration, motion_weight, motion_file, i):
        # Normalize and standardize quaternions
        for f_i in range(motion_data.shape[0]):
            root_rot = self.get_root_rot(motion_data[f_i])
            root_rot = pose3d.QuaternionNormalize(root_rot)
            root_rot = motion_util.standardize_quaternion(root_rot)
            motion_data[
                f_i,
                self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX] = root_rot
        
        # Compute velocities from position differences
        frame_rate = 30.0  # 30Hz
        dt = 1.0 / frame_rate
        
        # Only compute velocities if we have at least 2 frames
        if motion_data.shape[0] > 1:
            # Compute linear velocities (root position)
            lin_vel = np.zeros((motion_data.shape[0], self.LINEAR_VEL_SIZE))
            # Skip first frame for velocities since we need two frames to compute
            for f_i in range(1, motion_data.shape[0]):
                pos_curr = self.get_root_pos(motion_data[f_i])
                pos_prev = self.get_root_pos(motion_data[f_i-1])
                lin_vel[f_i] = (pos_curr - pos_prev) / dt
            
            # Compute angular velocities (from quaternions)
            ang_vel = np.zeros((motion_data.shape[0], self.ANGULAR_VEL_SIZE))
            for f_i in range(1, motion_data.shape[0]):
                quat_curr = self.get_root_rot(motion_data[f_i])
                quat_prev = self.get_root_rot(motion_data[f_i-1])

                # Get the relative rotation between frames
                quat_diff = transformations.quaternion_multiply(
                    quat_curr,
                    transformations.quaternion_inverse(quat_prev)
                )
                
                # Convert to axis-angle representation
                axis, angle = pose3d.QuaternionToAxisAngle(quat_diff)

                # Ensure angle is in the range [0, pi]
                if angle > np.pi:
                    angle = 2 * np.pi - angle
                    axis = -axis
                
                # Angular velocity is axis * angle / dt
                ang_vel[f_i] = axis * angle / dt
                ang_vel[f_i] = self.world_to_body(ang_vel[f_i], quat_prev)
                lin_vel[f_i] = self.world_to_body(lin_vel[f_i], quat_prev)
            
            # Compute joint velocities
            joint_vel = np.zeros((motion_data.shape[0], self.JOINT_VEL_SIZE))
            for f_i in range(1, motion_data.shape[0]):
                joint_curr = self.get_joint_pose(motion_data[f_i])
                joint_prev = self.get_joint_pose(motion_data[f_i-1])
                joint_vel[f_i] = (joint_curr - joint_prev) / dt
            
            # First frame velocities are the same as second frame to avoid zeros
            lin_vel[0] = lin_vel[1]
            ang_vel[0] = ang_vel[1]
            joint_vel[0] = joint_vel[1]
        else:
            # If only one frame, all velocities are zero
            lin_vel = np.zeros((motion_data.shape[0], self.LINEAR_VEL_SIZE))
            ang_vel = np.zeros((motion_data.shape[0], self.ANGULAR_VEL_SIZE))
            joint_vel = np.zeros((motion_data.shape[0], self.JOINT_VEL_SIZE))
        
        # Store all computed velocities as properties in the class
        self.lin_vel = torch.tensor(lin_vel, dtype=torch.float32, device=device)
        self.ang_vel = torch.tensor(ang_vel, dtype=torch.float32, device=device)
        # if self.selected_joint_indices is not None:
        #     joint_vel = joint_vel[:, self.selected_joint_indices]

        self.joint_vel = torch.tensor(joint_vel, dtype=torch.float32, device=device)

        
        # Store trajectory data (without the first 7 dimensions for regular traj)
        # if self.selected_joint_indices is not None:
        #     joint_data = motion_data[:, self.JOINT_POS_START_IDX:self.JOINT_POS_END_IDX]
        #     joint_data = joint_data[:, self.selected_joint_indices]  # 只选中指定索引
        # else:
        # joint_data = motion_data[:, self.JOINT_POS_START_IDX:self.JOINT_POS_END_IDX]

        # self.trajectories.append(torch.tensor(
        #     motion_data[:, self.JOINT_POS_START_IDX:self.JOINT_POS_END_IDX],
        #     dtype=torch.float32, device=device))

        self.trajectories.append(torch.tensor(
            motion_data[:, self.JOINT_POS_START_IDX:self.JOINT_POS_END_IDX],
            dtype=torch.float32, device=device))

        
        # Store full trajectory data with velocities
        # Create an extended motion data array that includes original data and computed velocities
        extended_motion_data = np.zeros((motion_data.shape[0], motion_data.shape[1] + self.LINEAR_VEL_SIZE + self.ANGULAR_VEL_SIZE + self.JOINT_VEL_SIZE + key_pos_data.shape[1]))
        extended_motion_data[:, :motion_data.shape[1]] = motion_data  # Original data
        # Fill in the rest of the extended motion data
        print(f"Extended motion data shape: {extended_motion_data.shape}")
        # Add velocities
        extended_motion_data[:, self.LINEAR_VEL_START_IDX:self.LINEAR_VEL_END_IDX] = lin_vel
        extended_motion_data[:, self.ANGULAR_VEL_START_IDX:self.ANGULAR_VEL_END_IDX] = ang_vel
        extended_motion_data[:, self.JOINT_VEL_START_IDX:self.JOINT_VEL_END_IDX] = joint_vel
        # Add key point positions and orientations to the end of the extended motion data
        extended_motion_data[:, self.KEY_POINT_POS_START_IDX:self.KEY_POINT_QUAT_END_IDX] = key_pos_data

        self.extended_traj.append(torch.tensor(
            extended_motion_data[:, self.JOINT_POS_START_IDX:],
            dtype=torch.float32, device=device))
        # Store extended data as tensor
        self.trajectories_full.append(torch.tensor(
            extended_motion_data,
            dtype=torch.float32, device=device))
        self.trajectory_idxs.append(i)
        self.trajectory_weights.append(motion_weight)
        self.trajectory_frame_durations.append(frame_duration)
        traj_len = (motion_data.shape[0] - 1) * frame_duration
        self.trajectory_lens.append(traj_len)
        self.trajectory_num_frames.append(float(motion_data.shape[0]))

        print(f"Loaded {traj_len}s motion from {motion_file}.")

    def get_cartesian_data(self, motion_data, key_pos_data, device, frame_duration, motion_weight, motion_file, i):
        """Process motion data in Cartesian format."""
        # Normalize and standardize quaternions
        for f_i in range(motion_data.shape[0]):
            root_rot = self.get_root_rot(motion_data[f_i])
            root_rot = pose3d.QuaternionNormalize(root_rot)
            root_rot = motion_util.standardize_quaternion(root_rot)
            motion_data[
                f_i,
                self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX] = root_rot
            
        # Compute velocities from position differences
        frame_rate = 30.0  # 30Hz
        dt = 1.0 / frame_rate
        # Only compute velocities if we have at least 2 frames
        if motion_data.shape[0] > 1:
            # Compute linear velocities (root position)
            lin_vel = np.zeros((motion_data.shape[0], self.LINEAR_VEL_SIZE))
            # Skip first frame for velocities since we need two frames to compute
            for f_i in range(1, motion_data.shape[0]):
                pos_curr = self.get_root_pos(motion_data[f_i])
                pos_prev = self.get_root_pos(motion_data[f_i-1])
                
                lin_vel[f_i] = (pos_curr - pos_prev) / dt
            
            # Compute angular velocities (from quaternions)
            ang_vel = np.zeros((motion_data.shape[0], self.ANGULAR_VEL_SIZE))
            torso_ang_vel = np.zeros((motion_data.shape[0], 3))  # For torso angular velocity
            for f_i in range(1, motion_data.shape[0]):
                quat_curr = self.get_root_rot(motion_data[f_i])
                quat_prev = self.get_root_rot(motion_data[f_i-1])
                quat_torso_curr = self.get_torso_pose(motion_data[f_i])
                quat_torso_prev = self.get_torso_pose(motion_data[f_i-1])
                
                # Get the relative rotation between frames
                quat_diff = transformations.quaternion_multiply(
                    quat_curr,
                    transformations.quaternion_inverse(quat_prev)
                )
                # Also compute torso angular velocity
                torso_quat_diff = transformations.quaternion_multiply(
                    quat_torso_curr,
                    transformations.quaternion_inverse(quat_torso_prev)
                )
                
                # Convert to axis-angle representation
                axis, angle = pose3d.QuaternionToAxisAngle(quat_diff)
                torso_axis, torso_angle = pose3d.QuaternionToAxisAngle(torso_quat_diff)

                # Ensure angle is in the range [0, pi]
                if angle > np.pi:
                    angle = 2 * np.pi - angle
                    axis = -axis
                if torso_angle > np.pi:
                    torso_angle = 2 * np.pi - torso_angle
                    torso_axis = -torso_axis
                
                # Angular velocity is axis * angle / dt
                ang_vel[f_i] = axis * angle / dt
                torso_ang_vel[f_i] = torso_axis * torso_angle / dt
            
            # Compute joint velocities
            joint_vel = np.zeros((motion_data.shape[0], self.JOINT_VEL_SIZE))
            for f_i in range(1, motion_data.shape[0]):
                joint_curr = self.get_joint_pose(motion_data[f_i])
                joint_prev = self.get_joint_pose(motion_data[f_i-1])
                joint_vel[f_i] = (joint_curr - joint_prev) / dt
            
            # First frame velocities are the same as second frame to avoid zeros
            lin_vel[0] = lin_vel[1]
            ang_vel[0] = ang_vel[1]
            joint_vel[0] = joint_vel[1]
        else:
            # If only one frame, all velocities are zero
            lin_vel = np.zeros((motion_data.shape[0], self.LINEAR_VEL_SIZE))
            ang_vel = np.zeros((motion_data.shape[0], self.ANGULAR_VEL_SIZE))
            joint_vel = np.zeros((motion_data.shape[0], self.JOINT_VEL_SIZE))
        
        # Store all computed velocities as properties in the class
        self.lin_vel = torch.tensor(lin_vel, dtype=torch.float32, device=device)
        self.ang_vel = torch.tensor(ang_vel, dtype=torch.float32, device=device)
        # if self.selected_joint_indices is not None:
        #     joint_vel = joint_vel[:, self.selected_joint_indices]

        self.joint_vel = torch.tensor(joint_vel, dtype=torch.float32, device=device)

        
        # Store trajectory data (without the first 7 dimensions for regular traj)
        # if self.selected_joint_indices is not None:
        #     joint_data = motion_data[:, self.JOINT_POS_START_IDX:self.JOINT_POS_END_IDX]
        #     joint_data = joint_data[:, self.selected_joint_indices]  # 只选中指定索引
        # else:
        # joint_data = motion_data[:, self.JOINT_POS_START_IDX:self.JOINT_POS_END_IDX]

        self.trajectories.append(torch.tensor(
            motion_data[:, self.JOINT_POS_START_IDX:self.TORSO_VEL_END_IDX],
            dtype=torch.float32, device=device))
    
        # Store full trajectory data with velocities
        # Create an extended motion data array that includes original data and computed velocities
        extended_motion_data = np.zeros((motion_data.shape[0], motion_data.shape[1] + self.LINEAR_VEL_SIZE + self.ANGULAR_VEL_SIZE + self.JOINT_VEL_SIZE + self.KEY_POINT_POS_SIZE + self.KEY_POINT_QUAT_SIZE))

        extended_motion_data[:, :motion_data.shape[1]] = motion_data  # Original data
        # Fill in the rest of the extended motion data
        
        # Add velocities
        extended_motion_data[:, self.LINEAR_VEL_START_IDX:self.LINEAR_VEL_END_IDX] = lin_vel
        extended_motion_data[:, self.ANGULAR_VEL_START_IDX:self.ANGULAR_VEL_END_IDX] = ang_vel
        extended_motion_data[:, self.JOINT_VEL_START_IDX:self.JOINT_VEL_END_IDX] = joint_vel

        # Add key point positions and orientations to the end of the extended motion data
        extended_motion_data[:, self.KEY_POINT_POS_START_IDX:self.KEY_POINT_POS_END_IDX] = motion_data[:, :self.KEY_POINT_QUAT_SIZE + self.KEY_POINT_POS_SIZE]
        self.extended_traj.append(torch.tensor(
            extended_motion_data[:, self.JOINT_POS_START_IDX:],
            dtype=torch.float32, device=device))
        # Store extended data as tensor
        self.trajectories_full.append(torch.tensor(
            extended_motion_data,
            dtype=torch.float32, device=device))
        self.trajectory_idxs.append(i)
        self.trajectory_weights.append(motion_weight)
        self.trajectory_frame_durations.append(frame_duration)
        traj_len = (motion_data.shape[0] - 1) * frame_duration
        self.trajectory_lens.append(traj_len)
        self.trajectory_num_frames.append(float(motion_data.shape[0]))

        print(f"Loaded {traj_len}s motion from {motion_file}.")



    def get_combined_motion_data(self, motion_data, key_pos_data, device, frame_duration, motion_weight, motion_file, i):
        """统一版本，合并了 joint data 和 cartesian data 的处理逻辑。"""
        print('motion_data.shape:',motion_data.shape)
        # 标准化 root 旋转
        for f_i in range(motion_data.shape[0]):
            root_rot = self.get_root_rot(motion_data[f_i])
            root_rot = pose3d.QuaternionNormalize(root_rot)
            root_rot = motion_util.standardize_quaternion(root_rot)
            motion_data[f_i, self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX] = root_rot
        # Velocity 计算准备
        frame_rate = 30.0
        dt = 1.0 / frame_rate
        N = motion_data.shape[0]

        lin_vel = np.zeros((N, self.LINEAR_VEL_SIZE))
        ang_vel = np.zeros((N, self.ANGULAR_VEL_SIZE))
        joint_vel = np.zeros((N, self.JOINT_VEL_SIZE))
        torso_ang_vel = np.zeros((N, 3))

        if N > 1:
            for f_i in range(1, N):
                pos_curr = self.get_root_pos(motion_data[f_i])
                pos_prev = self.get_root_pos(motion_data[f_i - 1])
                lin_vel[f_i] = (pos_curr - pos_prev) / dt

                quat_curr = self.get_root_rot(motion_data[f_i])
                quat_prev = self.get_root_rot(motion_data[f_i - 1])
                quat_diff = transformations.quaternion_multiply(
                    quat_curr, transformations.quaternion_inverse(quat_prev)
                )
                axis, angle = pose3d.QuaternionToAxisAngle(quat_diff)
                if angle > np.pi:
                    angle = 2 * np.pi - angle
                    axis = -axis
                ang_vel[f_i] = axis * angle / dt

                # torso angular velocity（cartesian 特有）
                if hasattr(self, "get_torso_pose"):
                    quat_torso_curr = self.get_torso_pose(root_data[f_i])
                    quat_torso_prev = self.get_torso_pose(root_data[f_i - 1])
                    torso_diff = transformations.quaternion_multiply(
                        quat_torso_curr, transformations.quaternion_inverse(quat_torso_prev)
                    )
                    t_axis, t_angle = pose3d.QuaternionToAxisAngle(torso_diff)
                    if t_angle > np.pi:
                        t_angle = 2 * np.pi - t_angle
                        t_axis = -t_axis
                    torso_ang_vel[f_i] = t_axis * t_angle / dt

                # joint velocity（joint 特有）
                joint_curr = self.get_joint_pose(motion_data[f_i])
                joint_prev = self.get_joint_pose(motion_data[f_i - 1])
                joint_vel[f_i] = (joint_curr - joint_prev) / dt

            # 第一帧复制第二帧
            lin_vel[0] = lin_vel[1]
            ang_vel[0] = ang_vel[1]
            joint_vel[0] = joint_vel[1]
            if hasattr(self, "get_torso_pose"):
                torso_ang_vel[0] = torso_ang_vel[1]

        # 存入 tensor
        self.lin_vel = torch.tensor(lin_vel, dtype=torch.float32, device=device)
        self.ang_vel = torch.tensor(ang_vel, dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(joint_vel, dtype=torch.float32, device=device)

        # trajectory（含 torso_vel）
        joint_end = self.TORSO_VEL_END_IDX if hasattr(self, "get_torso_pose") else self.JOINT_POS_END_IDX
        self.trajectories.append(torch.tensor(
            motion_data[:, self.JOINT_POS_START_IDX:joint_end],
            dtype=torch.float32, device=device
        ))
        print('key_pos_data.shape:', key_pos_data.shape)
        # 扩展 motion 数据拼接
        # total_extra_dim = self.LINEAR_VEL_SIZE + self.ANGULAR_VEL_SIZE + self.JOINT_VEL_SIZE + key_pos_data.shape[1]
        # total_extra_dim = self.JOINT_VEL_SIZE
        # joint_extra_dim = self.JOINT_VEL_SIZE + self.KEY_POINT_POS_SIZE
        # root_extra_dim = self.POS_SIZE + self.ROT_SIZE + self.LINEAR_VEL_SIZE + self.ANGULAR_VEL_SIZE
        # extended_motion_data = np.zeros((N, motion_data.shape[1] + Joint_extra_dim + root_extra_dim))
        # extended_motion_data = np.zeros((N, motion_data.shape[1]))
        # extended_motion_data[:, :self.POS_SIZE] = self.root_pos
        # extended_motion_data[:, self.POS_SIZE:self.POS_SIZE + self.ROT_SIZE] = self.root_rot
        # extended_motion_data[:, self.POS_SIZE + self.ROT_SIZE:self.POS_SIZE + self.ROT_SIZE + self.JOINT_POS_SIZE] = motion_data
        # extended_motion_data[:, self.POS_SIZE + self.ROT_SIZE + self.JOINT_POS_SIZE:self.POS_SIZE + self.ROT_SIZE + self.JOINT_POS_SIZE + self.LINEAR_VEL_SIZE] = self.lin_vel
        # extended_motion_data[:, self.POS_SIZE + self.ROT_SIZE + self.JOINT_POS_SIZE + self.JOINT_POS_SIZE + self.LINEAR_VEL_SIZE:self.POS_SIZE + self.ROT_SIZE + self.JOINT_POS_SIZE + self.LINEAR_VEL_SIZE + self.ANGULAR_VEL_SIZE] = self.ang_vel
        # extended_motion_data[:, self.POS_SIZE + self.ROT_SIZE + self.JOINT_POS_SIZE + self.JOINT_POS_SIZE + self.LINEAR_VEL_SIZE + self.ANGULAR_VEL_SIZE:self.POS_SIZE + self.ROT_SIZE + self.JOINT_POS_SIZE + self.LINEAR_VEL_SIZE + self.ANGULAR_VEL_SIZE + self.JOINT_VEL_SIZE] = joint_vel
        # extended_motion_data[:, self.POS_SIZE + self.ROT_SIZE + self.JOINT_POS_SIZE + self.JOINT_POS_SIZE + self.LINEAR_VEL_SIZE + self.ANGULAR_VEL_SIZE + self.JOINT_VEL_SIZE:self.POS_SIZE + self.ROT_SIZE + self.JOINT_POS_SIZE + self.LINEAR_VEL_SIZE + self.ANGULAR_VEL_SIZE + self.JOINT_VEL_SIZE + self.KEY_POINT_POS_SIZE] = key_pos_data

        # extended_motion_data[:, :motion_data.shape[1]] = motion_data

        # extended_motion_data[:, motion_data.shape[1]:motion_data.shape[1] + self.JOINT_VEL_SIZE] = joint_vel
        # extended_motion_data[:, motion_data.shape[1] + self.JOINT_VEL_SIZE :motion_data.shape[1] + self.JOINT_VEL_SIZE + self.KEY_POINT_POS_SIZE] = key_pos_data


        # extended_motion_data[:, motion_data.shape[1] + self.JOINT_VEL_SIZE + self.KEY_POINT_POS_SIZE:motion_data.shape[1] + self.JOINT_VEL_SIZE + self.KEY_POINT_POS_SIZE + self.POS_SIZE] = self.root_pos
        # extended_motion_data[:, motion_data.shape[1] + self.JOINT_VEL_SIZE + self.KEY_POINT_POS_SIZE + self.POS_SIZE:motion_data.shape[1] + self.JOINT_VEL_SIZE + self.KEY_POINT_POS_SIZE + self.POS_SIZE + self.ROT_SIZE] = self.root_rot
        # extended_motion_data[:, motion_data.shape[1] + self.JOINT_VEL_SIZE + self.KEY_POINT_POS_SIZE + self.POS_SIZE + self.ROT_SIZE:motion_data.shape[1] + self.JOINT_VEL_SIZE + self.KEY_POINT_POS_SIZE + self.POS_SIZE + self.ROT_SIZE + self.LINEAR_VEL_SIZE] = self.lin_vel
        # extended_motion_data[:, motion_data.shape[1] + self.JOINT_VEL_SIZE + self.KEY_POINT_POS_SIZE + self.POS_SIZE + self.ROT_SIZE + self.LINEAR_VEL_SIZE:motion_data.shape[1] + self.JOINT_VEL_SIZE + self.KEY_POINT_POS_SIZE + self.POS_SIZE + self.ROT_SIZE + self.LINEAR_VEL_SIZE + self.LINEAR_VEL_SIZE] = self.ang_vel
        # extended_motion_data[:, self.ANGULAR_VEL_START_IDX:self.ANGULAR_VEL_END_IDX] = ang_vel
        # extended_motion_data[:, self.JOINT_VEL_START_IDX:self.JOINT_VEL_END_IDX] = joint_vel
        # extended_motion_data[:, self.KEY_POINT_POS_START_IDX:self.KEY_POINT_POS_END_IDX] = key_pos_data
        # print('extended_motion_data.shape:',extended_motion_data.shape)
        # self.extended_traj.append(torch.tensor(
        #     extended_motion_data[:, self.JOINT_POS_START_IDX:],
        #     dtype=torch.float32, device=device
        # ))
        extended_motion_data = np.zeros((motion_data.shape[0], motion_data.shape[1] + self.LINEAR_VEL_SIZE + self.ANGULAR_VEL_SIZE + self.JOINT_VEL_SIZE + key_pos_data.shape[1]))
        extended_motion_data[:, :motion_data.shape[1]] = motion_data  # Original data
        # Fill in the rest of the extended motion data
        print(f"Extended motion data shape: {extended_motion_data.shape}")
        # Add velocities
        extended_motion_data[:, self.LINEAR_VEL_START_IDX:self.LINEAR_VEL_END_IDX] = lin_vel
        extended_motion_data[:, self.ANGULAR_VEL_START_IDX:self.ANGULAR_VEL_END_IDX] = ang_vel
        extended_motion_data[:, self.JOINT_VEL_START_IDX:self.JOINT_VEL_END_IDX] = joint_vel
        # Add key point positions and orientations to the end of the extended motion data
        extended_motion_data[:, self.KEY_POINT_POS_START_IDX:self.KEY_POINT_QUAT_END_IDX] = key_pos_data

        self.extended_traj.append(torch.tensor(
            extended_motion_data[:, self.JOINT_POS_START_IDX:],
            dtype=torch.float32, device=device
        ))
        # print('extended_motion_data:',extended_motion_data.shape)
        self.trajectories_full.append(torch.tensor(extended_motion_data, dtype=torch.float32, device=device))

        self.trajectory_idxs.append(i)
        self.trajectory_weights.append(motion_weight)
        self.trajectory_frame_durations.append(frame_duration)
        traj_len = (N - 1) * frame_duration
        self.trajectory_lens.append(traj_len)
        self.trajectory_num_frames.append(float(N))

        print(f"Loaded {traj_len:.2f}s motion from {motion_file}.")

    def get_root_pos(self, pose):
        """Get root position from a pose vector."""
        return pose[self.ROOT_POS_START_IDX:self.ROOT_POS_END_IDX]

    def get_root_pos_batch(self, poses):
        """Get root positions from a batch of pose vectors."""
        return poses[:, self.ROOT_POS_START_IDX:self.ROOT_POS_END_IDX]

    def get_root_rot(self, pose):
        """Get root rotation from a pose vector."""
        return pose[self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX]

    def get_root_rot_batch(self, poses):
        """Get root rotations from a batch of pose vectors."""
        return poses[:, self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX]

    def get_joint_pose(self, pose):
        """Get joint poses from a pose vector."""
        # return pose[self.JOINT_POS_START_IDX:self.JOINT_POS_END_IDX]
        return pose[self.JOINT_POS_START_IDX:self.JOINT_POS_END_IDX]
        # return pose[0:self.JOINT_POS_SIZE]


    def get_joint_pose_batch(self, poses):
        """Get joint poses from a batch of pose vectors."""
        # print('JOINT_POS_START_IDX:',self.JOINT_POS_START_IDX)
        # print('JOINT_POS_END_IDX:',self.JOINT_POS_END_IDX)
        # print('poses.shape:',poses.shape)
        # print('poses[:, self.JOINT_POS_START_IDX:self.JOINT_POS_END_IDX].shape:',poses[:, self.JOINT_POS_START_IDX:self.JOINT_POS_END_IDX].shape)
        # print('poses[:, :].shape:',poses[:, :].shape)
        return poses[:, self.JOINT_POS_START_IDX:self.JOINT_POS_END_IDX]
        # return poses[:, :self.JOINT_POS_SIZE]
        
        # return poses[:, :]

    def get_linear_vel(self, pose):
        return pose[self.LINEAR_VEL_START_IDX:self.LINEAR_VEL_END_IDX]
    
    def get_linear_vel_batch(self, pose):
        return pose[:, self.LINEAR_VEL_START_IDX:self.LINEAR_VEL_END_IDX]
    
    def get_angular_vel(self, pose):
        return pose[self.ANGULAR_VEL_START_IDX:self.ANGULAR_VEL_END_IDX]  

    def get_angular_vel_batch(self, poses):
        return poses[:, self.ANGULAR_VEL_START_IDX:self.ANGULAR_VEL_END_IDX]
    
    def get_joint_vel(self, pose):
        # return pose[self.JOINT_POS_SIZE:self.JOINT_POS_SIZE + self.JOINT_VEL_SIZE]
        return pose[self.JOINT_VEL_START_IDX:self.JOINT_VEL_END_IDX]


    def get_joint_vel_batch(self, poses):
        return poses[:, self.JOINT_VEL_START_IDX:self.JOINT_VEL_END_IDX]
    
    def get_key_pos_batch(self, poses):
        return poses[:, self.KEY_POINT_POS_START_IDX:self.KEY_POINT_POS_END_IDX]
    
    def get_key_quat_batch(self, poses):
        return poses[:, self.KEY_POINT_QUAT_START_IDX:self.KEY_POINT_QUAT_END_IDX]

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(
            self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights,
            replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(
            0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        """Linear interpolation between values."""
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        idx_high = min(idx_high, n - 1)  # Ensure we don't go out of bounds
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / np.maximum(self.trajectory_lens[traj_idxs], 1e-10)  # Avoid division by zero
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        
        # Clamp indices to valid range
        idx_low = np.clip(idx_low, 0, n - 1)
        idx_high = np.clip(idx_high, 0, n - 1)
        
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        idx_high = min(idx_high, n - 1)  # Ensure we don't go out of bounds
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        """Returns full frames for the given trajectories at the specified times."""
        """找到时间点对应的各项观测量，并在低位高位之间进行插值，返回插值后观测量"""
        p = times / np.maximum(self.trajectory_lens[traj_idxs], 1e-10)  # Avoid division by zero
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        
        # Clamp indices to valid range
        idx_low = np.clip(idx_low, 0, n - 1)  
        idx_high = np.clip(idx_high, 0, n - 1)
        
        # Initialize tensors to hold the interpolation values
        # For positions and orientations
        all_frame_pos_starts = torch.zeros(len(traj_idxs), self.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), self.POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), self.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), self.ROT_SIZE, device=self.device)
        all_frame_joint_starts = torch.zeros(len(traj_idxs), self.JOINT_POS_SIZE, device=self.device)
        all_frame_joint_ends = torch.zeros(len(traj_idxs), self.JOINT_POS_SIZE, device=self.device)
        all_frame_key_pos_starts = torch.zeros(len(traj_idxs), self.KEY_POINT_POS_SIZE, device=self.device)
        all_frame_key_pos_ends = torch.zeros(len(traj_idxs), self.KEY_POINT_POS_SIZE, device=self.device)
        # all_frame_key_quat_starts = torch.zeros(len(traj_idxs), self.KEY_POINT_QUAT_SIZE, device=self.device)
        # all_frame_key_quat_ends = torch.zeros(len(traj_idxs), self.KEY_POINT_QUAT_SIZE, device=self.device)

        # For velocities
        all_frame_lin_vel_starts = torch.zeros(len(traj_idxs), self.LINEAR_VEL_SIZE, device=self.device)
        all_frame_lin_vel_ends = torch.zeros(len(traj_idxs), self.LINEAR_VEL_SIZE, device=self.device)
        all_frame_ang_vel_starts = torch.zeros(len(traj_idxs), self.ANGULAR_VEL_SIZE, device=self.device)
        all_frame_ang_vel_ends = torch.zeros(len(traj_idxs), self.ANGULAR_VEL_SIZE, device=self.device)
        all_frame_joint_vel_starts = torch.zeros(len(traj_idxs), self.JOINT_VEL_SIZE, device=self.device)
        all_frame_joint_vel_ends = torch.zeros(len(traj_idxs), self.JOINT_VEL_SIZE, device=self.device)

        
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            
            # Extract components for each trajectory's frames - positions and orientations
            all_frame_pos_starts[traj_mask] = self.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = self.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            
            all_frame_rot_starts[traj_mask] = self.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = self.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            # print('all_frame_joint_starts.shape:',all_frame_joint_starts.shape)
            # print('get_joint_pose_batch.shape:',self.get_joint_pose_batch(trajectory[idx_low[traj_mask]]).shape)
            all_frame_joint_starts[traj_mask] = self.get_joint_pose_batch(trajectory[idx_low[traj_mask]])
            all_frame_joint_ends[traj_mask] = self.get_joint_pose_batch(trajectory[idx_high[traj_mask]])

            # Extract velocity components
            all_frame_lin_vel_starts[traj_mask] = self.get_linear_vel_batch(trajectory[idx_low[traj_mask]])
            all_frame_lin_vel_ends[traj_mask] = self.get_linear_vel_batch(trajectory[idx_high[traj_mask]])
            
            all_frame_ang_vel_starts[traj_mask] = self.get_angular_vel_batch(trajectory[idx_low[traj_mask]])
            all_frame_ang_vel_ends[traj_mask] = self.get_angular_vel_batch(trajectory[idx_high[traj_mask]])
            
            all_frame_joint_vel_starts[traj_mask] = self.get_joint_vel_batch(trajectory[idx_low[traj_mask]])
            all_frame_joint_vel_ends[traj_mask] = self.get_joint_vel_batch(trajectory[idx_high[traj_mask]])

            # Extract key point positions and orientations
            all_frame_key_pos_starts[traj_mask] = self.get_key_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_key_pos_ends[traj_mask] = self.get_key_pos_batch(trajectory[idx_high[traj_mask]])

            all_frame_key_quat_starts[traj_mask] = self.get_key_quat_batch(trajectory[idx_low[traj_mask]])
            all_frame_key_quat_ends[traj_mask] = self.get_key_quat_batch(trajectory[idx_high[traj_mask]])

        
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        # Interpolate position linearly
        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        
        # Use quaternion interpolation for rotation
        rot_blend = utils.quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        
        # Interpolate joint positions and velocities linearly
        joint_blend = self.slerp(all_frame_joint_starts, all_frame_joint_ends, blend)

        lin_vel_blend = self.slerp(all_frame_lin_vel_starts, all_frame_lin_vel_ends, blend)
        ang_vel_blend = self.slerp(all_frame_ang_vel_starts, all_frame_ang_vel_ends, blend)
        joint_vel_blend = self.slerp(all_frame_joint_vel_starts, all_frame_joint_vel_ends, blend)

        # Key point positions and orientations
        key_pos_blend = self.slerp(all_frame_key_pos_starts, all_frame_key_pos_ends, blend)
        key_quat_blend = utils.quaternion_slerp(all_frame_key_quat_starts, all_frame_key_quat_ends, blend)
        
        return torch.cat([
            pos_blend, 
            rot_blend,
            joint_blend,
            lin_vel_blend, 
            ang_vel_blend, 
            joint_vel_blend,
            key_pos_blend,
            key_quat_blend
        ], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        """Returns a batch of random full frames."""
        if self.preload_transitions:
            idxs = np.random.choice(
                self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:

            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        # Get original frame data
        root_pos0, root_pos1 = self.get_root_pos(frame0), self.get_root_pos(frame1)
        root_rot0, root_rot1 = self.get_root_rot(frame0), self.get_root_rot(frame1)
        joints0, joints1 = self.get_joint_pose(frame0), self.get_joint_pose(frame1)
        
        # Get velocity data - offset by original data length
        offset = self.ROOT_POS_END_IDX + self.ROT_SIZE + self.JOINT_POS_SIZE
        
        # Extract velocities using tensor slicing
        lin_vel0 = frame0[offset:offset+self.LINEAR_VEL_SIZE]
        lin_vel1 = frame1[offset:offset+self.LINEAR_VEL_SIZE]
        
        offset += self.LINEAR_VEL_SIZE
        ang_vel0 = frame0[offset:offset+self.ANGULAR_VEL_SIZE]
        ang_vel1 = frame1[offset:offset+self.ANGULAR_VEL_SIZE]
        
        offset += self.ANGULAR_VEL_SIZE
        joint_vel0 = frame0[offset:offset+self.JOINT_VEL_SIZE]
        joint_vel1 = frame1[offset:offset+self.JOINT_VEL_SIZE]

        # Blend positions linearly
        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        
        # Use quaternion interpolation for rotation
        blend_root_rot = transformations.quaternion_slerp(
            root_rot0.cpu().numpy(), root_rot1.cpu().numpy(), blend)
        blend_root_rot = torch.tensor(
            motion_util.standardize_quaternion(blend_root_rot),
            dtype=torch.float32, device=self.device)
            
        # Blend joint positions and velocities linearly
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_lin_vel = self.slerp(lin_vel0, lin_vel1, blend)
        blend_ang_vel = self.slerp(ang_vel0, ang_vel1, blend)
        blend_joint_vel = self.slerp(joint_vel0, joint_vel1, blend)

        # Concatenate all blended components
        return torch.cat([
            blend_root_pos, 
            blend_root_rot, 
            blend_joints, 
            blend_lin_vel, 
            blend_ang_vel, 
            blend_joint_vel
        ])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(
                    self.preloaded_s.shape[0], size=mini_batch_size)
                
                # Get only the joint positions for the state
                s = self.preloaded_s[idxs, self.JOINT_POS_START_IDX:self.KEY_POINT_POS_END_IDX]

                # Add root height (Z coordinate)
                s = torch.cat([
                    s,
                    self.preloaded_s[idxs, self.ROOT_POS_START_IDX + 2:self.ROOT_POS_START_IDX + 3]], dim=-1)
                
                # Same for next state
                s_next = self.preloaded_s_next[idxs, self.JOINT_POS_START_IDX:self.KEY_POINT_POS_END_IDX]
                s_next = torch.cat([
                    s_next,
                    self.preloaded_s_next[idxs, self.ROOT_POS_START_IDX + 2:self.ROOT_POS_START_IDX + 3]], dim=-1)
              
            else:


                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                
                for traj_idx, frame_time in zip(traj_idxs, times):
                    frame = self.get_frame_at_time(traj_idx, frame_time)
                    next_frame = self.get_frame_at_time(traj_idx, frame_time + self.time_between_frames)
                    
                    # We need to get the root height for each frame
                    full_frame = self.get_full_frame_at_time(traj_idx, frame_time)
                    full_next_frame = self.get_full_frame_at_time(traj_idx, frame_time + self.time_between_frames)
                    
                    # Append joint positions and root height
                    s.append(torch.cat([frame, full_frame[self.ROOT_POS_START_IDX + 2:self.ROOT_POS_START_IDX + 3]]))
                    s_next.append(torch.cat([next_frame, full_next_frame[self.ROOT_POS_START_IDX + 2:self.ROOT_POS_START_IDX + 3]]))
                s = torch.stack(s)
                s_next = torch.stack(s_next)
            yield s, s_next

    def world_to_body(self, vec_w, quat_wb):
        """
        把世界系向量 vec_w (长度3) 旋到机体系。
        """
        # Pinocchio/pybullet 四元数顺序 = (x, y, z, w)；transformations 也如此
        q_bw = transformations.quaternion_inverse(quat_wb)          # q^{-1}
        vec_q = np.concatenate([vec_w, [0.0]])                       # (x,y,z,0)
        rotated = transformations.quaternion_multiply(
                    transformations.quaternion_multiply(q_bw, vec_q),
                    quat_wb)[:3]
        return rotated



    def load_robot_motion(self, motion_file):
        """
        Load robot motion data from a pickle file.
        """
        with open(motion_file, "rb") as f:
            motion_data = pickle.load(f)
            motion_fps = motion_data["fps"]
            motion_root_pos = motion_data["root_pos"]
            motion_root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]] # from xyzw to wxyz
            motion_dof_pos = motion_data["dof_pos"]
            motion_local_body_pos = motion_data["local_body_pos"]
            motion_link_body_list = motion_data["link_body_list"]
        return motion_data, motion_fps, motion_root_pos,motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list



    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.extended_traj[0].shape[1] + 1 # Joint positions + lin_vel + ang_vel+ root height