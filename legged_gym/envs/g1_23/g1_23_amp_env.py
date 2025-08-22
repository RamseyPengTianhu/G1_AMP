
from legged_gym.envs.base.g1_legged_robot import G1LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class G123AMPRobot(G1LeggedRobot):
    def compute_observations(self):
        """ Computes observations
        """
        _, _, yaw = get_euler_xyz(self.base_quat)

        # local offset 旋转到 world
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat = torch.stack([
            torch.stack([cos_yaw, sin_yaw], dim=1),
            torch.stack([-sin_yaw,  cos_yaw], dim=1)
        ], dim=1)  # [N, 2, 2]
        world_target_offset = self.target_pos[:, :2] - self.root_states[:, :2]  # [N, 2]
        local_target_offset = torch.bmm(rot_mat, world_target_offset.unsqueeze(-1)).squeeze(-1)  # [N, 2]
        local_target_offset = torch.cat((local_target_offset, (self.target_pos[:, 2] - self.root_states[:, 2]).unsqueeze(1)), dim=-1)  # [N, 3]
        hands_z = self.rigid_body_states[:, self.end_effector_index, 2]  # [num_envs, 2]
        # hands_z_mean = hands_z.mean(dim=1, keepdim=True)  # [num_envs, 1]


        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    # self.commands[:, :4] * self.commands_scale,
                                    # local_target_offset,
                                    self.command_height, 
                                    hands_z,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.privileged_obs_buf += (2 * torch.rand_like(self.privileged_obs_buf) - 1) * self.noise_scale_vec

        # Remove velocity observations from policy observation.
        if self.num_obs == self.num_privileged_obs - 3:
            self.obs_buf = self.privileged_obs_buf[:, 3:]
        else:
            self.obs_buf = torch.clone(self.privileged_obs_buf)


    def _reset_target_pos_box(self, env_ids=None):
        if env_ids is None or len(env_ids) == 0:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = len(env_ids)

        # -------- 1) 初始化 buffer --------
        if not hasattr(self, "command_height") or self.command_height.shape != (self.num_envs, 1):
            self.command_height = torch.zeros((self.num_envs, 1), device=self.device)

        if not hasattr(self, "has_hit"):
            self.has_hit = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        if not hasattr(self, "root_pos_start"):
            self.root_pos_start = torch.zeros((self.num_envs, 3), device=self.device)

        # -------- 2) 读高度范围 --------
        if hasattr(self, "command_ranges"):
            if "command_height" in self.command_ranges:
                h_min, h_max = self.command_ranges["command_height"]
            elif "target_z" in self.command_ranges:
                h_min, h_max = self.command_ranges["target_z"]
            elif "height_z" in self.command_ranges:
                h_min, h_max = self.command_ranges["height_z"]
            else:
                h_min, h_max = self.cfg.commands.ranges.target_z
        else:
            h_min, h_max = self.cfg.commands.ranges.target_z

        # -------- 3) 分情况设置高度 --------
        sine_mode = getattr(self, "_height_mode", None) == "sine"
        if sine_mode:
            # 正弦模式：保持 play.py 的实时更新值
            target_z = self.command_height[env_ids]
        else:
            # 固定/采样模式
            if abs(h_max - h_min) < 1e-8:
                target_z = torch.full((n, 1), float(h_min), device=self.device)
            else:
                target_z = torch.empty((n, 1), device=self.device).uniform_(float(h_min), float(h_max))

        # -------- 4) 写入 buffer --------
        self.command_height[env_ids] = target_z
        self.object_pos_z = self.command_height.clone()
        self.has_hit[env_ids] = False
        self.root_pos_start[env_ids] = self.root_states[env_ids, 0:3].detach()

        # Debug 打印
        # print("sine_mode:", sine_mode)
        # print("target_z:", target_z)
        # print("self.command_height[env_ids]:", self.command_height[env_ids])

    def _reset_target_pos(self, env_ids=None):
        # env_ids: 要重置 target 的环境编号, 支持 batch
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # 采样 base/root 当前 xy 坐标
        base_pos = self.root_states[env_ids, :2]  # [N, 2]
        _, _, yaw = get_euler_xyz(self.base_quat[env_ids])

        # 随机采样 target 的相对偏移（比如半径在 [0.5, 1.5] 米，角度 0-2pi）
        radius = torch.empty(len(env_ids), device=self.device).uniform_(self.cfg.commands.ranges.target_radius[0], self.cfg.commands.ranges.target_radius[1])
        theta = torch.empty(len(env_ids), device=self.device).uniform_(self.cfg.commands.ranges.target_theta[0], self.cfg.commands.ranges.target_theta[1])
        offset = torch.stack([
            radius * torch.cos(theta),
            radius * torch.sin(theta)
        ], dim=-1)  # [N, 2]

        # local offset 旋转到 world
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat = torch.stack([
            torch.stack([cos_yaw, -sin_yaw], dim=1),
            torch.stack([sin_yaw,  cos_yaw], dim=1)
        ], dim=1)  # [N, 2, 2]

        offset_world = torch.bmm(rot_mat, offset.unsqueeze(-1)).squeeze(-1)  # [N, 2]

        # target 在世界坐标系下
        target_xy = base_pos + offset_world  # [N, 2]

        # target_z = torch.full((len(env_ids), 1), self.cfg.commands.ranges.target_z, device=self.device)   # [N, 1]
        target_z = torch.empty(len(env_ids), device=self.device).uniform_(self.cfg.commands.ranges.target_z[0], self.cfg.commands.ranges.target_z[1]).unsqueeze(1)
        self.target_pos[env_ids] = torch.cat([target_xy, target_z], dim=-1)  # [N, 3]
        self.has_hit[env_ids] = False  # 重置 has_hit 状态



        
    def _reward_strike(self):
        # 1. 位置和速度
        x_star = self.target_pos                 # [num_envs, 3]
        x_root = self.root_states[:, :3]         # [num_envs, 3]
        x_eff = self.rigid_body_states[:, self.end_effector_index, 0:3]                # [num_envs, 3]
        x_eff_dot = self.rigid_body_states[:, self.end_effector_index, 7:10]            # [num_envs, 3]
        base_vel = self.root_states[:, 7: 10]

        # 2. 单位向量
        d_star = x_star - x_root
        d_star_unit = d_star / (torch.norm(d_star, dim=1, keepdim=True) + 1e-8)

        # 3. 距离
        eff_dist = torch.norm(x_star - x_eff, dim=1)
        root_dist = torch.norm(x_star - x_root, dim=1)

        # 5. r_near
        dot_v = torch.sum(d_star_unit * x_eff_dot, dim=1)
        r_near = 0.2 * torch.exp(-2 * eff_dist**2) + 0.8 * torch.clamp((2/3) * dot_v, 0, 1)

        # 6. r_far
        v_star=1.0           # 目标方向速度
        dot_v_base = torch.sum(d_star_unit * base_vel, dim=1)
        vel_term = torch.clamp(v_star - dot_v_base, min=0)
        r_far = 0.7 * torch.exp(-0.5 * root_dist ** 2) + 0.3 * torch.exp(-vel_term ** 2)

        # 7. reward 逻辑
        reward = torch.where(
            self.has_hit,
            torch.ones_like(root_dist),
            torch.where(
                root_dist < 1.375,
                0.3 * r_near + 0.3,
                0.3 * r_far
            )
        )

        return reward


    def _reward_twohands_to_height(self):
        """
        两手对齐目标高度（只track z）：距离_z + 同步 +（可选）根漂移。
        依赖：
        - self.object_pos_z: [B] 或 [B,1]，目标高度（世界系 z）
        - self.hand_indices: [idx_L, idx_R]
        - self.rigid_body_states: [..., 13]（第 3 个坐标是 z）
        - (可选) self.root_pos_start 用于根漂移罚
        """
        import torch
        B = self.root_states.shape[0]
        L, R = self.end_effector_index

        # 目标高度 z，统一成 [B,1]
        target_z = self.object_pos_z
        if target_z.ndim == 1:
            target_z = target_z.unsqueeze(1)                # [B,1]
        elif target_z.shape[1] != 1:
            target_z = target_z[:, :1]                      # 兜底取第一列

        # 两只手的当前高度 z -> [B,2]
        hand_z = torch.stack([
            self.rigid_body_states[:, L, 2],               # 左手 z
            self.rigid_body_states[:, R, 2],               # 右手 z
        ], dim=1)                                          # [B,2]

        # 与目标高度的绝对误差 -> [B,2]
        dz = torch.abs(hand_z - target_z)                  # [B,2]

        # 位置项（高斯）：越接近目标 z 越好
        sigma_z = 0.06                                     # 6cm
        r_pos_each = torch.exp(- (dz ** 2) / (2 * sigma_z ** 2))  # [B,2]
        r_pos = r_pos_each.mean(dim=1)                             # [B]

        # 双手同步（高度差越小越好）
        r_sync = torch.exp(- torch.abs(hand_z[:, 0] - hand_z[:, 1]) / 0.03)     # [B]

        # 命中保持：两手都在阈值内
        eps_z = 0.03                                 # 3cm
        hold = 0.5 * ((dz < eps_z).float().prod(dim=1))     # [B]

        # （可选）根漂移惩罚：只要你还想压制走动
        p_drift = 0.0
        if hasattr(self, "root_pos_start"):
            root_drift = torch.norm(self.root_states[:, 0:3] - self.root_pos_start, dim=1)
            p_drift = -0.5 * torch.clamp(root_drift - 0.03, min=0.0)

        reward = r_pos + 0.2 * r_sync + hold + p_drift
        return torch.clamp(reward, 0.0, 3.0)



    

    def _reward_minimize_torso_angular_velocity(self):
        """
        Penalize large angular velocities in the upper body to stabilize motion.

        Returns:
            torch.Tensor: The reward value for minimizing angular velocity.
        """
        _, pelvis_ang_vel, waist_ang_vel, torso_ang_vel = self._extract_upper_body_angular_velocity()


        # Compute penalty for angular velocity magnitude
        angular_velocity_penalty = torch.norm(torso_ang_vel, dim=1)  # Shape: [num_envs]

        # Reward is inversely proportional to the penalty
        reward = torch.exp(-angular_velocity_penalty)  # Penalize high angular velocity
        return reward
    
    def _reward_minimize_waist_pitch_deviation(self, target_pitch=0.0, weight=1.0, log=False):
        """
        Penalize deviation of waist_pitch_joint from a target angle.

        Args:
            target_pitch (float): Desired waist pitch angle in radians (default: 0.0).
            weight (float): Scaling factor for the reward (default: 1.0).
            log (bool): Whether to log the deviation (default: False).

        Returns:
            torch.Tensor: Reward value for minimizing waist pitch deviation.
        """
        self.waist_pitch_index = self.dof_names.index('waist_pitch_joint')  # Rotates around Y-axis

        # Extract waist pitch joint angle
        waist_pitch_angle = self.dof_pos[:, self.waist_pitch_index]  # [num_envs]

        # Compute penalty for deviation from target pitch
        deviation_penalty = torch.abs(waist_pitch_angle - target_pitch)  # Absolute deviation

        # Log deviation for debugging
        if log:
            print(f"Waist Pitch Deviation: {torch.mean(deviation_penalty).item()}")

        # Compute final reward (exponential penalty with weight)
        reward = weight * torch.exp(-2.0 * deviation_penalty)
        return reward
        
    def _reward_minimize_waist_roll_deviation(self, target_roll=0.0, weight=1.0, log=False):
        """
        Penalize deviation of waist_pitch_joint from a target angle.

        Args:
            target_pitch (float): Desired waist pitch angle in radians (default: 0.0).
            weight (float): Scaling factor for the reward (default: 1.0).
            log (bool): Whether to log the deviation (default: False).

        Returns:
            torch.Tensor: Reward value for minimizing waist pitch deviation.
        """
        self.waist_roll_index = self.dof_names.index('waist_roll_joint')  # Rotates around Y-axis

        # Extract waist pitch joint angle
        waist_roll_angle = self.dof_pos[:, self.waist_roll_index]  # [num_envs]

        # Compute penalty for deviation from target pitch
        deviation_penalty = torch.abs(waist_roll_angle - target_roll)  # Absolute deviation

        # Log deviation for debugging
        if log:
            print(f"Waist Pitch Deviation: {torch.mean(deviation_penalty).item()}")

        # Compute final reward (exponential penalty with weight)
        reward = weight * torch.exp(-2.0 * deviation_penalty)

        return reward
    
    def _reward_minimize_waist_yaw_deviation(self, target_yaw=0.0, weight=1.0, log=False):
        """
        Penalize deviation of waist_pitch_joint from a target angle.

        Args:
            target_pitch (float): Desired waist pitch angle in radians (default: 0.0).
            weight (float): Scaling factor for the reward (default: 1.0).
            log (bool): Whether to log the deviation (default: False).

        Returns:
            torch.Tensor: Reward value for minimizing waist pitch deviation.
        """
        self.waist_yaw_index = self.dof_names.index('waist_yaw_joint')  # Rotates around Y-axis

        # Extract waist pitch joint angle
        waist_yaw_angle = self.dof_pos[:, self.waist_yaw_index]  # [num_envs]

        # Compute penalty for deviation from target pitch
        deviation_penalty = torch.abs(waist_yaw_angle - target_yaw)  # Absolute deviation

        # Log deviation for debugging
        if log:
            print(f"Waist Pitch Deviation: {torch.mean(deviation_penalty).item()}")

        # Compute final reward (exponential penalty with weight)
        reward = weight * torch.exp(-2.0 * deviation_penalty)

        return reward
    
    def _reward_torso_yaw_smoothness(self):
        """Discourages excessive or jerky torso yaw motion."""
        torso_yaw_idx = self.dof_names.index("waist_yaw_joint")
        yaw_vel = self.dof_vel[:, torso_yaw_idx]
        yaw_acc = self.dof_acc[:, torso_yaw_idx] if hasattr(self, "dof_acc") else torch.zeros_like(yaw_vel)
        
        smoothness_reward = -0.5 * yaw_vel**2 - 0.01 * yaw_acc**2
        return smoothness_reward
    
    def _extract_upper_body_angular_velocity(self):
        """
        Extract angular velocities (roll rate, pitch rate, yaw rate) for the pelvis,
        waist_roll_link, and torso_link individually.

        Returns:
            tuple: Summed absolute angular velocities and individual angular velocities
                for pelvis, waist, and torso.
        """

        # Extract indices for each body part separately
        pelvis_idx = self.body_names.index('pelvis')
        waist_idx = self.body_names.index('waist_roll_link')
        torso_idx = self.body_names.index('torso_link')

        # Extract angular velocity states for each body part
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]

        pelvis_state = self.rigid_body_states_view[:, pelvis_idx, :]
        # print(f"pelvis_state: {pelvis_state}")
        waist_state = self.rigid_body_states_view[:, waist_idx, :]
        torso_state = self.rigid_body_states_view[:, torso_idx, :]

        # Extract angular velocity components (indices 10:13)
        pelvis_ang_vel = pelvis_state[:, 10:13]  # Shape: [num_envs, 3]
        waist_ang_vel = waist_state[:, 10:13]    # Shape: [num_envs, 3]
        torso_ang_vel = torso_state[:, 10:13]    # Shape: [num_envs, 3]

        # Compute the sum of absolute angular velocities
        upper_body_ang_vel_sum = torch.abs(pelvis_ang_vel) + torch.abs(waist_ang_vel) + torch.abs(torso_ang_vel)

        return upper_body_ang_vel_sum, pelvis_ang_vel, waist_ang_vel, torso_ang_vel

    def _reward_orientation(self):
        # 假设 projected_gravity 已经是归一化向量在根坐标系下的 [gx, gy, gz]
        pitch_roll_err = torch.sum(self.projected_gravity[:, :2]**2, dim=1)   # (B,)
        pitch_roll_vel = torch.sum(self.base_ang_vel[:, :2]**2, dim=1)        # (B,)
        
        k_angle = 300.0   # 角度误差强度
        k_vel   = 100.0    # 角速度误差强度
        
        r_angle = torch.exp(-k_angle * pitch_roll_err)
        r_vel   = torch.exp(-k_vel   * pitch_roll_vel)
        
        # 取两项几何平均，避免任何一项为 0
        return torch.sqrt(r_angle * r_vel)

def flip_g1_actor_obs(obs):
    if obs is None:
        return obs

    flipped_obs = torch.zeros_like(obs)
    # base_ang_vel
    flipped_obs[..., :3] = obs[..., :3]
    flipped_obs[..., 0] *= -1  # Flip the x component of base_ang_vel
    flipped_obs[..., 2] *= -1  # Flip the z component of base_ang_vel
    # projected_gravity
    flipped_obs[..., 3:6] = obs[..., 3:6]
    flipped_obs[..., 4] *= -1  # Flip the y component of projected_gravity

    # command
    flipped_obs[..., 6:9] = obs[..., 6:9]
    flipped_obs[..., 7] *= -1  # Flip the vy component of command
    flipped_obs[..., 8] *= -1  # Flip the wz component of command

    # dof_pos
    # legs
    flipped_obs[..., 9:15] = obs[..., 15:21]
    flipped_obs[..., 15:21] = obs[..., 9:15]
    # waist
    flipped_obs[..., 21:24] = obs[..., 21:24]
    # arm
    flipped_obs[..., 24:27] = obs[..., 27:30]
    flipped_obs[..., 27:30] = obs[..., 24:27]

    # dof_vel
    # legs
    flipped_obs[..., 30:36] = obs[..., 36:42]
    flipped_obs[..., 36:42] = obs[..., 30:36]
    # waist
    flipped_obs[..., 42:45] = obs[..., 42:45]
    # arm
    flipped_obs[..., 45:48] = obs[..., 48:51]
    flipped_obs[..., 48:51] = obs[..., 45:48]

    # actions
    # legs
    flipped_obs[..., 51:57] = obs[..., 57:63]
    flipped_obs[..., 57:63] = obs[..., 51:57]
    # waist
    flipped_obs[..., 63:66] = obs[..., 63:66]
    # arm
    flipped_obs[..., 66:69] = obs[..., 69:72]
    flipped_obs[..., 69:72] = obs[..., 66:69]

    vel_offset = act_offset = 21
    base_offset = 9
    # Flip the sign of specific bases in the observation
    # hip_roll, hip_yaw, ankle_roll, waist_yaw, waist_roll, shoulder_roll
    for base in [1, 2, 5, 7, 8, 11, 12, 13, 16, 19]:
        flipped_obs[..., base + base_offset] *= -1
        flipped_obs[..., base + base_offset + vel_offset] *= -1
        flipped_obs[..., base + base_offset + vel_offset + act_offset] *= -1
    return torch.cat([obs, flipped_obs], dim=0)


def flip_g1_critic_obs(obs):
    if obs is None:
        return obs

    flipped_obs = torch.zeros_like(obs)
    # base_lin_vel
    flipped_obs[..., :3] = obs[..., :3]
    flipped_obs[..., 1] *= -1  # Flip the y component of base_lin_vel
    # base_ang_vel
    flipped_obs[..., 3:6] = obs[..., 3:6]
    flipped_obs[..., 3] *= -1  # Flip the x component of base_ang_vel
    flipped_obs[..., 5] *= -1  # Flip the z component of base_ang_vel
    # projected_gravity
    flipped_obs[..., 6:9] = obs[..., 6:9]
    flipped_obs[..., 7] *= -1  # Flip the y component of projected_gravity
    # command
    flipped_obs[..., 9:12] = obs[..., 9:12]
    flipped_obs[..., 10] *= -1  # Flip the vy component of command
    flipped_obs[..., 11] *= -1  # Flip the wz component of command

    # dof_pos
    # legs
    flipped_obs[..., 12:18] = obs[..., 18:24]
    flipped_obs[..., 18:24] = obs[..., 12:18]
    # waist
    flipped_obs[..., 24:27] = obs[..., 24:27]
    # arm
    flipped_obs[..., 27:30] = obs[..., 30:33]
    flipped_obs[..., 30:33] = obs[..., 27:30]

    # dof_vel
    # legs
    flipped_obs[..., 33:39] = obs[..., 39:45]
    flipped_obs[..., 39:45] = obs[..., 33:39]
    # waist
    flipped_obs[..., 45:48] = obs[..., 45:48]
    # arm
    flipped_obs[..., 48:51] = obs[..., 51:54]
    flipped_obs[..., 51:54] = obs[..., 48:51]

    # actions
    # legs
    flipped_obs[..., 54:60] = obs[..., 60:66]
    flipped_obs[..., 60:66] = obs[..., 54:60]
    # waist
    flipped_obs[..., 66:69] = obs[..., 66:69]
    # arm
    flipped_obs[..., 69:72] = obs[..., 72:75]
    flipped_obs[..., 72:75] = obs[..., 69:72]

    vel_offset = act_offset = 21
    # Flip the sign of specific bases in the observation
    # hip_roll, hip_yaw, ankle_roll, waist_yaw, waist_roll, shoulder_roll
    base_offset = 12
    for base in [1, 2, 5, 7, 8, 11, 12, 13, 16, 19]:
        flipped_obs[..., base + base_offset] *= -1
        flipped_obs[..., base + vel_offset + base_offset] *= -1
        flipped_obs[..., base_offset + vel_offset + act_offset] *= -1
    return torch.cat([obs, flipped_obs], dim=0)


def flip_g1_actions(actions):
    if actions is None:
        return None

    flip_actions = torch.zeros_like(actions)
    # legs
    flip_actions[..., :6] = actions[..., 6:12]
    flip_actions[..., 6:12] = actions[..., :6]
    # waist
    flip_actions[..., 12:15] = actions[..., 12:15]
    # arm
    flip_actions[..., 15:18] = actions[..., 18:21]
    flip_actions[..., 18:21] = actions[..., 15:18]

    # hip_roll, hip_yaw, ankle_roll, waist_yaw, waist_roll, shoulder_roll
    for base in [1, 2, 5, 7, 8, 11, 12, 13, 16, 19]:
        flip_actions[..., base] *= -1
    return torch.cat([actions, flip_actions], dim=0)
    
def data_augmentation_func_g1(obs, actions, env, obs_type):
        if obs_type == "policy":
            obs_batch = flip_g1_actor_obs(obs)
        else:
            obs_batch = flip_g1_critic_obs(obs)

        mean_actions_batch = flip_g1_actions(actions)
        return (obs_batch, mean_actions_batch)
