# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
1
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import math
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
import json


import numpy as np
import torch
import glob
import joblib
import pickle

MOTION_FILES = glob.glob('/home/tianhu/amass/Retargeted_Data/accad_to_g1_joblib/Female1General_c3d/A5_-_pick_up_box_stageii.pkl')


def _check_command_interface(path: str = "command_interface.json"):
    """读取命令文件，如果不存在则返回默认 target_z=0.9。"""
    if not os.path.exists(path):
        return {"mode": "fixed", "height_z": 0.90}
    with open(path, "r") as f:
        return json.load(f)


def _update_command_ranges(env, ci, t_now: float = None):
    mode = str(ci.get("mode", "fixed")).lower()
    # --- 时间源 ---
    if t_now is None:
        if hasattr(env, "sim_time"):
            t_now = float(env.sim_time)
        else:
            if not hasattr(env, "_frame_i"): env._frame_i = 0
            if not hasattr(env, "dt"): env.dt = 0.016
            t_now = env._frame_i * env.dt
            env._frame_i += 1

    # --- 生成 h_val ---
    if mode == "sine":
        h_min   = float(ci.get("h_min", 0.6))
        h_max   = float(ci.get("h_max", 1.2))
        freq_hz = float(ci.get("freq_hz", 0.5))
        phase   = math.radians(float(ci.get("phase_deg", 0.0)))
        if not hasattr(env, "_height_mode") or env._height_mode != "sine":
            env._height_mode = "sine"
            env._height_t0   = t_now
        h_mid = 0.5*(h_min + h_max)
        amp   = 0.5*(h_max - h_min)
        h_val = h_mid + amp * math.sin(2.0*math.pi*freq_hz*(t_now - env._height_t0) + phase)
    else:
        # 固定高度；若 JSON 没给就用兜底 0.9
        h_val = float(ci.get("height_z", ci.get("height", 0.9)))
        env._height_mode = "fixed"

    # --- 关键：立刻写入当前命令（[B,1]）---
    B, dev = env.num_envs, env.device

    env.command_height[:] = h_val  # 现在 self.command_height 会随 h_val 变化




def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.commands.ranges.target_point_radius = [1.0, 1.0]  # range of target point radius
    env_cfg.commands.ranges.target_point_theta = [np.pi, np.pi]
    # env_cfg.commands.ranges.target_z = 1.3  # target point z coordinate
    env_cfg.env.max_episode_length = 10

    train_cfg.runner.amp_num_preload_transitions = 10

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    obs, _ = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.policy, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    for i in range(2 * int(env.max_episode_length)):

        command_interface = _check_command_interface()
        _update_command_ranges(env, command_interface)
        # update_command_by_schedule(env, i)
        # use_mocap = True
        # if use_mocap:
        #     data = joblib.load(motion_file)
        #     # 强制转换成 float32
        #     dof_pos = torch.tensor(data["dof_pos"], device=env.device, dtype=torch.float32)  # [T, num_dofs]

        #     # 按照时间/步数选帧
        #     frame_idx = i % len(dof_pos)
        #     target_q = dof_pos[frame_idx]

        #     # 把 target_q 转换成 action (float32 + batch repeat)
        #     actions = (target_q - env.default_dof_pos) / env.cfg.control.action_scale
        #     actions = torch.clamp(actions, -1.0, 1.0).to(dtype=torch.float32)   # [29]
        #     print('1actions.shape:',actions.shape)


        #     # 扩展 batch 维度
        #     # actions = actions.unsqueeze(0)   # [1, 29]
        #     print('2actions.shape:',actions.shape)

        #     # 如果有多个 env，则 repeat
        #     if env.num_envs > 1:
        #         actions = actions.repeat(env.num_envs, 1)   # [num_envs, 29]

        #     print('3actions.shape:',actions.shape)
            
        #     # 下半身锁死
        #     lower_body_indices = env.cfg.asset.lower_body_indices
        #     actions[:, lower_body_indices] = 0.0
        #     print('4actions.shape:',actions.shape)





        #     # 如果你有多个 env，要扩展
        #     # if actions.shape[0] == 1 and env.num_envs > 1:
        #     #     actions = actions.repeat(env.num_envs, 1)
        # else:
        #     actions = policy(obs.detach())


        # use_mocap = False
        # if use_mocap:
        #     motion_file = MOTION_FILES[0]  # ✅ 取出字符串路径

        #     # 1) 先跑 policy 得到底座动作
        #     with torch.no_grad():
        #         base_actions = policy(obs.detach())            # [B, D]
        #     base_actions = base_actions.to(env.device, dtype=torch.float32)

        #     # 2) 读 mocap 并取当前帧的关节角
        #     data = joblib.load(motion_file)                   # 单文件路径字符串
        #     dof_pos_mc = torch.tensor(data["dof_pos"], device=env.device, dtype=torch.float32)  # [T, D]
        #     frame_idx = i % dof_pos_mc.shape[0]
        #     target_q   = dof_pos_mc[frame_idx]                # [D]  —— 假设和仿真 DOF 顺序一致

        #     # 3) 只取手腕关节的 indices（根据你的模型命名来）
        #     #    如果你已经在 env 里有 dof_name->index 的字典，就用它
        #     name_to_idx = getattr(env, "dof_dict", None) or getattr(env, "joint_name_to_index", None)
        #     if name_to_idx is None:
        #         raise RuntimeError("Need a joint name->index dict on env (dof_dict / joint_name_to_index).")

        #     wrist_names = ["left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_shoulder_pitch_joint",
        #     "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_shoulder_pitch_joint",
        #         "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
        #         "right_wrist_yaw_joint","right_wrist_pitch_joint","right_wrist_roll_joint"
        #     ]
        #     wrist_idx = [name_to_idx[n] for n in wrist_names if n in name_to_idx]
        #     wrist_idx = torch.as_tensor(wrist_idx, device=env.device, dtype=torch.long)

        #     # 4) 把 mocap 角度映射到动作空间： a = (q_target - q_default) / action_scale
        #     #    注意 default 可能是 [B,D] 或 [D]，这里统一取 [D]
        #     q_default = env.default_dof_pos[0] if env.default_dof_pos.ndim == 2 else env.default_dof_pos
        #     a_mocap_full = (target_q - q_default) / env.cfg.control.action_scale   # [D]
        #     a_mocap_full = torch.clamp(a_mocap_full, -1.0, 1.0)

        #     # 5) 融合：只覆盖手腕关节分量
        #     actions = base_actions.clone()                    # [B, D]
        #     actions[:, wrist_idx] = a_mocap_full[wrist_idx]   # 用 mocap 替换手腕动作

        #     # 6)（可选）平滑融合，避免抖： actions = α*mocap + (1-α)*policy，手腕上用
        #     # alpha = 0.6
        #     # actions[:, wrist_idx] = alpha * a_mocap_full[wrist_idx] + (1.0 - alpha) * base_actions[:, wrist_idx]

        #     # 7) 安全检查
        #     assert actions.ndim == 2, f"actions ndim={actions.ndim}, shape={actions.shape}"
        #     assert actions.shape[0] == env.num_envs, (actions.shape, env.num_envs)
        #     assert actions.shape[1] == env.num_actions, (actions.shape, env.num_actions)

        # else:
        #     with torch.no_grad():
        #         actions = policy(obs.detach())
        actions = policy(obs.detach())
        obs, rews, dones, infos, _, _ = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
