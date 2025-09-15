# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from .isaac_lab_tutorial_env_cfg import IsaacLabTutorialEnvCfg

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils
# 插入：导入Jetbot环境所需的刚体对象和相机配置
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg



# from isaaclab.sensors.camera import Camera, CameraCfg  # 你的版本里若模块名不同，请用等价类

def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "forward": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
            "command": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


class IsaacLabTutorialEnv(DirectRLEnv):
    cfg: IsaacLabTutorialEnvCfg

    def __init__(self, cfg: IsaacLabTutorialEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # 保持已有的地面平面生成
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.articulations["robot"] = self.robot

        # 插入：添加围墙（四面墙围成边界）
        wall_size = (3.0, 0.1, 0.4)  # 长x宽x高（北/南墙沿X方向延伸）
        wall_size_vert = (0.1, 3.0, 0.4)  # 东/西墙沿Y方向延伸
        wall_z = wall_size[2] / 2  # 墙体中心高度
        walls_config = [
            {"name": "Wall_N", "pos": (0.0, 1.5, wall_z), "size": wall_size},    # 北墙
            {"name": "Wall_S", "pos": (0.0, -1.5, wall_z), "size": wall_size},   # 南墙
            {"name": "Wall_E", "pos": (1.5, 0.0, wall_z), "size": wall_size_vert},  # 东墙
            {"name": "Wall_W", "pos": (-1.5, 0.0, wall_z), "size": wall_size_vert}, # 西墙
        ]
        self._walls = []
        for wall in walls_config:
            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/{wall['name']}",
                spawn=sim_utils.CuboidCfg(
                    size=wall["size"],
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.8))  # 墙体灰色
                ,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    collision_props=sim_utils.CollisionPropertiesCfg()
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=wall["pos"], rot=(0.0, 0.0, 0.0, 1.0),
                    lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
                )
            )
            wall_obj = RigidObject(cfg)
            self._walls.append(wall_obj)
            self.scene.rigid_objects[wall['name'].lower()] = wall_obj  # 添加到场景以参与仿真
        # 插入：添加随机障碍物（大立方体）
        obstacle_size = (0.3, 0.3, 0.3)
        half_height = obstacle_size[2] / 2  # 障碍物初始高度一半
        init_positions = [(0.5, 0.5, half_height), (-0.5, 0.5, half_height), (0.5, -0.5, half_height)]
        self._obstacles = []
        # 障碍物颜色列表（可选不同颜色）
        obstacle_colors = [
            # (1.0, 0.0, 0.0),  # 红
            (0.0, 1.0, 0.0),  # 绿
            (0.0, 0.0, 1.0),  # 蓝
            (1.0, 1.0, 0.0),  # 黄
            (1.0, 0.0, 1.0),  # 紫
            (0.0, 1.0, 1.0),  # 青
        ]

        for i, pos in enumerate(init_positions):
            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Obstacle_{i}",
                spawn=sim_utils.CuboidCfg(
                    size=obstacle_size,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=obstacle_colors[i]),  # 障碍物颜色
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    collision_props=sim_utils.CollisionPropertiesCfg()
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=pos, rot=(0.0, 0.0, 0.0, 1.0),
                    lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
                )
            )
            obstacle_obj = RigidObject(cfg)
            self._obstacles.append(obstacle_obj)
            self.scene.rigid_objects[f"obstacle_{i}"] = obstacle_obj
        # 插入：添加目标物体（红色小立方体）
        target_size = (0.1, 0.1, 0.1)
        target_half_h = target_size[2] / 2
        target_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Target",
            spawn=sim_utils.CuboidCfg(
                size=target_size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))  # 目标物体红色
            ,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg()
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, -0.8, target_half_h), rot=(0.0, 0.0, 0.0, 1.0),
                lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
            )
        )
        self._target = RigidObject(target_cfg)
        self.scene.rigid_objects["target"] = self._target
        # 插入：添加Jetbot机载摄像头传感器（RGB相机）
        camera_cfg = CameraCfg(
            # prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
            prim_path="/World/envs/env_.*/Robot/front_cam",
            update_period=0.0167,  # 相机更新周期，大约60Hz
            height=320, width=320,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0,
                                            horizontal_aperture=20.955, clipping_range=(0.1, 1e5)),
            # offset=CameraCfg.OffsetCfg(pos=(0.5, 0.0, 0.1), rot=(0.0, 0.0, 0.0, 1.0), convention="ros")
            offset=CameraCfg.OffsetCfg(pos=(0.15, 0.0, 0.20), rot=(0.0, 0.0, 0.0, 1.0), convention="world")
        )
        self._camera = Camera(camera_cfg)
        self.scene.sensors["camera"] = self._camera
        # 保持已有的克隆环境和光源配置
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _visualize_markers(self):
        self.marker_locations = self.robot.data.root_pos_w
        self.forward_marker_orientations = self.robot.data.root_quat_w
        self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()

        loc = self.marker_locations + self.marker_offset
        loc = torch.vstack((loc, loc))
        rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations))

        all_envs = torch.arange(self.cfg.scene.num_envs)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))

        self.visualization_markers.visualize(loc, rots, marker_indices=indices)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        # 删除原有实现（例如之前返回线速度等的代码）
        # 插入：获取相机RGB图像作为观察
        # 获取相机传感器输出的图像张量，去除Alpha通道
        rgb_image = self._camera.data.output["rgb"][..., :3]  # 形状: (envs, height, width, 3)
        # 返回观测字典（以 "policy" 为键）
        return {"policy": rgb_image}

    def _get_rewards(self) -> torch.Tensor:
        # 删除原有占位的奖励计算（如果有，例如返回0的占位）
        # 插入：根据机器人与目标的距离和碰撞情况计算奖励
        # 机器人与目标的距离（二维平面距离）
        robot_xy = self.robot.data.root_state_w[:, 0:2]  # 机器人底座在世界坐标的 (x, y)
        target_xy = self._target.data.body_pos_w[:, 0, 0:2]  # 目标物体的位置 (x, y)
        dist_to_target = torch.norm(robot_xy - target_xy, p=2, dim=-1)
        # 距目标越近奖励越高（采用负距离作为即时奖励）
        reward = -dist_to_target
        # 额外奖励和惩罚项
        target_reached = dist_to_target < 0.2  # 判定是否到达目标阈值
        # 检测碰撞：机器人是否撞墙或撞障碍物
        wall_collision = torch.any(torch.abs(robot_xy) > 1.3, dim=-1)
        # 计算机器人到每个障碍的距离，判断是否碰撞（阈值约为0.3m）
        obs_positions = torch.stack([obs.data.body_pos_w[:, 0, 0:2] for obs in self._obstacles], dim=1)
        dist_to_obs = torch.norm(robot_xy.unsqueeze(1) - obs_positions, p=2, dim=-1)
        obs_collision = torch.any(dist_to_obs < 0.3, dim=1)
        collision = torch.logical_or(wall_collision, obs_collision)
        # 如果到达目标，给予额外正奖励；如果发生碰撞，给予负奖励
        reward += target_reached.float() * 10.0
        reward += collision.float() * (-10.0)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 删除原有实现（如果有，仅基础时间截断判断等）
        # 插入：基于目标达成、碰撞和时间步长确定终止标志
        robot_xy = self.robot.data.root_state_w[:, 0:2]
        print(f"Robot XY: {robot_xy}")
        target_xy = self._target.data.body_pos_w[:, 0, 0:2]
        dist_to_target = torch.norm(robot_xy - target_xy, p=2, dim=-1)
        # 条件1: 到达目标
        reached = dist_to_target < 0.2

        if reached.sum().item() > 0:
            print(f"Reached: {reached.sum().item()}/{self.cfg.scene.num_envs}, Distances: {dist_to_target}")
        # 条件2: 发生碰撞（撞墙或障碍）
        wall_collision = torch.any(torch.abs(robot_xy) > 1.5, dim=-1)
        print(f"Wall Collisions: {wall_collision.sum().item()}/{self.cfg.scene.num_envs}")
        obs_positions = torch.stack([obs.data.body_pos_w[:, 0, 0:2] for obs in self._obstacles], dim=1)
        dist_to_obs = torch.norm(robot_xy.unsqueeze(1) - obs_positions, p=2, dim=-1)
        obs_collision = torch.any(dist_to_obs < 0.3, dim=1)
        print(f"Obs Collisions: {obs_collision.sum().item()}/{self.cfg.scene.num_envs}")

        collision = torch.logical_or(wall_collision, obs_collision)
        # if collision.sum().item() > 0:
        #     print(f"Collisions: {collision.sum().item()}/{self.cfg.scene.num_envs}")
        # 终止条件：达到目标或发生碰撞（成功或失败终止）
        terminated = torch.logical_or(reached, collision)
        # 截断条件：达到最大步长（episode超时）
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        if truncated.sum().item() > 0:
            print(f"Truncated: {truncated.sum().item()}/{self.cfg.scene.num_envs}")
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None = None):
        # 删除原有实现中空的或默认的重置逻辑（如果有）
        # 插入：调用基类重置逻辑并添加自定义复位
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)


        # 参数：最小安全距离,考虑到障碍物具体的大小
        min_dist = 0.5

        # 重置机器人位置和朝向（放置到环境中心，速度清零）
        root_states = self.robot.data.default_root_state[env_ids].clone()
        root_states[:, :3] += self.scene.env_origins[env_ids]  # 设置位置为环境原点
        self.robot.write_root_pose_to_sim(root_states[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_states[:, 7:], env_ids=env_ids)
        # 重置机器人的关节（轮子）位置和速度
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # 随机重置目标
        target_state = self._target.data.default_root_state[env_ids].clone()
        rand_xy = torch.rand((len(env_ids), 2), device=self.device) * 2.0 - 1.0
        rand_xy *= 0.8
        target_xy = self.scene.env_origins[env_ids][:, 0:2] + rand_xy
        target_state[:, 0:2] = target_xy
        self._target.write_root_pose_to_sim(target_state[:, :7], env_ids=env_ids)
        self._target.write_root_velocity_to_sim(target_state[:, 7:], env_ids=env_ids)

        # ------------------------
        # 障碍物逐个重置，避免重叠
        placed_positions = [target_xy]   # 已经放好的物体位置（先有目标）
        for obs in self._obstacles:
            obj_state = obs.data.default_root_state[env_ids].clone()
            new_xy = torch.zeros((len(env_ids), 2), device=self.device)

            for i, env_origin in enumerate(self.scene.env_origins[env_ids]):
                while True:
                    candidate = (torch.rand(2, device=self.device) * 2.0 - 1.0) * 0.8
                    candidate = env_origin[:2] + candidate
                    # 检查与已放置的物体距离
                    dists = [torch.norm(candidate - p[i]) for p in placed_positions]
                    if all(d > min_dist for d in dists):
                        new_xy[i] = candidate
                        break


            obj_state[:, 0:2] = new_xy
            obs.write_root_pose_to_sim(obj_state[:, :7], env_ids=env_ids)
            obs.write_root_velocity_to_sim(obj_state[:, 7:], env_ids=env_ids)
            placed_positions.append(new_xy)
            # placed_positions.append(new_xy.unsqueeze(0))  # 加入到已放置列表


# # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # All rights reserved.
# #
# # SPDX-License-Identifier: Apache-2.0
#
# from __future__ import annotations
#
# import math
# import torch
# from collections.abc import Sequence
#
# import isaaclab.sim as sim_utils
# from isaaclab.assets import Articulation
# from isaaclab.envs import DirectRLEnv
# from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
# from .isaac_lab_tutorial_env_cfg import IsaacLabTutorialEnvCfg
#
# from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# import isaaclab.utils.math as math_utils
#
# # from isaaclab.sensors.camera import Camera, CameraCfg  # 你的版本里若模块名不同，请用等价类
#
# def define_markers() -> VisualizationMarkers:
#     """Define markers with various different shapes."""
#     marker_cfg = VisualizationMarkersCfg(
#         prim_path="/Visuals/myMarkers",
#         markers={
#                 "forward": sim_utils.UsdFileCfg(
#                     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
#                     scale=(0.25, 0.25, 0.5),
#                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
#                 ),
#                 "command": sim_utils.UsdFileCfg(
#                     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
#                     scale=(0.25, 0.25, 0.5),
#                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
#                 ),
#         },
#     )
#     return VisualizationMarkers(cfg=marker_cfg)
#
# class IsaacLabTutorialEnv(DirectRLEnv):
#     cfg: IsaacLabTutorialEnvCfg
#
#     def __init__(self, cfg: IsaacLabTutorialEnvCfg, render_mode: str | None = None, **kwargs):
#         super().__init__(cfg, render_mode, **kwargs)
#         self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
#
#     def _setup_scene(self):
#         # self.goal_pos_w = torch.zeros((self.cfg.scene.num_envs, 3), device="cuda")  # 世界系目标位置
#         # self.goal_pos_w[:, 2] = 0.05  # 目标小球离地高度（可视化用）
#
#         self.robot = Articulation(self.cfg.robot_cfg)
#         # add ground plane
#         spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
#         # clone and replicate
#         self.scene.clone_environments(copy_from_source=False)
#         # add articulation to scene
#         self.scene.articulations["robot"] = self.robot
#         # add lights
#         light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
#         light_cfg.func("/World/Light", light_cfg)
#
#         self.visualization_markers = define_markers()
#
#         self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()
#         self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda()
#         self.commands = torch.randn((self.cfg.scene.num_envs, 3)).cuda()
#         self.commands[:,-1] = 0.0
#         self.commands = self.commands/torch.linalg.norm(self.commands, dim=1, keepdim=True)
#
#         # offsets to account for atan range and keep things on [-pi, pi]
#         ratio = self.commands[:,1]/(self.commands[:,0]+1E-8)
#         gzero = torch.where(self.commands > 0, True, False)
#         lzero = torch.where(self.commands < 0, True, False)
#         plus = lzero[:,0]*gzero[:,1]
#         minus = lzero[:,0]*lzero[:,1]
#         offsets = torch.pi*plus - torch.pi*minus
#         self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)
#
#         self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
#         self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
#         self.marker_offset[:,-1] = 0.5
#         self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
#         self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
#
#
#     def _visualize_markers(self):
#         self.marker_locations = self.robot.data.root_pos_w
#         self.forward_marker_orientations = self.robot.data.root_quat_w
#         self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()
#
#         loc = self.marker_locations + self.marker_offset
#         loc = torch.vstack((loc, loc))
#         rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations))
#
#         all_envs = torch.arange(self.cfg.scene.num_envs)
#         indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
#
#         self.visualization_markers.visualize(loc, rots, marker_indices=indices)
#
#     def _pre_physics_step(self, actions: torch.Tensor) -> None:
#         self.actions = actions.clone()# + torch.ones_like(actions)
#         self._visualize_markers()
#
#     def _apply_action(self) -> None:
#         self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)
#
#
#     def _get_observations(self) -> dict:
#
#         self.velocity = self.robot.data.root_com_vel_w
#         self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)
#         # obs = torch.hstack((self.velocity, self.commands))
#
#         dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
#         cross = torch.cross(self.forwards, self.commands, dim=-1)[:,-1].reshape(-1,1)
#         forward_speed = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
#         obs = torch.hstack((dot, cross, forward_speed))
#
#         observations = {"policy": obs}
#         return observations
#
#
#
#     def _get_rewards(self) -> torch.Tensor:
#         forward_reward = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
#         alignment_reward = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
#         total_reward = forward_reward + alignment_reward
#         # total_reward = forward_reward*alignment_reward
#         # total_reward = forward_reward*alignment_reward + forward_reward
#         # total_reward = forward_reward*torch.exp(alignment_reward)
#         return total_reward
#
#
#
#     def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
#         time_out = self.episode_length_buf >= self.max_episode_length - 1
#
#         return False, time_out
#
#
#
#     def _reset_idx(self, env_ids: Sequence[int] | None):
#         if env_ids is None:
#             env_ids = self.robot._ALL_INDICES
#         super()._reset_idx(env_ids)
#
#         self.commands[env_ids] = torch.randn((len(env_ids), 3)).cuda()
#         self.commands[env_ids,-1] = 0.0
#         self.commands[env_ids] = self.commands[env_ids]/torch.linalg.norm(self.commands[env_ids], dim=1, keepdim=True)
#
#         ratio = self.commands[env_ids][:,1]/(self.commands[env_ids][:,0]+1E-8)
#         gzero = torch.where(self.commands[env_ids] > 0, True, False)
#         lzero = torch.where(self.commands[env_ids]< 0, True, False)
#         plus = lzero[:,0]*gzero[:,1]
#         minus = lzero[:,0]*lzero[:,1]
#         offsets = torch.pi*plus - torch.pi*minus
#         self.yaws[env_ids] = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)
#
#         default_root_state = self.robot.data.default_root_state[env_ids]
#         default_root_state[:, :3] += self.scene.env_origins[env_ids]
#
#         self.robot.write_root_state_to_sim(default_root_state, env_ids)
#         self._visualize_markers()
#
#
#
