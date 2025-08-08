"""
Navigation-only DynaMem runner that avoids importing any manipulation/IK modules
(pinocchio/hppfcl). Use this when run_dynamem.py segfaults on hppfcl import.

Drop this file at: stretch/app/run_dynamem_nav.py
Run with:       python3 -m stretch.app.run_dynamem_nav --query "mug"

This script intentionally does NOT import: 
  - stretch.agent.robot_agent_dynamem
  - any manipulation wrappers or pinocchio-based IK bits

It builds the voxel map + navigation space directly and drives A* planning
for: (a) open-vocabulary target navigation, or (b) frontier exploration.

Swap the ROBOT CLIENT import in the section marked "ROBOT CLIENT" to match
your hardware client (e.g., ROS2 driver). Defaults to the sim client so the
file is import-safe on any machine.
"""
from __future__ import annotations
import argparse
import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np

# =========================
# ROBOT CLIENT (choose one)
# =========================
# 1) Sim client (import-safe anywhere). Works with Mujoco sim.
from stretch.dynamem_sim.test.sim_robot_client import SimRobotClient as RobotClient  # noqa: F401

# 2) If you have a ROS2 hardware client, swap to your class below and comment out the sim import.
# Example (adjust if your path/class differs):
# from stretch.ros2.ros2_robot_client import Ros2RobotClient as RobotClient  # noqa: F401

# Core types & params
from stretch.core import Parameters
from stretch.core.interfaces import Observations

# Mapping + planning (these are pure-Python/CUDA; no pinocchio/hppfcl)
from stretch.mapping.voxel import (
    SparseVoxelMapDynamem as SparseVoxelMap,
    SparseVoxelMapNavigationSpaceDynamem as SparseVoxelMapNavigationSpace,
)
from stretch.motion.algo.a_star import AStar

# Perception pieces (also safe)
from stretch.perception.encoders import MaskSiglipEncoder
from stretch.perception.detection.owl import OwlPerception

# Rerun visualizer (optional)
import rerun as rr
import torch


class NavOnlyAgent:
    def __init__(self, robot: RobotClient, params: Parameters, mllm: bool = False, log: Optional[str] = None):
        self.robot = robot
        self.params = params
        self.mllm = mllm
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Logging dir
        if not os.path.exists("dynamem_log"):
            os.makedirs("dynamem_log")
        dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join("dynamem_log", log or f"nav_{dt}")

        # Build voxel map + nav space WITHOUT touching manipulation / IK
        self._create_nav_stack()

        # Obs counter
        self.obs_count = 0

    def _create_nav_stack(self):
        # For nav-only, we still want VLM features + open-vocab detection
        encoder = MaskSiglipEncoder(device=self.device, version="so400m")
        if self.mllm:
            detection = OwlPerception(version="owlv2-B-p16", device=self.device, confidence_threshold=0.01)
            semantic_memory_resolution = 0.1
            image_shape = (360, 270)
        else:
            detection = OwlPerception(version="owlv2-L-p14-ensemble", device=self.device, confidence_threshold=0.2)
            semantic_memory_resolution = 0.05
            image_shape = (480, 360)

        self.voxel_map = SparseVoxelMap(
            resolution=self.params["voxel_size"],
            semantic_memory_resolution=semantic_memory_resolution,
            local_radius=self.params["local_radius"],
            obs_min_height=self.params["obs_min_height"],
            obs_max_height=self.params["obs_max_height"],
            obs_min_density=self.params["obs_min_density"],
            grid_resolution=0.1,
            min_depth=self.params["min_depth"],
            max_depth=self.params["max_depth"],
            pad_obstacles=self.params["pad_obstacles"],
            add_local_radius_points=self.params.get("add_local_radius_points", default=True),
            remove_visited_from_obstacles=self.params.get("remove_visited_from_obstacles", default=False),
            smooth_kernel_size=self.params.get("filters/smooth_kernel_size", -1),
            use_median_filter=self.params.get("filters/use_median_filter", False),
            median_filter_size=self.params.get("filters/median_filter_size", 5),
            median_filter_max_error=self.params.get("filters/median_filter_max_error", 0.01),
            use_derivative_filter=self.params.get("filters/use_derivative_filter", False),
            derivative_filter_threshold=self.params.get("filters/derivative_filter_threshold", 0.5),
            detection=detection,
            encoder=encoder,
            image_shape=image_shape,
            log=self.log_dir,
            mllm=self.mllm,
        )
        self.space = SparseVoxelMapNavigationSpace(
            self.voxel_map,
            rotation_step_size=self.params.get("motion_planner/rotation_step_size", 0.2),
            dilate_frontier_size=self.params.get("motion_planner/frontier/dilate_frontier_size", 2),
            dilate_obstacle_size=self.params.get("motion_planner/frontier/dilate_obstacle_size", 0),
        )
        self.planner = AStar(self.space)

    # -------------- sensing + mapping --------------
    def update_once(self):
        obs: Observations = self.robot.get_observation()
        self.obs_count += 1
        self.voxel_map.process_rgbd_images(obs.rgb, obs.depth, obs.camera_K, obs.camera_pose)

    def rotate_in_place(self, steps: int = 8):
        xyt = self.robot.get_base_pose()
        self.robot.head_to(head_pan=0, head_tilt=-0.6, blocking=True)
        for _ in range(steps):
            xyt[2] += 2 * np.pi / steps
            self.robot.move_base_to(xyt, blocking=True)
            self.update_once()

    # -------------- planning + control --------------
    def _frontier_or_localize(self, text: Optional[str], start_pose: np.ndarray):
        target_point = None
        if text:
            target_point, _, _, _ = self.voxel_map.localize_text(text, debug=False, return_debug=True)
        if target_point is None:
            target_point = self.space.sample_frontier(self.planner, start_pose, text)
        return target_point

    def execute_goal(self, text: Optional[str]):
        self.robot.switch_to_navigation_mode()
        self.robot.look_front()

        # quick map sweep
        self.rotate_in_place(steps=8)

        start = self.robot.get_base_pose()
        goal = self._frontier_or_localize(text, start)
        if goal is None:
            print("[NavOnly] No goal found (neither localization nor frontier).")
            return False

        point = self.space.sample_navigation(start, self.planner, goal)
        if point is None:
            print("[NavOnly] No navigable point for the sampled goal.")
            return False

        plan = self.planner.plan(start, point)
        if not (plan and plan.success):
            print(f"[NavOnly] Planning failed: {getattr(plan, 'reason', 'unknown')}")
            return False

        waypoints = [pt.state for pt in plan.trajectory]
        traj = self.planner.clean_path_for_xy(waypoints)
        self.robot.execute_trajectory(traj, pos_err_threshold=self.params["trajectory_pos_err_threshold"],
                                      rot_err_threshold=self.params["trajectory_rot_err_threshold"], blocking=True)
        return True


def build_default_params() -> Parameters:
    # Start from repo defaults if you have a YAML; otherwise create minimal set
    # If your project normally loads a YAML, replace this with your loader.
    return Parameters(
        voxel_size=0.05,
        local_radius=2.0,
        obs_min_height=0.0,
        obs_max_height=2.0,
        obs_min_density=3,
        min_depth=0.2,
        max_depth=3.5,
        pad_obstacles=True,
        motion_planner={
            "rotation_step_size": 0.2,
            "frontier": {"dilate_frontier_size": 2, "dilate_obstacle_size": 0, "min_dist": 0.6, "step_dist": 0.3},
            "goals": {"manipulation_radius": 0.75},
        },
        trajectory_pos_err_threshold=0.05,
        trajectory_rot_err_threshold=0.15,
        agent={"sweep_head_on_update": False},
    )


def main():
    parser = argparse.ArgumentParser(description="Navigation-only DynaMem runner (no hppfcl)")
    parser.add_argument("--query", type=str, default="", help="Open-vocabulary target to navigate to. Empty = explore")
    parser.add_argument("--mllm", action="store_true", help="Use mLLM-friendly settings (coarser semantic memory)")
    parser.add_argument("--no-viz", action="store_true", help="Disable Rerun viz")
    args = parser.parse_args()

    if not args.no_viz:
        rr.init("Stretch_NavOnly", spawn=True)

    params = build_default_params()

    # Make robot client
    robot = RobotClient()
    robot.move_to_nav_posture()

    agent = NavOnlyAgent(robot, params, mllm=args.mllm)

    text = args.query.strip()
    text = text if text else None

    ok = agent.execute_goal(text)
    if not ok and text is not None:
        print("[NavOnly] Trying exploration fallback…")
        agent.execute_goal(None)


if __name__ == "__main__":
    sys.exit(main())
