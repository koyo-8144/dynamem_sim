# Copyright (c) Hello Robot
# MIT-licensed Meta base agent portions

import os
import timeit
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import zmq

# Manipulation wrappers (vision-driven picking/placing helpers)
# from stretch.agent.manipulation.dynamem_manipulation.dynamem_manipulation import (
#     DynamemManipulationWrapper as ManipulationWrapper,
# )
from stretch.agent.manipulation.dynamem_manipulation.grasper_utils import (
    capture_and_process_image,
    move_to_point,
    pickup,
    process_image_for_placing,
)

# Base agent (speech, some motion wrappers)
# from stretch.agent.robot_agent import RobotAgent as RobotAgentBase

# TTS
from stretch.audio.text_to_speech import get_text_to_speech

# Observation struct + params + abstract interfaces
from stretch.core.interfaces import Observations
from stretch.core.parameters import Parameters
from stretch.core.robot import AbstractGraspClient, AbstractRobotClient

# DynaMem voxel map + navigation space
from stretch.mapping.instance import Instance
from stretch.mapping.voxel import SparseVoxelMapDynamem as SparseVoxelMap
from stretch.mapping.voxel import (
    SparseVoxelMapNavigationSpaceDynamem as SparseVoxelMapNavigationSpace,
)
from stretch.mapping.voxel import SparseVoxelMapProxy

# Planner
from stretch.motion.algo.a_star import AStar

# Perception stacks
from stretch.perception.detection.owl import OwlPerception
from stretch.perception.encoders import MaskSiglipEncoder
from stretch.perception.wrapper import OvmmPerception

# Default manipulation postures
INIT_LIFT_POS = 0.45
INIT_WRIST_PITCH = -1.57
INIT_ARM_POS = 0
INIT_WRIST_ROLL = 0
INIT_WRIST_YAW = 0
INIT_HEAD_PAN = -1.57
INIT_HEAD_TILT = -0.65

from robot_agent_test import RobotAgentTest as RobotAgentBase


class RobotAgentTest(RobotAgentBase):
    """
    DynaMem-enabled agent:
      - builds/maintains dynamic voxel map (3D obstacles + open-vocab semantic memory)
      - localizes language queries to 3D points (VLM or mLLM hybrid)
      - samples frontiers when goal unknown/absent
      - plans paths with A*
      - wraps simple manipulation via AnyGrasp pipelines
      - streams to Rerun for live viz
    """

    def __init__(
        self,
        robot: AbstractRobotClient,
        parameters: Union[Parameters, Dict[str, Any]],
        semantic_sensor: Optional[OvmmPerception] = None,
        grasp_client: Optional[AbstractGraspClient] = None,
        voxel_map: Optional[SparseVoxelMap] = None,
        debug_instances: bool = True,
        show_instances_detected: bool = False,
        use_instance_memory: bool = False,
        realtime_updates: bool = False,
        obs_sub_port: int = 4450,
        re: int = 3,
        manip_port: int = 5557,
        log: Optional[str] = None,
        server_ip: Optional[str] = "127.0.0.1",
        mllm: bool = False,
        manipulation_only: bool = False,
    ):
        self.reset_object_plans()

        # Normalize Parameters
        if isinstance(parameters, Dict):
            self.parameters = Parameters(**parameters)
        elif isinstance(parameters, Parameters):
            self.parameters = parameters
        else:
            raise RuntimeError(f"parameters of unsupported type: {type(parameters)}")

        self.robot = robot
        self.grasp_client = grasp_client
        self.debug_instances = debug_instances
        self.show_instances_detected = show_instances_detected

        self.semantic_sensor = semantic_sensor
        self.pos_err_threshold = parameters["trajectory_pos_err_threshold"]
        self.rot_err_threshold = parameters["trajectory_rot_err_threshold"]

        # Rerun visualizer provided by the client
        self.rerun_visualizer = self.robot._rerun
        self.setup_custom_blueprint()

        # Query strategy flags
        self.mllm = mllm
        self.manipulation_only = manipulation_only

        # Lazy-created detection pipeline for place()
        self.owl_sam_detector = None

        # Device for encoders/detectors
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Logs
        if not os.path.exists("dynamem_log"):
            os.makedirs("dynamem_log")
        if log is None:
            current_datetime = datetime.now()
            self.log = "dynamem_log/debug_" + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        else:
            self.log = "dynamem_log/" + log

        # Build obstacle/semantic map + navigation space + planner
        self.create_obstacle_map(parameters)

        # Shared voxel map proxy so other components can read safely
        self._voxel_map_lock = Lock()
        self.voxel_map_proxy = SparseVoxelMapProxy(self.voxel_map, self._voxel_map_lock)

        # Observation cache & flags
        self.obs_count = 0
        self.obs_history: List[Observations] = []

        self.guarantee_instance_is_reachable = self.parameters.guarantee_instance_is_reachable
        self.use_scene_graph = self.parameters["use_scene_graph"]
        self.tts = get_text_to_speech(self.parameters["tts_engine"])
        self._use_instance_memory = use_instance_memory
        self._realtime_updates = realtime_updates

        # Config: head sweeping to gather more context during updates
        self._sweep_head_on_update = parameters["agent"]["sweep_head_on_update"]

        # Task-level state (e.g., current object/receptacle being manipulated)
        self.current_receptacle: Instance = None
        self.current_object: Instance = None
        self.target_object = None
        self.target_receptacle = None

        # Feature matching thresholds for grasp/localization
        self._is_match_threshold = parameters.get("encoder_args/feature_match_threshold", 0.05)
        self._grasp_match_threshold = parameters.get(
            "encoder_args/grasp_feature_match_threshold", 0.05
        )

        # Frontier/exploration + goal sampling knobs
        self._default_expand_frontier_size = parameters["motion_planner"]["frontier"][
            "default_expand_frontier_size"
        ]
        self._frontier_min_dist = parameters["motion_planner"]["frontier"]["min_dist"]
        self._frontier_step_dist = parameters["motion_planner"]["frontier"]["step_dist"]
        self._manipulation_radius = parameters["motion_planner"]["goals"]["manipulation_radius"]
        self._voxel_size = parameters["voxel_size"]

        # Socket for the workstation-side manipulation server (AnyGrasp etc.)
        context = zmq.Context()
        self.manip_socket = context.socket(zmq.REQ)
        self.manip_socket.connect("tcp://" + server_ip + ":" + str(manip_port))

        # Arm geometry differs across revisions; set proper end link and gripper range
        if re == 1 or re == 2:
            stretch_gripper_max = 0.3
            end_link = "link_straight_gripper"
        else:
            stretch_gripper_max = 0.64
            end_link = "link_gripper_s3_body"
        self.transform_node = end_link
        # self.manip_wrapper = ManipulationWrapper(
        #     self.robot, stretch_gripper_max=stretch_gripper_max, end_link=end_link
        # )

        # Start safely in nav posture
        self.robot.move_to_nav_posture()
        self.reset_object_plans()

        self.re = re

        # Scene graph (optional relational reasoning)
        self.scene_graph = None

        # For resume-able navigation (store tail of planned waypoints)
        self._previous_goal = None

        self._running = True

        # Kick off any background threads this agent needs
        self._start_threads()

    # ------------------------------
    # Map + planner setup
    # ------------------------------

    def create_obstacle_map(self, parameters):
        """
        Build the semantic + obstacle voxel map and the navigation space wrapper.
        Resolution / detector choices change depending on mLLM vs feature-only mode.
        """
        # Encoder only needed for mapping/localization (not manipulation-only bench)
        if self.manipulation_only:
            self.encoder = None
        else:
            self.encoder = MaskSiglipEncoder(device=self.device, version="so400m")

        # Detector / memory resolution tradeoffs:
        # - mLLM mode uses faster/lower-res OWL since mLLM does heavy lifting
        # - feature mode uses higher-res memory for better similarity grounding
        if self.manipulation_only:
            self.detection_model = None
            semantic_memory_resolution = 0.1
            image_shape = (360, 270)
        elif self.mllm:
            self.detection_model = OwlPerception(
                version="owlv2-B-p16", device=self.device, confidence_threshold=0.01
            )
            semantic_memory_resolution = 0.1
            image_shape = (360, 270)
        else:
            self.detection_model = OwlPerception(
                version="owlv2-L-p14-ensemble", device=self.device, confidence_threshold=0.2
            )
            semantic_memory_resolution = 0.05
            image_shape = (480, 360)

        # Core DynaMem voxel map
        self.voxel_map = SparseVoxelMap(
            resolution=parameters["voxel_size"],          # occupancy/obstacle voxel size
            semantic_memory_resolution=semantic_memory_resolution,  # finer = better VLM granularity
            local_radius=parameters["local_radius"],      # local region to densify around robot
            obs_min_height=parameters["obs_min_height"],
            obs_max_height=parameters["obs_max_height"],
            obs_min_density=parameters["obs_min_density"],
            grid_resolution=0.1,                          # 2D nav grid discretization
            min_depth=parameters["min_depth"],
            max_depth=parameters["max_depth"],
            pad_obstacles=parameters["pad_obstacles"],
            add_local_radius_points=parameters.get("add_local_radius_points", default=True),
            remove_visited_from_obstacles=parameters.get("remove_visited_from_obstacles", default=False),
            # filters for denoising
            smooth_kernel_size=parameters.get("filters/smooth_kernel_size", -1),
            use_median_filter=parameters.get("filters/use_median_filter", False),
            median_filter_size=parameters.get("filters/median_filter_size", 5),
            median_filter_max_error=parameters.get("filters/median_filter_max_error", 0.01),
            use_derivative_filter=parameters.get("filters/use_derivative_filter", False),
            derivative_filter_threshold=parameters.get("filters/derivative_filter_threshold", 0.5),
            # perception hooks
            detection=self.detection_model,
            encoder=self.encoder,
            image_shape=image_shape,
            log=self.log,
            mllm=self.mllm,
        )

        # Navigation wrapper over voxel map: exposes frontiers & motion primitives
        self.space = SparseVoxelMapNavigationSpace(
            self.voxel_map,
            rotation_step_size=parameters.get("motion_planner/rotation_step_size", 0.2),
            dilate_frontier_size=parameters.get("motion_planner/frontier/dilate_frontier_size", 2),
            dilate_obstacle_size=parameters.get("motion_planner/frontier/dilate_obstacle_size", 0),
        )

        # Simple A* planner in that space
        self.planner = AStar(self.space)

    # ------------------------------
    # Rerun layout
    # ------------------------------

    def setup_custom_blueprint(self):
        """Define a multi-pane Rerun blueprint: 3D, monologue text, and camera views."""
        main = rrb.Horizontal(
            rrb.Spatial3DView(name="3D View", origin="world"),
            rrb.Vertical(
                rrb.TextDocumentView(name="text", origin="robot_monologue"),
                rrb.Spatial2DView(name="image", origin="/observation_similar_to_text"),
            ),
            rrb.Vertical(
                rrb.Spatial2DView(name="head_rgb", origin="/world/head_camera"),
                rrb.Spatial2DView(name="ee_rgb", origin="/world/ee_camera"),
            ),
            column_shares=[2, 1, 1],
        )
        my_blueprint = rrb.Blueprint(
            rrb.Vertical(main, rrb.TimePanel(state=True)),
            collapse_panels=True,
        )
        rr.send_blueprint(my_blueprint)

    # ------------------------------
    # Mapping/update utilities
    # ------------------------------

    def compute_blur_metric(self, image):
        """Heuristic “sharpness” score via Sobel gradients; used to pick better frames."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.magnitude(Gx, Gy).mean()

    def update_map_with_pose_graph(self):
        """
        If pose graph is available from SLAM:
          - retro-correct prior observations’ poses
          - re-fuse best recent observation into voxel map
          - (optional) update scene graph
        Also garbage-collects old observations that aren’t anchor nodes.
        """
        t0 = timeit.default_timer()
        self.pose_graph = self.robot.get_pose_graph()
        t1 = timeit.default_timer()

        # Retro-correct prior obs (align them to latest pose-graph)
        self._obs_history_lock.acquire()
        for idx in range(len(self.obs_history)):
            lidar_ts = self.obs_history[idx].lidar_timestamp
            gps_past = self.obs_history[idx].gps

            for vertex in self.pose_graph:
                # exact timestamp match → snap
                if abs(vertex[0] - lidar_ts) < 0.05:
                    self.obs_history[idx].is_pose_graph_node = True
                    self.obs_history[idx].gps = np.array([vertex[1], vertex[2]])
                    self.obs_history[idx].compass = np.array([vertex[3]])
                    if self.obs_history[idx].task_observations is None and self.semantic_sensor is not None:
                        self.obs_history[idx] = self.semantic_sensor.predict(self.obs_history[idx])
                # close in space → associate to that vertex for future deltas
                elif (
                    np.linalg.norm(gps_past - np.array([vertex[1], vertex[2]])) < 0.3
                    and self.obs_history[idx].pose_graph_timestamp is None
                ):
                    self.obs_history[idx].is_pose_graph_node = True
                    self.obs_history[idx].pose_graph_timestamp = vertex[0]
                    self.obs_history[idx].initial_pose_graph_gps = np.array([vertex[1], vertex[2]])
                    self.obs_history[idx].initial_pose_graph_compass = np.array([vertex[3]])
                    if self.obs_history[idx].task_observations is None and self.semantic_sensor is not None:
                        self.obs_history[idx] = self.semantic_sensor.predict(self.obs_history[idx])
                # if already associated, apply delta updates
                elif self.obs_history[idx].pose_graph_timestamp == vertex[0]:
                    delta_gps = vertex[1:3] - self.obs_history[idx].initial_pose_graph_gps
                    delta_compass = vertex[3] - self.obs_history[idx].initial_pose_graph_compass
                    self.obs_history[idx].gps = self.obs_history[idx].gps + delta_gps
                    self.obs_history[idx].compass = self.obs_history[idx].compass + delta_compass

        t2 = timeit.default_timer()

        # Choose a “best” recent observation by sharpness and fuse into map
        if len(self.obs_history) > 0:
            obs_history = self.obs_history[-5:]
            blurs = [self.compute_blur_metric(obs.rgb) for obs in obs_history]
            obs = obs_history[blurs.index(max(blurs))]
        else:
            obs = None
        self._obs_history_lock.release()

        if obs is not None and self.robot.in_navigation_mode():
            self.voxel_map.process_rgbd_images(obs.rgb, obs.depth, obs.camera_K, obs.camera_pose)

        # Optional: build/update scene graph (relationships)
        if self.use_scene_graph:
            self._update_scene_graph()
            self.scene_graph.get_relationships()

        # GC: trim older, non-anchor observations
        self._obs_history_lock.acquire()
        if len(self.obs_history) > 500:
            del_count = 0
            del_idx = 0
            while del_count < 15 and len(self.obs_history) > 0 and del_idx < len(self.obs_history):
                if not self.obs_history[del_idx].is_pose_graph_node:
                    del self.obs_history[del_idx]
                    del_count += 1
                else:
                    del_idx += 1
                    if del_idx >= len(self.obs_history):
                        break
        self._obs_history_lock.release()

    def update(self):
        """
        Regular “pull one RGBD, fuse into voxel map” step.
        Called during rotate_in_place/look_around flows when realtime updates are disabled.
        """
        obs = self.robot.get_observation()
        self.obs_count += 1
        self.voxel_map.process_rgbd_images(obs.rgb, obs.depth, obs.camera_K, obs.camera_pose)

        # Stream to Rerun if we have new geometry
        if self.voxel_map.voxel_pcd._points is not None:
            self.rerun_visualizer.update_voxel_map(space=self.space)
        if self.voxel_map.semantic_memory._points is not None:
            self.rerun_visualizer.log_custom_pointcloud(
                "world/semantic_memory/pointcloud",
                self.voxel_map.semantic_memory._points.detach().cpu(),
                self.voxel_map.semantic_memory._rgb.detach().cpu() / 255.0,
                0.03,
            )

    # ------------------------------
    # Simple scanning behaviors
    # ------------------------------

    def look_around(self):
        """Small head sweep + map update to harvest more context nearby."""
        print("*" * 10, "Look around to check", "*" * 10)
        for pan in [0.6, -0.2, -1.0, -1.8]:
            self.robot.head_to(pan, -0.6, blocking=True)
            self.update()

    def rotate_in_place(self):
        """Base spin + updates: good initial map seeding."""
        print("*" * 10, "Rotate in place", "*" * 10)
        xyt = self.robot.get_base_pose()
        print("xyt: ", xyt)
        self.robot.head_to(head_pan=0, head_tilt=-0.6, blocking=True)
        for i in range(8):
            xyt[2] += 2 * np.pi / 8
            self.robot.move_base_to(xyt, blocking=True)
            if not self._realtime_updates:
                self.update()

    # ------------------------------
    # Navigation entrypoints
    # ------------------------------

    def execute_action(self, text: str):
        """
        Core “navigate” invocation:
          - If text is empty → frontier exploration
          - Else → try visual grounding to a target 3D point, then plan with A*
        Returns (finished: bool|None, end_point: np.ndarray|None)
        """
        if not self._realtime_updates:
            self.robot.look_front()
            self.look_around()
            self.robot.look_front()
            self.robot.switch_to_navigation_mode()

        self.robot.switch_to_navigation_mode()

        start = self.robot.get_base_pose()
        res = self.process_text(text, start)
        if len(res) == 0 and text != "" and text is not None:
            # fall back to exploration if grounded plan failed
            res = self.process_text("", start)

        if len(res) > 0:
            print("Plan successful!")
            # Convention: if the second-to-last entry is NaNs, the last element is the object point
            if len(res) >= 2 and np.isnan(res[-2]).all():
                if len(res) > 2:
                    self.robot.execute_trajectory(
                        res[:-2],
                        pos_err_threshold=self.pos_err_threshold,
                        rot_err_threshold=self.rot_err_threshold,
                        blocking=True,
                    )
                return True, res[-1]
            else:
                self.robot.execute_trajectory(
                    res,
                    pos_err_threshold=self.pos_err_threshold,
                    rot_err_threshold=self.rot_err_threshold,
                    blocking=True,
                )
                return False, None
        else:
            print("Failed. Try again!")
            return None, None

    def run_exploration(self):
        """One exploration step (frontier sample + plan)."""
        status, _ = self.execute_action("")
        if status is None:
            print("Exploration failed! Perhaps nowhere to explore!")
            return False
        return True

    def process_text(self, text, start_pose):
        """
        Turn a query string into a trajectory:
          1) Try re-using last tail of a saved traj if it still verifies
          2) If text provided → localize to a 3D point via voxel_map.localize_text()
          3) Else or if failed → sample a frontier
          4) Sample final nav waypoint near localized_point
          5) Plan A* → waypoints (cap to ~8 steps if long; save tail for resume)
          6) Construct trajectory; if the goal is an object, append [nan,nan,nan] + target
        """
        print("Processing", text, "starts")

        # Clear old viz channels
        self.rerun_visualizer.clear_identity("world/object")
        self.rerun_visualizer.clear_identity("world/robot_start_pose")
        self.rerun_visualizer.clear_identity("world/direction")
        self.rerun_visualizer.clear_identity("robot_monologue")
        self.rerun_visualizer.clear_identity("/observation_similar_to_text")

        debug_text = ""
        mode = "navigation"
        obs = None
        localized_point = None
        waypoints = None

        # Try to reuse tail of prior plan if text still verifies for that target
        if text is not None and text != "" and self.space.traj is not None:
            traj_target_point = self.space.traj[-1]
            if self.voxel_map.verify_point(text, traj_target_point):
                localized_point = traj_target_point
                debug_text += "## Last visual grounding results looks fine so directly use it.\n"

        print("Target verification finished")

        # Fresh localization via DynaMem if needed
        if text is not None and text != "" and localized_point is None:
            (
                localized_point,
                debug_text,
                obs,
                pointcloud,
            ) = self.voxel_map.localize_text(text, debug=True, return_debug=True)
            print("Target point selected!")

        # No text or failed grounding → frontier exploration
        if text is None or text == "" or localized_point is None:
            debug_text += "## Navigation fails, so robot starts exploring environments.\n"
            localized_point = self.space.sample_frontier(self.planner, start_pose, text)
            mode = "exploration"

        # If we have an associated obs, display it in Rerun
        if obs is not None and mode == "navigation":
            obs_id = self.voxel_map.find_obs_id_for_text(text)
            rgb = self.voxel_map.observations[obs_id - 1].rgb
            self.rerun_visualizer.log_custom_2d_image("/observation_similar_to_text", rgb)

        if localized_point is None:
            return []

        # Normalize to 3D
        if len(localized_point) == 2:
            localized_point = np.array([localized_point[0], localized_point[1], 0])

        # Choose a good nav endpoint near the localized point
        point = self.space.sample_navigation(start_pose, self.planner, localized_point)
        print("Navigation endpoint selected")

        # Plan with A*
        waypoints = None
        res = self.planner.plan(start_pose, point) if point is not None else None
        if res is not None and res.success:
            waypoints = [pt.state for pt in res.trajectory]
        elif res is not None:
            waypoints = None
            print("[FAILURE]", res.reason)

        # Build trajectory (with resume logic)
        traj = []
        if waypoints is not None:
            # Viz the object location
            self.rerun_visualizer.log_custom_pointcloud(
                "world/object",
                [localized_point[0], localized_point[1], 1.5],
                torch.Tensor([1, 0, 0]),
                0.1,
            )

            finished = len(waypoints) <= 8 and mode == "navigation"
            if finished:
                self.space.traj = None
            else:
                # Save tail of trajectory + final localized point for resume
                self.space.traj = waypoints[8:] + [[np.nan, np.nan, np.nan], localized_point]

            # Cap first chunk we’ll actually execute now
            if not finished:
                waypoints = waypoints[:8]

            traj = self.planner.clean_path_for_xy(waypoints)
            if finished:
                # Special marker + the 3D target point appended
                traj.append([np.nan, np.nan, np.nan])
                if isinstance(localized_point, torch.Tensor):
                    localized_point = localized_point.tolist()
                traj.append(localized_point)
            print("Planned trajectory:", traj)

        # Narrate plan intent
        if self.robot is not None:
            if text:
                self.robot.say("I am looking for a " + text + ".")
            else:
                self.robot.say("I am exploring the environment.")

        # Write monologue to Rerun
        if text:
            debug_text = "### The goal is to navigate to " + text + ".\n" + debug_text
        else:
            debug_text = (
                "### I have not received any text query from human user.\n"
                " ### So, I plan to explore the environment with Frontier-based exploration.\n"
            )
        debug_text = "# Robot's monologue: \n" + debug_text
        self.rerun_visualizer.log_text("robot_monologue", debug_text)

        # Draw arrows for the planned path
        if traj is not None:
            origins, vectors = [], []
            for idx in range(len(traj)):
                if idx != len(traj) - 1:
                    origins.append([traj[idx][0], traj[idx][1], 1.5])
                    vectors.append(
                        [traj[idx + 1][0] - traj[idx][0], traj[idx + 1][1] - traj[idx][1], 0]
                    )
            self.rerun_visualizer.log_arrow3D(
                "world/direction", origins, vectors, torch.Tensor([0, 1, 0]), 0.1
            )
            self.rerun_visualizer.log_custom_pointcloud(
                "world/robot_start_pose",
                [start_pose[0], start_pose[1], 1.5],
                torch.Tensor([0, 0, 1]),
                0.1,
            )

        return traj

    def navigate(self, text, max_step=10):
        """
        Keep executing_action until ‘finished’ or step budget exceeded.
        Returns end_point if succeeded, else None.
        """
        rr.init("Stretch_robot", recording_id=uuid4(), spawn=True)
        finished = False
        step = 0
        end_point = None
        while not finished and step < max_step:
            print("*" * 20, step, "*" * 20)
            step += 1
            finished, end_point = self.execute_action(text)
            if finished is None:
                print("Navigation failed! The path might be blocked!")
                return None
        return end_point

    # ------------------------------
    # Place / Pick wrappers (AnyGrasp flows)
    # ------------------------------

    def place(self, text, local=True, init_tilt=INIT_HEAD_TILT, base_node="camera_depth_optical_frame"):
        """
        Place the held object onto/into a target described by `text`.
        - local=True: OWL-SAM place detector on-board
        - local=False: offload to workstation manipulation server
        """
        self.robot.switch_to_manipulation_mode()
        self.robot.look_at_ee()
        self.manip_wrapper.move_to_position(head_pan=INIT_HEAD_PAN, head_tilt=init_tilt)

        if not local:
            rotation, translation = capture_and_process_image(
                mode="place", obj=text, socket=self.manip_socket, hello_robot=self.manip_wrapper,
            )
        else:
            if self.owl_sam_detector is None:
                from stretch.perception.detection.owl import OWLSAMProcessor
                self.owl_sam_detector = OWLSAMProcessor(confidence_threshold=0.1)
            rotation, translation = process_image_for_placing(
                obj=text, hello_robot=self.manip_wrapper, detection_model=self.owl_sam_detector, save_dir=self.log,
            )

        if rotation is None:
            return False

        # Pre-place posture
        self.manip_wrapper.move_to_position(lift_pos=1.05)
        self.manip_wrapper.move_to_position(wrist_yaw=0, wrist_pitch=0)

        # Move + open gripper
        move_to_point(self.manip_wrapper, translation, base_node, self.transform_node, move_mode=0)
        self.manip_wrapper.move_to_position(gripper_pos=1, blocking=True)

        # Wiggle wrist to release sticky objects
        self.manip_wrapper.move_to_position(
            lift_pos=min(self.manip_wrapper.robot.get_six_joints()[1] + 0.3, 1.1)
        )
        self.manip_wrapper.move_to_position(wrist_roll=2.5, blocking=True)
        self.manip_wrapper.move_to_position(wrist_roll=-2.5, blocking=True)

        # Retract & reset
        self.manip_wrapper.move_to_position(gripper_pos=1, lift_pos=1.05, arm_pos=0)
        self.manip_wrapper.move_to_position(wrist_pitch=-1.57)
        self.manip_wrapper.move_to_position(base_trans=-self.manip_wrapper.robot.get_six_joints()[0])
        return True

    def get_voxel_map(self):
        return self.voxel_map

    def manipulate(self, text, init_tilt=INIT_HEAD_TILT, base_node="camera_depth_optical_frame", skip_confirmation=False):
        """
        Pick up an object described by `text` using the AnyGrasp workstation server.
        """
        self.robot.switch_to_manipulation_mode()
        self.robot.look_at_ee()

        # Pre-pick posture
        gripper_pos = 1
        self.manip_wrapper.move_to_position(
            arm_pos=INIT_ARM_POS,
            head_pan=INIT_HEAD_PAN,
            head_tilt=init_tilt,
            gripper_pos=gripper_pos,
            lift_pos=INIT_LIFT_POS,
            wrist_pitch=INIT_WRIST_PITCH,
            wrist_roll=INIT_WRIST_ROLL,
            wrist_yaw=INIT_WRIST_YAW,
        )

        # Estimate grasp pose
        rotation, translation, depth, width = capture_and_process_image(
            mode="pick", obj=text, socket=self.manip_socket, hello_robot=self.manip_wrapper,
        )
        if rotation is None:
            return False

        # Choose gripper width based on detected object width and robot rev
        if width < 0.05 and self.re == 3:
            gripper_width = 0.45
        elif width < 0.075 and self.re == 3:
            gripper_width = 0.6
        else:
            gripper_width = 1

        # Confirm + execute pick
        if skip_confirmation or input("Do you want to do this manipulation? Y or N ") != "N":
            pickup(
                self.manip_wrapper,
                rotation,
                translation,
                base_node,
                self.transform_node,
                gripper_depth=depth,
                gripper_width=gripper_width,
            )

        # Return base to prior offset (we know that spot was navigable)
        self.manip_wrapper.move_to_position(base_trans=-self.manip_wrapper.robot.get_six_joints()[0])
        return True
