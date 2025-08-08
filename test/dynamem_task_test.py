# Copyright (c) Hello Robot, Inc.
# All rights reserved.

import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Ops & agent orchestration
import sys
sys.path.append("/home/koyo/stretch_ai/src")
from stretch.agent.operations import GraspObjectOperation
# from stretch.agent.robot_agent_dynamem import RobotAgent
from stretch.agent.task.emote import EmoteTask
# from stretch.agent.task.pickup.hand_over_task import HandOverTask

# Robot + config
from stretch.core import Parameters
from stretch.core import AbstractRobotClient
from stretch.core.parameters import get_parameters

# Perception helpers (for optional visual servoing mode)
from stretch.perception import create_semantic_sensor

# Utils
from stretch.utils.image import numpy_image_to_bytes
from stretch.utils.logger import Logger

from robot_agent_dynamem_test import RobotAgentTest

logger = Logger(__name__)


# def compute_tilt(camera_xyz, target_xyz):
#     """
#     Compute a head-tilt angle that roughly looks from camera to a 3D target point.
#     Returns a negative tilt so the head pitches down toward closer/lower points.
#     """
#     if not isinstance(camera_xyz, np.ndarray):
#         camera_xyz = np.array(camera_xyz)
#     if not isinstance(target_xyz, np.ndarray):
#         target_xyz = np.array(target_xyz)
#     vector = camera_xyz - target_xyz
#     # atan2(z, horiz_distance). Negated because positive pitch is up.
#     return -np.arctan2(vector[2], np.linalg.norm(vector[:2]))


class DynamemTaskExecutorTest:
    """
    High-level task executor that glues together:
      - the ZMQ robot client (motion, head/arm control, obs)
      - the DynaMem RobotAgent (mapping, object localization, exploration)
      - optional grasp operation wrapper (visual servo)
      - small “tasks” (emotes, handover)
    It receives a list of (command, arg) pairs and executes them in order.
    """

    def __init__(
        self,
        robot: AbstractRobotClient,
        parameters: Parameters,
        match_method: str = "feature",
        visual_servo: bool = False,
        device_id: int = 0,
        output_path: Optional[str] = None,
        server_ip: Optional[str] = "127.0.0.1",
        skip_confirmations: bool = True,
        explore_iter: int = 5,
        mllm: bool = False,
        manipulation_only: bool = False,
        discord_bot=None,
    ) -> None:
        self.robot = robot
        self.parameters = parameters
        self.discord_bot = discord_bot

        # Behavior flags/knobs
        self.visual_servo = visual_servo
        self.match_method = match_method
        self.skip_confirmations = skip_confirmations
        self.explore_iter = explore_iter
        self.manipulation_only = manipulation_only

        # Type safety – the executor expects an abstract robot client interface
        if not isinstance(self.robot, AbstractRobotClient):
            raise TypeError(f"Expected AbstractRobotClient, got {type(self.robot)}")

        # Optional semantic sensor for visual servo flows (adds open-vocab features to obs)
        print("- Create semantic sensor if visual servoing is enabled")
        if self.visual_servo:
            self.semantic_sensor = create_semantic_sensor(
                parameters=self.parameters,
                device_id=device_id,
                verbose=False,
            )
        else:
            # make sure downstream doesn’t try to use an encoder if VS is disabled
            self.parameters["encoder"] = None
            self.semantic_sensor = None

        print("- Start robot agent with data collection")
        # RobotAgent = DynaMem brain:
        #   - fuses RGBD into voxel map
        #   - localizes open-vocab queries
        #   - picks goals/frontiers & plans
        #   - can run manipulation wrappers
        self.agent = RobotAgentTest(
            self.robot,
            self.parameters,
            self.semantic_sensor,
            log=output_path,
            server_ip=server_ip,
            mllm=mllm,
            manipulation_only=manipulation_only,
        )
        self.agent.start()

        # Optional grasp operation (visual servo), otherwise agent’s default manipulation
        if self.visual_servo:
            self.grasp_object = GraspObjectOperation("grasp_the_object", self.agent)
        else:
            self.grasp_object = None

        # Tiny “emote” task bundle (wave, nod, etc.)
        self.emote_task = EmoteTask(self.agent)

    # ------------------------------
    # Small helpers invoked by main()
    # ------------------------------

    def _find(self, target_object: str) -> np.ndarray:
        """
        Navigate to an object name using DynaMem’s open-vocab localization + planner.
        Returns a 3D point for the object if successful, else None.
        """
        self.robot.switch_to_navigation_mode()
        point = self.agent.navigate(target_object)

        # Persist spatio-semantic memory after each navigate attempt (for analysis/replay)
        self.agent.voxel_map.write_to_pickle(filename=None)

        if point is None:
            logger.error(f"Navigation Failure: Could not find the object {target_object}")
            return None

        # Dump a snapshot for debugging
        cv2.imwrite(target_object + ".jpg", self.robot.get_observation().rgb[:, :, [2, 1, 0]])

        # Orient base to face the object a bit more (quarter turn)
        self.robot.switch_to_navigation_mode()
        xyt = self.robot.get_base_pose()
        xyt[2] = xyt[2] + np.pi / 2
        self.robot.move_base_to(xyt, blocking=True)
        return point

    def _pickup(
        self,
        target_object: str,
        point: Optional[np.ndarray] = None,
        skip_confirmations: bool = False,
    ) -> None:
        """
        Pick up an object. If visual servo is enabled, run GraspObjectOperation; else
        use RobotAgent’s manipulation routine (e.g., AnyGrasp flow in OK-Robot style).
        """
        self.robot.switch_to_manipulation_mode()
        camera_xyz = self.robot.get_head_pose()[:3, 3]
        theta = compute_tilt(camera_xyz, point) if point is not None else -0.6

        if self.grasp_object is not None:
            # Visual-servo path: grasp via servo loop GUI (debug-friendly)
            self.robot.say("Grasping the " + str(target_object) + ".")
            print("Using operation to grasp object:", target_object)
            print(" - Point:", point)
            print(" - Theta:", theta)
            state = self.robot.get_six_joints()
            state[1] = 1.0  # lift a bit for clearance
            self.robot.arm_to(state, blocking=True)
            self.grasp_object(
                target_object=target_object,
                object_xyz=point,
                match_method="feature",
                show_object_to_grasp=False,
                show_servo_gui=True,
                delete_object_after_grasp=False,
            )
            self.robot.move_to_nav_posture()  # retract/tuck
        else:
            # Default manipulation path via agent
            print("Using self.agent to grasp object:", target_object)
            self.agent.manipulate(target_object, theta, skip_confirmation=skip_confirmations)
        self.robot.look_front()

    def _take_picture(self, channel=None) -> None:
        """Snapshot from head RGB. Optional Discord send."""
        obs = self.robot.get_observation()
        if channel is None:
            now = datetime.datetime.now()
            filename = f"stretch_image_{now.strftime('%Y-%m-%d_%H-%M-%S')}.png"
            Image.fromarray(obs.rgb).save(filename)
        else:
            self.discord_bot.send_message(
                channel=channel, message="Head camera:", content=numpy_image_to_bytes(obs.rgb)
            )

    def _take_ee_picture(self, channel=None) -> None:
        """Snapshot from EE RGB. Optional Discord send."""
        obs = self.robot.get_servo_observation()
        if channel is None:
            now = datetime.datetime.now()
            filename = f"stretch_image_{now.strftime('%Y-%m-%d_%H-%M-%S')}.png"
            Image.fromarray(obs.ee_rgb).save(filename)
        else:
            self.discord_bot.send_message(
                channel=channel,
                message="End effector camera:",
                content=numpy_image_to_bytes(obs.ee_rgb),
            )

    def _place(self, target_receptacle: str, point: Optional[np.ndarray]) -> None:
        """
        Place into/onto a receptacle found with DynaMem. Uses the agent’s
        place() which can run entirely on-board (local=True) using OWL-SAM.
        """
        self.robot.switch_to_manipulation_mode()
        camera_xyz = self.robot.get_head_pose()[:3, 3]
        theta = compute_tilt(camera_xyz, point) if point is not None else -0.6

        self.robot.say("Placing object on the " + str(target_receptacle) + ".")
        self.agent.place(target_receptacle, init_tilt=theta, local=self.visual_servo)
        self.robot.move_to_nav_posture()

    def _hand_over(self) -> None:
        """Find a person and present the object (simple handover demo)."""
        logger.alert(f"[Pickup task] Hand Over")
        try:
            hand_over_task = HandOverTask(self.agent)
            task = hand_over_task.get_task()
        except Exception as e:
            print(f"Error creating task: {e}")
            self.robot.stop()
            raise e
        task.run()

    # ------------------------------
    # Main entry: execute scripted/LLM commands
    # ------------------------------

    def __call__(self, response: List[Tuple[str, str]], channel=None) -> bool:
        """
        Execute a list of commands like:
          [("explore", None)] or
          [("pickup", "blue cup"), ("place", "sink")] …
        Returns False if a “quit” command is processed; True otherwise.
        """
        i = 0

        if response is None or len(response) == 0:
            logger.error("No commands to execute!")
            self.agent.robot_say("I'm sorry, I didn't understand that.")
            return True

        # We do NOT reset the agent’s memory each call (lifelong operation)

        while i < len(response):
            command, args = response[i]
            logger.info(f"Command: {i} {command} {args}")

            if command == "say":
                logger.info(f"Saying: {args}")
                self.agent.robot_say(args)
                if channel is not None:
                    # strip quotes if they were passed along
                    if args and args[0] == '"' and args[-1] == '"':
                        args = args[1:-1]
                    self.discord_bot.send_message(channel=channel, message=args)

            elif command == "pickup":
                # Navigate → pick
                logger.info(f"[Pickup task] Pickup: {args}")
                target_object = args

                if self.skip_confirmations or (
                    not self.skip_confirmations
                    and input("Do you want to run navigation? [Y/n]: ").upper() != "N"
                ):
                    self.robot.move_to_nav_posture()
                    point = self._find(args)
                else:
                    point = None

                # If we couldn’t find it, bail out early
                if self.skip_confirmations:
                    if point is not None:
                        self._pickup(target_object, point=point)
                    else:
                        logger.error("Could not find the object.")
                        self.robot.say("I could not find the " + str(args) + ".")
                        i += 1
                        continue
                else:
                    if input("Do you want to run picking? [Y/n]: ").upper() != "N":
                        self._pickup(target_object, point=point)
                    else:
                        logger.info("Skip picking!")
                        i += 1
                        continue

            elif command == "place":
                # Navigate → place
                logger.info(f"[Pickup task] Place: {args}")
                target_object = args

                if self.skip_confirmations or (
                    not self.skip_confirmations
                    and input("Do you want to run navigation? [Y/n]: ").upper() != "N"
                ):
                    point = self._find(args)
                else:
                    point = None

                if self.skip_confirmations:
                    if point is not None:
                        self._place(target_object, point=point)
                    else:
                        logger.error("Could not find the object.")
                        self.robot.say("I could not find the " + str(args) + ".")
                        i += 1
                        continue
                else:
                    if input("Do you want to run placement? [Y/n]: ").upper() != "N":
                        self._place(target_object, point=point)
                    else:
                        logger.info("Skip placing!")
                        i += 1
                        continue

            elif command == "hand_over":
                self._hand_over()

            elif command == "wave":
                logger.info("[Pickup task] Waving.")
                self.agent.move_to_manip_posture()
                self.emote_task.get_task("wave").run()
                self.agent.move_to_manip_posture()

            elif command == "rotate_in_place":
                logger.info("Rotate in place to scan environments.")
                self.agent.rotate_in_place()
                self.agent.voxel_map.write_to_pickle(filename=None)

            elif command == "read_from_pickle":
                logger.info(f"Load the semantic memory from past runs, pickle file name: {args}.")
                self.agent.voxel_map.read_from_pickle(args)

            elif command == "go_home":
                logger.info("[Pickup task] Going home.")
                if self.agent.get_voxel_map().is_empty():
                    logger.warning("No map data available. Cannot go home.")
                else:
                    self.agent.go_home()

            elif command == "explore":
                logger.info("[Pickup task] Exploring.")
                for _ in range(self.explore_iter):
                    self.agent.run_exploration()

            elif command == "find":
                logger.info(f"[Pickup task] Finding {args}.")
                _ = self._find(args)

            elif command == "nod_head":
                logger.info("[Pickup task] Nodding head.")
                self.emote_task.get_task("nod_head").run()

            elif command == "shake_head":
                logger.info("[Pickup task] Shaking head.")
                self.emote_task.get_task("shake_head").run()

            elif command == "avert_gaze":
                logger.info("[Pickup task] Averting gaze.")
                self.emote_task.get_task("avert_gaze").run()

            elif command == "take_picture":
                self._take_picture(channel)

            elif command == "take_ee_picture":
                self._take_ee_picture(channel)

            elif command == "quit":
                logger.info("[Pickup task] Quitting.")
                self.robot.stop()
                return False

            elif command == "end":
                logger.info("[Pickup task] Ending.")
                break

            else:
                logger.error(f"Skipping unknown command: {command}")

            i += 1

        return True  # keep main loop alive unless we saw "quit"


