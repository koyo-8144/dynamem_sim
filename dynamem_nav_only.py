from typing import Optional, List, Tuple
import numpy as np
import cv2
import sys
sys.path.append("/home/koyo/stretch_ai/src")
from stretch.core import AbstractRobotClient, Parameters
from stretch.utils.logger import Logger
from stretch.agent.robot_agent_base import RobotAgentBase  # Simpler base class
from stretch.agent.task.emote import EmoteTask

logger = Logger(__name__)


class DynamemNavOnlyExecutor:
    def __init__(
        self,
        robot: AbstractRobotClient,
        parameters: Parameters,
        explore_iter: int = 5,
        output_path: Optional[str] = None,
        server_ip: Optional[str] = "127.0.0.1",
    ) -> None:
        self.robot = robot
        self.parameters = parameters
        self.explore_iter = explore_iter

        print("- Start navigation-only robot agent")
        self.agent = RobotAgentBase(
            robot=self.robot,
            parameters=self.parameters,
            semantic_sensor=None,
            log=output_path,
            server_ip=server_ip,
        )
        self.agent.start()
        self.emote_task = EmoteTask(self.agent)

    def __call__(self, response: List[Tuple[str, str]], channel=None) -> bool:
        for command, args in response:
            logger.info(f"[NavOnly] Command: {command} {args}")
            if command == "explore":
                for _ in range(self.explore_iter):
                    self.agent.run_exploration()
            elif command == "rotate_in_place":
                self.agent.rotate_in_place()
                self.agent.voxel_map.write_to_pickle(filename=None)
            elif command == "read_from_pickle":
                self.agent.voxel_map.read_from_pickle(args)
            elif command == "take_picture":
                obs = self.robot.get_observation()
                cv2.imwrite("stretch_snapshot.png", obs.rgb[:, :, ::-1])
            elif command == "quit":
                self.robot.stop()
                return False
            else:
                logger.warn(f"Unknown command {command}")
        return True
