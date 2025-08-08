# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.



# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from typing import Optional
import click

# Task executor that implements DynaMem navigation/exploration (and optionally manipulation).
# from stretch.agent.task.dynamem import DynamemTaskExecutor

# ZMQ-based robot client (receives obs, sends actions) – works with the ROS2 bridge server.
# from stretch.agent.zmq_client import HomeRobotZmqClient

import sys
sys.path.append("/home/koyo/stretch_ai/src")
# Loads YAML parameters used throughout (motion thresholds, joint tolerances, planning defaults, etc.)
from stretch.core.parameters import get_parameters

# LLM wiring (optional; not needed if you’re driving via keyboard/menu)
from stretch.llms import (
    LLMChatWrapper,
    PickupPromptBuilder,
    get_llm_choices,
    get_llm_client,
)


# sys.path.append("/home/koyo/stretch_ai/src/stretch/dynamem_sim")
from dynamem_task_test import DynamemTaskExecutorTest
from dummy_stretch_client_test import DummyStretchClientTest
from dynamem_ros2_client import DynaMemROS2Client


@click.command()
# CLI options for running locally vs robot, toggling LLM/voice, method choice, etc.
@click.option("--server_ip", "--server-ip", default="127.0.0.1", type=str)
@click.option("--manual-wait", default=False, is_flag=True)
@click.option("--random-goals", default=False, is_flag=True)
@click.option("--explore-iter", default=3)
@click.option("--method", default="dynamem", type=str)
# Mode gate lets the executor know whether to run only navigation, only manipulation, save, or both.
@click.option("--mode", default="", type=click.Choice(["navigation", "manipulation", "save", ""]))
@click.option(
    "--use_llm",
    "--use-llm",
    is_flag=True,
    help="Set to use the language model",
)
@click.option(
    "--llm",
    # default="gemma2b",
    default="qwen25-3B-Instruct",
    help="Client to use for language model. Recommended: gemma2b, openai",
    type=click.Choice(get_llm_choices()),
)
@click.option("--debug_llm", "--debug-llm", is_flag=True, help="Set to debug the language model")
@click.option(
    "--use_voice",
    "--use-voice",
    is_flag=True,
    help="Set to use voice input",
)
# Visual servo (for manipulation), safe to leave off if only exploring/navigating
@click.option(
    "--visual_servo",
    "--vs",
    "-V",
    "--visual-servo",
    default=False,
    is_flag=True,
    help="Use visual servoing grasp",
)
# If you ran the ROS2 bridge on a robot or VM with a different IP, pass it here
@click.option(
    "--robot_ip", type=str, default="", help="Robot IP address (leave empty for saved default)"
)
@click.option("--target_object", type=str, default=None, help="Target object to grasp")
@click.option(
    "--target_receptacle", "--receptacle", type=str, default=None, help="Target receptacle to place"
)
@click.option(
    "--skip_confirmations",
    "--skip",
    "-S",
    "-y",
    "--yes",
    is_flag=True,
    help="Skip many confirmations",
)
@click.option(
    "--input-path",
    type=click.Path(),
    default=None,
    help="Input path with default value None",
)
@click.option(
    "--output-path",
    type=click.Path(),
    default=None,
    help="Input path with default value None",
)
@click.option(
    "--match-method",
    "--match_method",
    type=click.Choice(["class", "feature"]),
    default="feature",
    help="feature for visual servoing",
)
# mLLM toggles (for visual grounding via multimodal LLM)
@click.option(
    "--mllm-for-visual-grounding",
    "--mllm",
    "-M",
    is_flag=True,
    help="Use GPT4o for visual grounding",
)
@click.option("--device_id", default=0, type=int, help="Device ID for semantic sensor")
@click.option(
    "--manipulation-only", "--manipulation", is_flag=True, help="For debugging manipulation"
)
def main(
    server_ip,
    manual_wait,
    explore_iter: int = 3,
    mode: str = "navigation",
    method: str = "dynamem",
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    robot_ip: str = "",
    visual_servo: bool = False,
    skip_confirmations: bool = True,
    device_id: int = 0,
    target_object: str = None,
    target_receptacle: str = None,
    use_llm: bool = False,
    use_voice: bool = False,
    debug_llm: bool = False,
    llm: str = "qwen25-3B-Instruct",
    manipulation_only: bool = False,
    **kwargs,
):
    """
    Entry point for the DynaMem demo app.
    High level flow:
      1) Load parameters
      2) Create ZMQ client to talk to the ROS2 bridge
      3) Build a DynamemTaskExecutor to run exploration / navigation / (optional manipulation)
      4) If no LLM: ask the user what to do via terminal; else pass LLM output as commands to executor
    """

    print("- Load parameters")
    # Reads src/stretch/config/dynav_config.yaml (mapping, exploration, thresholds, etc.)
    parameters = get_parameters("dynav_config.yaml")
    print("parameters: ", parameters)

    print("- Create robot client")
    # ZMQ client spawns threads and starts receiving obs + state from the ROS2 bridge.
    # robot = HomeRobotZmqClient(robot_ip=robot_ip)
    # robot = DummyStretchClientTest()
    robot = DynaMemROS2Client()
    print("robot: ", robot)

    print("- Create task executor")
    # The executor is the high level “brain”:
    #   - builds/updates DynaMem voxel map
    #   - selects exploration goals/frontiers
    #   - calls robot.move_base_to() to navigate
    #   - (optionally) visual grounding + manipulation
    executor = DynamemTaskExecutorTest(
        robot,
        parameters,
        visual_servo=visual_servo,
        match_method=kwargs["match_method"],
        device_id=device_id,
        output_path=output_path,
        server_ip=server_ip,
        skip_confirmations=skip_confirmations,
        mllm=kwargs["mllm_for_visual_grounding"],
        manipulation_only=manipulation_only,
    )
    print("executor: ", executor)

    # If we’re not doing manipulation-only, start with a map seed:
    #  - default: rotate in place to get a panoramic scan (RGB-D swept into the voxel map)
    #  - or, load observations from a pickle
    if not manipulation_only:
        if input_path is None:
            start_command = [("rotate_in_place", "")]
        else:
            start_command = [("read_from_pickle", input_path)]
        executor(start_command)

    # If you want LLM “chat to control robot”, set use_llm=True.
    # For pure exploration without LLM, the loop below prompts once per turn.
    prompt = PickupPromptBuilder()

    llm_client = None
    if use_llm:
        llm_client = get_llm_client(llm, prompt=prompt)
        chat_wrapper = LLMChatWrapper(llm_client, prompt=prompt, voice=use_voice)

    # Control loop: either prompt the user (no LLM) or parse LLM outputs into commands
    ok = True
    while ok:
        say_this = None
        if llm_client is None:
            # Manual command mode – simple CLI to choose between exploration or PnP
            explore = input(
                "Enter desired mode [E (explore and mapping) / M (Open vocabulary pick and place)]: "
            )
            if explore.upper() == "E":
                # Executor expects a list of (verb, arg) tuples
                llm_response = [("explore", None)]
            else:
                # Minimal text inputs for manipulation targets
                if target_object is None or len(target_object) == 0:
                    target_object = input("Enter the target object: ")
                if target_receptacle is None or len(target_receptacle) == 0:
                    target_receptacle = input("Enter the target receptacle: ")
                llm_response = [("pickup", target_object), ("place", target_receptacle)]
        else:
            # LLM-driven control: ask the model for the next action(s)
            llm_response = chat_wrapper.query(verbose=debug_llm)
            if debug_llm:
                print("Parsed LLM Response:", llm_response)

        # Execute the parsed command list. Executor returns False to quit.
        ok = executor(llm_response)
        # Reset targets so manual mode asks again next loop
        target_object = None
        target_receptacle = None

    # Clean shutdown
    robot.stop()


if __name__ == "__main__":
    main()
