# run_dynamem_sim.py

from stretch.dynamem_sim.test.sim_robot_client import SimRobotClient
from dynamem_nav_only import DynamemNavOnlyExecutor
import sys
sys.path.append("/home/koyo/stretch_ai/src")
from stretch.core.parameters import get_parameters

import click


@click.command()
@click.option("--explore-iter", default=5, help="Number of exploration iterations")
@click.option("--server-ip", default="127.0.0.1", help="Server IP for voxel/LLM server")
@click.option("--output-path", type=click.Path(), default=None, help="Where to save logs/maps")
@click.option("--input-path", type=click.Path(), default=None, help="Where to load a voxel map")
def main(explore_iter, server_ip, output_path, input_path):
    print("=== DynaMem Navigation-Only Simulation ===")

    print("- Loading parameters")
    parameters = get_parameters("dynav_config.yaml")

    print("- Initializing simulated robot client")
    robot = SimRobotClient()

    print("- Creating DynaMem Nav-Only Executor")
    executor = DynamemNavOnlyExecutor(
        robot=robot,
        parameters=parameters,
        explore_iter=explore_iter,
        output_path=output_path,
        server_ip=server_ip,
    )

    if input_path is not None:
        executor([("read_from_pickle", input_path)])

    # Run explore for N iterations
    executor([("explore", None)])

    print("- Finished exploration")


if __name__ == "__main__":
    main()
