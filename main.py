"""
Main training script for the SUMO Deep Reinforcement Learning project.

This script handles the training of a DQN agent for autonomous vehicle routing in SUMO.
It supports two types of agents:
1. 'vanilla': A standard DQN agent.
2. 'enhanced': A Dueling Double DQN agent with Prioritized Experience Replay (PER).

Usage:
    python main.py --agent-type [vanilla|enhanced]
"""

import argparse
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import traci
import yaml

from enhanced.dqn_agent import DQNAgent as EnhancedAgent
from vanilla.dqn_agent import DQNAgent as VanillaAgent

# Global map for SUMO lane data
edge_to_lanes_map: Dict[str, List[str]] = defaultdict(list)


def parse_config(config_path: str) -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_tripinfo_xml(file_path: str) -> Optional[Dict[str, float]]:
    """
    Parses a SUMO tripinfo XML file to extract performance metrics.

    Args:
        file_path (str): The path to the tripinfo.xml file.

    Returns:
        Optional[Dict[str, float]]: A dictionary of metrics or None if parsing fails.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except (FileNotFoundError, ET.ParseError):
        return None

    data_template = {
        "duration": [],
        "routeLength": [],
        "waitingTime": [],
        "timeLoss": [],
        "rerouteNo": [],
    }
    hv_data = {k: list(v) for k, v in data_template.items()}
    av_data = {k: list(v) for k, v in data_template.items()}

    for trip in root.findall("tripinfo"):
        try:
            vtype = trip.get("vType", "HV")
            data = av_data if vtype == "AV" else hv_data
            data["duration"].append(float(trip.get("duration", 0)))
            data["routeLength"].append(float(trip.get("routeLength", 0)))
            data["waitingTime"].append(float(trip.get("waitingTime", 0)))
            data["timeLoss"].append(float(trip.get("timeLoss", 0)))
            data["rerouteNo"].append(float(trip.get("rerouteNo", 0)))
        except (ValueError, TypeError):
            continue  # Skip trips with invalid data

    def avg(lst: List[float]) -> float:
        return np.mean(lst) if lst else 0.0

    return {
        "num_HV": float(len(hv_data["duration"])),
        "num_AV": float(len(av_data["duration"])),
        "avg_dur_HV": avg(hv_data["duration"]),
        "avg_dur_AV": avg(av_data["duration"]),
        "avg_wait_HV": avg(hv_data["waitingTime"]),
        "avg_wait_AV": avg(av_data["waitingTime"]),
        "avg_loss_HV": avg(hv_data["timeLoss"]),
        "avg_loss_AV": avg(av_data["timeLoss"]),
        "avg_len_HV": avg(hv_data["routeLength"]),
        "avg_len_AV": avg(av_data["routeLength"]),
        "avg_reroute_AV": avg(av_data["rerouteNo"]),
    }


def initialize_lane_map():
    """Initializes a global mapping from edge IDs to lane IDs."""
    global edge_to_lanes_map
    edge_to_lanes_map.clear()
    try:
        all_lane_ids = traci.lane.getIDList()
        for lane_id in all_lane_ids:
            edge_id = traci.lane.getEdgeID(lane_id)
            if not edge_id.startswith(":"):  # Ignore internal lanes
                edge_to_lanes_map[edge_id].append(lane_id)
    except traci.TraCIException:
        print("Error initializing lane map.", file=sys.stderr)


def get_lane_from_edge(dest_edge: str) -> Optional[str]:
    """Gets the first lane ID for a given edge ID."""
    lanes = edge_to_lanes_map.get(dest_edge)
    return lanes[0] if lanes else None


def get_state(
    vehicle_id: str, destination_pos: np.ndarray, norm_conf: Dict[str, float]
) -> Optional[np.ndarray]:
    """
    Constructs the state vector for a given vehicle.

    The state includes traffic conditions on the current, left, right, and
    straight-ahead roads, as well as positional information.

    Args:
        vehicle_id (str): The ID of the vehicle.
        destination_pos (np.ndarray): The (x, y) coordinates of the destination.
        norm_conf (Dict[str, float]): Normalization constants.

    Returns:
        Optional[np.ndarray]: The state vector or None if state cannot be constructed.
    """
    try:
        current_lane_id = traci.vehicle.getLaneID(vehicle_id)
        current_edge_id = traci.lane.getEdgeID(current_lane_id)
    except traci.TraCIException:
        return None

    # Get connected edges
    road_ids = {"current": current_edge_id}
    try:
        links = traci.lane.getLinks(current_lane_id)
        for link in links:
            to_lane, direction = link[0], link[6]
            to_edge = traci.lane.getEdgeID(to_lane)
            if "r" in direction:
                road_ids["right"] = to_edge
            elif "s" in direction:
                road_ids["straight"] = to_edge
            elif "l" in direction:
                road_ids["left"] = to_edge
    except traci.TraCIException:
        pass  # No links found, proceed with available info

    # Gather traffic features from each road
    traffic_features = []
    for road_key in ["current", "right", "straight", "left"]:
        edge_id = road_ids.get(road_key)
        av_count, hv_count = 0, 0
        av_speeds, hv_speeds = [], []

        if edge_id:
            try:
                for v_id in traci.edge.getLastStepVehicleIDs(edge_id):
                    v_type = traci.vehicle.getTypeID(v_id)
                    v_speed = traci.vehicle.getSpeed(v_id)
                    if v_type == "AV":
                        av_count += 1
                        av_speeds.append(v_speed)
                    else:
                        hv_count += 1
                        hv_speeds.append(v_speed)
            except traci.TraCIException:
                pass  # Vehicle may have left simulation

        traffic_features.extend(
            [
                av_count,
                hv_count,
                np.mean(av_speeds) if av_speeds else 0.0,
                np.mean(hv_speeds) if hv_speeds else 0.0,
            ]
        )

    # Get positional features
    try:
        current_pos = traci.vehicle.getPosition(vehicle_id)
        positional_features = [
            current_pos[0],
            current_pos[1],
            destination_pos[0],
            destination_pos[1],
        ]
    except traci.TraCIException:
        return None

    # Combine and normalize state vector
    state_vector = np.array(traffic_features + positional_features, dtype=np.float32)
    state_vector[0:16:4] /= norm_conf["vehicle_count"]
    state_vector[1:16:4] /= norm_conf["vehicle_count"]
    state_vector[2:16:4] /= norm_conf["max_speed"]
    state_vector[3:16:4] /= norm_conf["max_speed"]
    state_vector[16:] /= norm_conf["grid_size"]

    return state_vector


def get_reward(
    vehicle_id: str,
    initial_dist_to_dest: float,
    destination_pos: np.ndarray,
    reward_conf: Dict[str, float],
) -> float:
    """
    Calculates the reward for a vehicle's current state.

    The reward is a weighted sum of normalized speed and normalized
    reduction in distance to the destination.

    Args:
        vehicle_id (str): The ID of the vehicle.
        initial_dist_to_dest (float): The initial distance to the destination.
        destination_pos (np.ndarray): The (x, y) coordinates of the destination.
        reward_conf (Dict[str, float]): Reward function parameters.

    Returns:
        float: The calculated reward.
    """
    try:
        # Normalized speed component
        current_speed = traci.vehicle.getSpeed(vehicle_id)
        max_speed = traci.vehicle.getMaxSpeed(vehicle_id)
        ds_n = current_speed / max_speed if max_speed > 0 else 0.0

        # Normalized distance component
        current_pos = np.array(traci.vehicle.getPosition(vehicle_id))
        current_dist_to_dest = np.linalg.norm(current_pos - destination_pos)
        dd_n = (
            (initial_dist_to_dest - current_dist_to_dest) / initial_dist_to_dest
            if initial_dist_to_dest > 0
            else 0.0
        )

        omega = reward_conf["omega"]
        return (omega * ds_n) + ((1 - omega) * dd_n)
    except (traci.TraCIException, ValueError):
        return 0.0


def execute_action(vehicle_id: str, action: int):
    """
    Executes a routing action for a vehicle.

    Args:
        vehicle_id (str): The ID of the vehicle.
        action (int): The action to take (0: right, 1: straight, 2: left).
    """
    action_to_direction = {0: "r", 1: "s", 2: "l"}
    chosen_direction = action_to_direction.get(action)
    if not chosen_direction:
        return

    try:
        current_lane = traci.vehicle.getLaneID(vehicle_id)
        links = traci.lane.getLinks(current_lane)
        for link in links:
            if link[6] == chosen_direction:
                target_edge = traci.lane.getEdgeID(link[0])
                traci.vehicle.changeTarget(vehicle_id, target_edge)
                return
    except traci.TraCIException:
        pass


def initialize_agent(agent_type: str, config: Dict[str, Any]):
    """Initializes and returns the specified DQN agent."""
    state_size = config["training"]["state_size"]
    action_size = config["training"]["action_size"]
    agent_config = {**config["agent"], **config[agent_type]}

    if agent_type == "vanilla":
        return VanillaAgent(state_size, action_size, agent_config, config["training"])
    elif agent_type == "enhanced":
        return EnhancedAgent(state_size, action_size, agent_config, config["training"])
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def log_episode_metrics(
    csv_path: str, episode_metrics: Dict[str, Any], first_episode: bool
):
    """Appends episode metrics to a CSV file."""
    df = pd.DataFrame([episode_metrics])
    if first_episode:
        df.to_csv(csv_path, index=False, header=True)
    else:
        df.to_csv(csv_path, index=False, header=False, mode="a")


def plot_results(csv_path: str, agent_type: str):
    """Generates and saves plots from the training metrics CSV."""
    df = pd.read_csv(csv_path)
    episodes = df["episode"]

    def smooth(series, window=10):
        return series.rolling(window, min_periods=1).mean()

    plot_configs = {
        "reward": ("Training Reward Progress", "Reward", "episode_reward"),
        "duration": (
            "Average Trip Duration",
            "Avg Trip Duration (s)",
            ["avg_dur_AV", "avg_dur_HV"],
        ),
        "waiting": (
            "Average Waiting Time",
            "Average Waiting Time (s)",
            ["avg_wait_AV", "avg_wait_HV"],
        ),
        "timeloss": (
            "Average Time Loss",
            "Average Time Loss (s)",
            ["avg_loss_AV", "avg_loss_HV"],
        ),
        "reroutes": ("Average AV Reroutes", "Avg Reroute Count", "avg_reroute_AV"),
    }

    for key, (title, ylabel, series_keys) in plot_configs.items():
        plt.figure(figsize=(12, 7))
        if isinstance(series_keys, list):
            plt.plot(episodes, smooth(df[series_keys[0]]), label="AV")
            plt.plot(episodes, smooth(df[series_keys[1]]), label="HV")
        else:
            plt.plot(episodes, smooth(df[series_keys]), label=ylabel)

        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{agent_type}_{key}_progress.png", bbox_inches="tight")
        plt.close()


def train(agent, config: Dict[str, Any], agent_type: str):
    """Main training loop."""
    scores = []
    checkpoint_dir = f"checkpoints/{agent_type}_dqn"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load from latest checkpoint if available
    start_episode = 1
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if ckpts:
        latest_ckpt_name = sorted(
            ckpts, key=lambda x: int(x.replace("checkpoint_ep", "").replace(".pth", ""))
        )[-1]
        latest_checkpoint = os.path.join(checkpoint_dir, latest_ckpt_name)
        if agent.load(latest_checkpoint):
            start_episode = int(latest_ckpt_name.split("ep")[-1].split(".")[0]) + 1

    csv_path = f"{agent_type}_training_metrics.csv"

    for episode in range(start_episode, config["training"]["num_episodes"] + 1):
        start_time = time.time()

        # Start SUMO simulation for the episode
        sumo_cmd = [
            "sumo",
            "-c",
            config["sumo"]["cfg_file"],
            "--tripinfo-output",
            config["sumo"]["tripinfo_output"],
            "--seed",
            str(random.randint(1, 2_000_000_000)),
            "--time-to-teleport",
            str(config["sumo"]["time_to_teleport"]),
        ]
        try:
            traci.start(sumo_cmd)
        except traci.TraCIException as e:
            print(f"Error starting SUMO: {e}", file=sys.stderr)
            break

        initialize_lane_map()

        step = 0
        vehicle_dests: Dict[str, np.ndarray] = {}
        vehicle_initial_dists: Dict[str, float] = {}
        last_decision_data: Dict[str, Tuple[np.ndarray, int]] = {}
        vehicle_episode_rewards: Dict[str, float] = defaultdict(float)

        while (
            traci.simulation.getMinExpectedNumber() > 0
            and step < config["sumo"]["max_steps"]
        ):
            traci.simulationStep()
            step += 1

            # Handle departed vehicles
            for v_id in traci.simulation.getDepartedIDList():
                if traci.vehicle.getTypeID(v_id) == "AV":
                    try:
                        route = traci.vehicle.getRoute(v_id)
                        if not route:
                            continue
                        dest_edge = route[-1]
                        lane_id = get_lane_from_edge(dest_edge)
                        if not lane_id:
                            continue

                        dest_pos = np.array(traci.lane.getShape(lane_id)[-1])
                        pos = np.array(traci.vehicle.getPosition(v_id))

                        vehicle_dests[v_id] = dest_pos
                        vehicle_initial_dists[v_id] = np.linalg.norm(pos - dest_pos)
                    except (traci.TraCIException, IndexError):
                        continue

            # Process active AVs
            active_avs = [
                v_id for v_id in traci.vehicle.getIDList() if v_id in vehicle_dests
            ]
            for v_id in active_avs:
                try:
                    if traci.vehicle.getLaneID(v_id).startswith(":"):
                        continue

                    lane_length = traci.lane.getLength(traci.vehicle.getLaneID(v_id))
                    pos_on_lane = traci.vehicle.getLanePosition(v_id)

                    if (lane_length - pos_on_lane) <= config["sumo"][
                        "decision_zone_length"
                    ]:
                        current_state = get_state(
                            v_id, vehicle_dests[v_id], config["normalization"]
                        )
                        if current_state is None:
                            continue

                        reward = get_reward(
                            v_id,
                            vehicle_initial_dists[v_id],
                            vehicle_dests[v_id],
                            config["reward"],
                        )
                        vehicle_episode_rewards[v_id] += reward

                        if v_id in last_decision_data:
                            prev_state, prev_action = last_decision_data[v_id]
                            agent.step(
                                prev_state, prev_action, reward, current_state, False
                            )

                        action = agent.act(current_state)
                        execute_action(v_id, action)
                        last_decision_data[v_id] = (current_state, action)
                except traci.TraCIException:
                    continue

            # Handle arrived vehicles
            for v_id in traci.simulation.getArrivedIDList():
                if v_id in last_decision_data:
                    prev_state, prev_action = last_decision_data[v_id]
                    final_reward = config["reward"]["final_reward"]
                    vehicle_episode_rewards[v_id] += final_reward
                    agent.step(prev_state, prev_action, final_reward, prev_state, True)
                    del last_decision_data[v_id]

                vehicle_dests.pop(v_id, None)
                vehicle_initial_dists.pop(v_id, None)

        traci.close()
        agent.update_epsilon(episode)

        # Logging and saving
        episode_reward = sum(vehicle_episode_rewards.values())
        scores.append(episode_reward)
        episode_metrics = parse_tripinfo_xml(config["sumo"]["tripinfo_output"])

        if episode_metrics:
            episode_metrics["episode"] = episode
            episode_metrics["episode_reward"] = episode_reward
            episode_metrics["epsilon"] = agent.epsilon
            log_episode_metrics(
                csv_path, episode_metrics, episode == 1 and start_episode == 1
            )

        if episode % 100 == 0 or episode == config["training"]["num_episodes"]:
            agent.save(os.path.join(checkpoint_dir, f"checkpoint_ep{episode}.pth"))

        duration = time.time() - start_time
        print(
            f"Episode {episode}/{config['training']['num_episodes']} | Reward: {episode_reward:.2f} | Epsilon: {agent.epsilon:.4f} | Duration: {duration:.2f}s"
        )

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent for SUMO.")
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=["vanilla", "enhanced"],
        required=True,
        help="The type of DQN agent to train.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    # Load configuration and initialize agent
    config = parse_config(args.config)
    agent = initialize_agent(args.agent_type, config)

    # Start training
    training_scores = train(agent, config, args.agent_type)

    # Plot final results
    print("Training finished. Generating plots...")
    plot_results(f"{args.agent_type}_training_metrics.csv", args.agent_type)
    print("Plots saved.")
