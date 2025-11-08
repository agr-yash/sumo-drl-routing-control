import os
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import traci

from .dqn_agent import DQNAgent

SUMO_CFG = "simulation.sumocfg"
SUMO_CMD = ["sumo", "-c", SUMO_CFG, "--tripinfo-output", "tripinfo.xml"]

STATE_SIZE = 20
ACTION_SIZE = 3
NUM_EPISODES = 500
MAX_STEPS = 5000
DECISION_ZONE_LENGTH = 50

all_lane_ids = []
edge_to_lanes_map = defaultdict(list)


def parse_tripinfo_xml(file_path="tripinfo.xml"):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception:
        return None

    hv_data, av_data = (
        {
            "duration": [],
            "routeLength": [],
            "waitingTime": [],
            "timeLoss": [],
            "rerouteNo": [],
        },
        {
            "duration": [],
            "routeLength": [],
            "waitingTime": [],
            "timeLoss": [],
            "rerouteNo": [],
        },
    )

    for trip in root.findall("tripinfo"):
        vtype = trip.get("vType", "HV")
        data = av_data if vtype == "AV" else hv_data
        try:
            data["duration"].append(float(trip.get("duration", 0)))
            data["routeLength"].append(float(trip.get("routeLength", 0)))
            data["waitingTime"].append(float(trip.get("waitingTime", 0)))
            data["timeLoss"].append(float(trip.get("timeLoss", 0)))
            data["rerouteNo"].append(float(trip.get("rerouteNo", 0)))
        except Exception:
            continue

    def avg(lst):
        return np.mean(lst) if lst else 0.0

    def count(lst):
        return len(lst)

    metrics = {
        "num_HV": count(hv_data["duration"]),
        "num_AV": count(av_data["duration"]),
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
    return metrics


csv_path = "vanilla_training_metrics.csv"
if not os.path.exists(csv_path):
    pd.DataFrame(
        columns=[
            "episode",
            "episode_reward",
            "epsilon",
            "num_AV",
            "num_HV",
            "avg_dur_AV",
            "avg_dur_HV",
            "avg_wait_AV",
            "avg_wait_HV",
            "avg_loss_AV",
            "avg_loss_HV",
            "avg_len_AV",
            "avg_len_HV",
            "avg_reroute_AV",
        ]
    ).to_csv(csv_path, index=False)


def get_state(vehicle_id, destination_pos):
    try:
        current_edge_id = traci.vehicle.getRoadID(vehicle_id)
        current_lane_id = traci.vehicle.getLaneID(vehicle_id)
    except traci.TraCIException:
        return None

    road_ids = {"current": current_edge_id}

    try:
        links = traci.lane.getLinks(current_lane_id)
        if not links:
            return None

        for link in links:
            to_lane = link[0]
            direction = link[6]
            to_edge = traci.lane.getEdgeID(to_lane)
            if "r" in direction:
                road_ids["right"] = to_edge
            elif "s" in direction:
                road_ids["straight"] = to_edge
            elif "l" in direction:
                road_ids["left"] = to_edge
    except traci.TraCIException:
        return None

    traffic_features = []
    for road_key in ["current", "right", "straight", "left"]:
        edge_id = road_ids.get(road_key)
        av_count = hv_count = 0
        av_speeds, hv_speeds = [], []

        if edge_id:
            try:
                vehicles_on_edge = traci.edge.getLastStepVehicleIDs(edge_id)
                for v_id in vehicles_on_edge:
                    try:
                        v_type = traci.vehicle.getTypeID(v_id)
                        v_speed = traci.vehicle.getSpeed(v_id)
                    except traci.TraCIException:
                        continue
                    if v_type == "AV":
                        av_count += 1
                        av_speeds.append(v_speed)
                    else:
                        hv_count += 1
                        hv_speeds.append(v_speed)
            except traci.TraCIException:
                pass

        avg_av_speed = np.mean(av_speeds) if av_speeds else 0.0
        avg_hv_speed = np.mean(hv_speeds) if hv_speeds else 0.0
        traffic_features.extend([av_count, hv_count, avg_av_speed, avg_hv_speed])

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

    state_vector = np.array(traffic_features + positional_features, dtype=np.float32)

    try:
        # Normalize traffic features
        state_vector[0:16:4] /= 50.0  # AV counts
        state_vector[1:16:4] /= 50.0  # HV counts
        state_vector[2:16:4] /= 15.0  # AV avg speeds
        state_vector[3:16:4] /= 15.0  # HV avg speeds
        # Normalize positional features
        state_vector[16:] /= 200.0
    except Exception:
        return None

    return state_vector


def get_reward(vehicle_id, initial_dist_to_dest, destination_pos):
    omega = 0.6
    try:
        current_speed = traci.vehicle.getSpeed(vehicle_id)
        max_speed = traci.vehicle.getMaxSpeed(vehicle_id)
        ds_n = current_speed / max_speed if max_speed > 0 else 0.0

        current_pos = np.array(traci.vehicle.getPosition(vehicle_id))
        current_dist_to_dest = np.linalg.norm(current_pos - np.array(destination_pos))
        dd_n = (
            current_dist_to_dest / initial_dist_to_dest
            if initial_dist_to_dest > 0
            else 0.0
        )

        reward = (omega * ds_n) - ((1 - omega) * dd_n)
        return reward
    except traci.TraCIException:
        return 0.0
    except Exception:
        return 0.0


def execute_action(vehicle_id, action):
    action_to_direction = {0: "r", 1: "s", 2: "l"}
    chosen_direction = action_to_direction.get(action)
    if chosen_direction is None:
        return

    try:
        current_lane = traci.vehicle.getLaneID(vehicle_id)
        links = traci.lane.getLinks(current_lane)
        for link in links:
            to_lane, direction = link[0], link[6]
            if direction == chosen_direction:
                target_edge = to_lane.split("_")[0]
                traci.vehicle.changeTarget(vehicle_id, target_edge)
                return
    except traci.TraCIException:
        pass
    except Exception:
        pass


def initialize_lane_map():
    global all_lane_ids, edge_to_lanes_map
    try:
        all_lane_ids = traci.lane.getIDList()
    except traci.TraCIException:
        all_lane_ids = []

    edge_to_lanes_map.clear()
    for lane_id in all_lane_ids:
        try:
            edge_id = traci.lane.getEdgeID(lane_id)
            if not edge_id.startswith(":"):
                edge_to_lanes_map[edge_id].append(lane_id)
        except traci.TraCIException:
            continue


def get_lane_from_edge(dest_edge):
    if dest_edge in edge_to_lanes_map:
        lanes = edge_to_lanes_map[dest_edge]
        chosen = lanes[0] if lanes else None
        return chosen
    return None


def train():
    agent = DQNAgent(
        state_size=STATE_SIZE, action_size=ACTION_SIZE, total_episodes=NUM_EPISODES
    )
    checkpoint_dir = "checkpoints/vanilla_dqn"
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        ckpts = [
            f
            for f in os.listdir(checkpoint_dir)
            if f.startswith("checkpoint_ep") and f.endswith(".pth")
        ]
        if ckpts:
            ckpts.sort(
                key=lambda x: int(x.replace("checkpoint_ep", "").replace(".pth", ""))
            )
            latest_checkpoint = os.path.join(checkpoint_dir, ckpts[-1])

    start_episode = 1
    if latest_checkpoint:
        if agent.load(latest_checkpoint):
            start_episode = (
                int(latest_checkpoint.split("checkpoint_ep")[-1].split(".pth")[0]) + 1
            )
    scores = []

    for episode in range(start_episode, NUM_EPISODES + 1):
        seed = random.randint(1, 2_000_000_000)
        episode_sumo_cmd = list(SUMO_CMD) + [
            "--seed",
            str(seed),
            "--time-to-teleport",
            "600",
        ]
        try:
            traci.start(episode_sumo_cmd)
        except Exception as e:
            print(f"Error starting SUMO: {e}", file=sys.stderr)
            break

        initialize_lane_map()

        step, episode_reward = 0, 0.0
        vehicle_dests, vehicle_initial_dists = {}, {}
        last_decision_data = defaultdict(lambda: None)
        vehicle_episode_rewards = defaultdict(float)

        try:
            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                step += 1

                if step > MAX_STEPS:
                    print("Safety stop: maximum steps reached.")
                    break

                try:
                    departed = traci.simulation.getDepartedIDList()
                except traci.TraCIException:
                    departed = []

                for v_id in departed:
                    try:
                        if traci.vehicle.getTypeID(v_id) == "AV":
                            route = traci.vehicle.getRoute(v_id)
                            if not route:
                                continue
                            dest_edge = route[-1]
                            lane_id = get_lane_from_edge(dest_edge)
                            if lane_id is None:
                                continue
                            dest_pos = traci.lane.getShape(lane_id)[-1]
                            vehicle_dests[v_id] = dest_pos
                            pos = traci.vehicle.getPosition(v_id)
                            vehicle_initial_dists[v_id] = np.linalg.norm(
                                np.array(pos) - np.array(dest_pos)
                            )
                    except (traci.TraCIException, IndexError):
                        continue

                try:
                    all_vehicle_ids = traci.vehicle.getIDList()
                except traci.TraCIException:
                    all_vehicle_ids = []
                active_vehicle_ids = [v for v in all_vehicle_ids if v in vehicle_dests]

                for v_id in active_vehicle_ids:
                    try:
                        current_lane = traci.vehicle.getLaneID(v_id)
                        if current_lane.startswith(":"):
                            continue

                        lane_length = traci.lane.getLength(current_lane)
                        pos_on_lane = traci.vehicle.getLanePosition(v_id)
                        distance_to_end = lane_length - pos_on_lane

                        if distance_to_end <= DECISION_ZONE_LENGTH:
                            current_state = get_state(v_id, vehicle_dests[v_id])
                            if current_state is None:
                                continue

                            reward = get_reward(
                                v_id, vehicle_initial_dists[v_id], vehicle_dests[v_id]
                            )
                            vehicle_episode_rewards[v_id] += reward

                            if last_decision_data[v_id] is not None:
                                prev_state, prev_action = last_decision_data[v_id]
                                agent.step(
                                    prev_state,
                                    prev_action,
                                    reward,
                                    current_state,
                                    False,
                                )

                            action = agent.act(current_state, eps=agent.epsilon)
                            execute_action(v_id, action)
                            last_decision_data[v_id] = (current_state, action)
                        else:
                            pass
                    except traci.TraCIException:
                        continue

                try:
                    arrived = traci.simulation.getArrivedIDList()
                except traci.TraCIException:
                    arrived = []

                for v_id in arrived:
                    if v_id in last_decision_data:
                        prev_state, prev_action = last_decision_data[v_id]
                        final_reward = 10.0
                        vehicle_episode_rewards[v_id] += final_reward
                        agent.step(
                            prev_state, prev_action, final_reward, prev_state, True
                        )
                        del last_decision_data[v_id]

                    vehicle_dests.pop(v_id, None)
                    vehicle_initial_dists.pop(v_id, None)

                try:
                    min_expected = traci.simulation.getMinExpectedNumber()
                except traci.TraCIException:
                    min_expected = None

                if min_expected == 0:
                    break

        except Exception as e:
            print(f"An error occurred during simulation: {e}", file=sys.stderr)
        finally:
            try:
                agent.update_epsilon(episode)
            except Exception as e:
                print(f"Error updating epsilon: {e}", file=sys.stderr)

            try:
                traci.close()
            except Exception as e:
                print(f"Error closing TRACI: {e}", file=sys.stderr)

        episode_reward = sum(vehicle_episode_rewards.values())
        scores.append(episode_reward)
        episode_metrics = parse_tripinfo_xml("tripinfo.xml")
        if episode_metrics:
            episode_metrics["episode"] = episode
            episode_metrics["episode_reward"] = episode_reward
            episode_metrics["epsilon"] = agent.epsilon

            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([episode_metrics])], ignore_index=True)
            df.to_csv(csv_path, index=False)

        if episode % max(1, NUM_EPISODES // 5) == 0 or episode == NUM_EPISODES:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_ep{episode}.pth"
            )
            try:
                agent.save(checkpoint_path)
            except Exception:
                pass

    df = pd.read_csv(csv_path)
    episodes = df["episode"]

    def smooth(series, window=5):
        return series.rolling(window, min_periods=1).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, smooth(df["episode_reward"]), label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Progress")
    plt.grid(True)
    plt.legend()
    plt.savefig("vanilla_reward_progress.png", bbox_inches="tight")

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, smooth(df["avg_dur_AV"]), label="AV Duration (s)")
    plt.plot(episodes, smooth(df["avg_dur_HV"]), label="HV Duration (s)")
    plt.xlabel("Episode")
    plt.ylabel("Avg Trip Duration (s)")
    plt.title("Average Trip Duration per Episode")
    plt.legend()
    plt.grid(True)
    plt.savefig("vanilla_duration_progress.png", bbox_inches="tight")

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, smooth(df["avg_wait_AV"]), label="AV Waiting (s)")
    plt.plot(episodes, smooth(df["avg_wait_HV"]), label="HV Waiting (s)")
    plt.xlabel("Episode")
    plt.ylabel("Average Waiting Time (s)")
    plt.title("Average Waiting Time per Episode")
    plt.legend()
    plt.grid(True)
    plt.savefig("vanilla_waiting_progress.png", bbox_inches="tight")

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, smooth(df["avg_loss_AV"]), label="AV Time Loss (s)")
    plt.plot(episodes, smooth(df["avg_loss_HV"]), label="HV Time Loss (s)")
    plt.xlabel("Episode")
    plt.ylabel("Average Time Loss (s)")
    plt.title("Average Time Loss per Episode")
    plt.legend()
    plt.grid(True)
    plt.savefig("vanilla_timeloss_progress.png", bbox_inches="tight")

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, smooth(df["avg_reroute_AV"]), label="AV Reroutes")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reroute Count")
    plt.title("Average AV Reroutes per Episode")
    plt.legend()
    plt.grid(True)
    plt.savefig("vanilla_reroutes_progress.png", bbox_inches="tight")

    return scores


if __name__ == "__main__":
    scores = train()
