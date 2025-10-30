import os
import sys
import random
import time
from collections import defaultdict
import numpy as np
import traci
from torch.utils.tensorboard import SummaryWriter
from dqn_agent import DQNAgent

# === SUMO CONFIGURATION ===
HOMEBREW_SUMO_ROOT = "/opt/homebrew/opt/sumo"
SUMO_TOOLS_PATH = os.path.join(HOMEBREW_SUMO_ROOT, "share", "sumo", "tools")
sys.path.append(SUMO_TOOLS_PATH)

SUMO_BINARY = os.path.join(HOMEBREW_SUMO_ROOT, "bin", "sumo")
SUMO_CFG = "simulation.sumocfg"
SUMO_CMD = [SUMO_BINARY, "-c", SUMO_CFG, "--tripinfo-output", "tripinfo.xml"]

# === TRAINING PARAMETERS ===
STATE_SIZE = 20
ACTION_SIZE = 3  # Right, Straight, Left
NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 2000
DECISION_ZONE_LENGTH = 50

writer = SummaryWriter(log_dir=f"runs/sumo_dqn_{int(time.time())}")

# === GLOBAL LANE MAP ===
all_lane_ids = []
edge_to_lanes_map = defaultdict(list)


# =====================================================================
#  STATE CONSTRUCTION
# =====================================================================
def get_state(vehicle_id, destination_pos):
    """Constructs a 20-element state vector for the given AV."""
    print(f"\n[INFO] ======== STATE FETCH for {vehicle_id} ========")
    try:
        current_edge_id = traci.vehicle.getRoadID(vehicle_id)
        current_lane_id = traci.vehicle.getLaneID(vehicle_id)
    except traci.TraCIException as e:
        print(f"[ERROR] Failed to fetch vehicle info: {e}")
        return None

    road_ids = {'current': current_edge_id}

    # --- Get connected edges ---
    try:
        links = traci.lane.getLinks(current_lane_id)
        if not links:
            print("[WARNING] No outgoing links found.")
            return None

        for link in links:
            to_lane = link[0]
            direction = link[6]
            to_edge = traci.lane.getEdgeID(to_lane)
            if 'r' in direction:
                road_ids['right'] = to_edge
            elif 's' in direction:
                road_ids['straight'] = to_edge
            elif 'l' in direction:
                road_ids['left'] = to_edge
    except traci.TraCIException as e:
        print(f"[ERROR] Reading lane links failed: {e}")
        return None

    traffic_features = []
    for road_key in ['current', 'right', 'straight', 'left']:
        edge_id = road_ids.get(road_key)
        av_count = hv_count = 0
        av_speeds, hv_speeds = [], []

        if edge_id:
            try:
                vehicles_on_edge = traci.edge.getLastStepVehicleIDs(edge_id)
                for v_id in vehicles_on_edge:
                    v_type = traci.vehicle.getTypeID(v_id)
                    v_speed = traci.vehicle.getSpeed(v_id)
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
        positional_features = [current_pos[0], current_pos[1], destination_pos[0], destination_pos[1]]
    except traci.TraCIException:
        print("[ERROR] Failed to get position.")
        return None

    state_vector = np.array(traffic_features + positional_features, dtype=np.float32)

    # Normalization
    state_vector[0:16:4] /= 50.0
    state_vector[1:16:4] /= 50.0
    state_vector[2:16:4] /= 15.0
    state_vector[3:16:4] /= 15.0
    state_vector[16:] /= 200.0

    print(f"[DEBUG] State vector ready for {vehicle_id}")
    return state_vector


# =====================================================================
#  REWARD FUNCTION
# =====================================================================
def get_reward(vehicle_id, initial_dist_to_dest, destination_pos):
    omega = 0.6
    try:
        current_speed = traci.vehicle.getSpeed(vehicle_id)
        max_speed = traci.vehicle.getMaxSpeed(vehicle_id)
        ds_n = current_speed / max_speed if max_speed > 0 else 0.0

        current_pos = np.array(traci.vehicle.getPosition(vehicle_id))
        current_dist_to_dest = np.linalg.norm(current_pos - np.array(destination_pos))
        dd_n = current_dist_to_dest / initial_dist_to_dest if initial_dist_to_dest > 0 else 0.0

        reward = (omega * ds_n) - ((1 - omega) * dd_n)
        return reward
    except traci.TraCIException:
        return 0.0


# =====================================================================
#  ACTION EXECUTION
# =====================================================================
def execute_action(vehicle_id, action):
    """Executes (0: right, 1: straight, 2: left)."""
    action_to_direction = {0: 'r', 1: 's', 2: 'l'}
    chosen_direction = action_to_direction.get(action)
    if chosen_direction is None:
        print(f"[WARNING] Invalid action {action} for {vehicle_id}")
        return

    try:
        current_lane = traci.vehicle.getLaneID(vehicle_id)
        links = traci.lane.getLinks(current_lane)
        for link in links:
            to_lane, direction = link[0], link[6]
            if direction == chosen_direction:
                target_edge = to_lane.split('_')[0]
                traci.vehicle.changeTarget(vehicle_id, target_edge)
                print(f"[INFO] Vehicle {vehicle_id} → {chosen_direction.upper()} to {target_edge}")
                return
    except traci.TraCIException:
        print(f"[ERROR] Failed to execute action for {vehicle_id}")


# =====================================================================
#  LANE MAP INITIALIZATION
# =====================================================================
def initialize_lane_map():
    global all_lane_ids, edge_to_lanes_map
    print("\n[INFO] ======== INITIALIZING LANE MAP ========")

    all_lane_ids = traci.lane.getIDList()
    edge_to_lanes_map.clear()
    for lane_id in all_lane_ids:
        try:
            edge_id = traci.lane.getEdgeID(lane_id)
            if not edge_id.startswith(':'):
                edge_to_lanes_map[edge_id].append(lane_id)
        except traci.TraCIException:
            pass

    print(f"[INFO] Lane map built with {len(edge_to_lanes_map)} edges.")


def get_lane_from_edge(dest_edge):
    if dest_edge in edge_to_lanes_map:
        lanes = edge_to_lanes_map[dest_edge]
        return lanes[0] if lanes else None
    return None


# =====================================================================
#  TRAINING LOOP
# =====================================================================
def train():
    agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    scores = []

    for episode in range(1, NUM_EPISODES + 1):
        print(f"\n\n================== EPISODE {episode} ==================")

        seed = random.randint(1, 2_000_000_000)
        episode_sumo_cmd = list(SUMO_CMD) + ["--seed", str(seed), "--time-to-teleport", "600"]
        traci.start(episode_sumo_cmd)
        initialize_lane_map()

        step, episode_reward = 0, 0
        vehicle_dests, vehicle_initial_dists = {}, {}
        last_decision_data = defaultdict(lambda: None)
        vehicle_episode_rewards = defaultdict(float)

        while step < MAX_STEPS_PER_EPISODE:
            traci.simulationStep()
            step += 1

            # Handle newly departed vehicles
            for v_id in traci.simulation.getDepartedIDList():
                try:
                    if traci.vehicle.getTypeID(v_id) == "AV":
                        dest_edge = traci.vehicle.getRoute(v_id)[-1]
                        lane_id = get_lane_from_edge(dest_edge)
                        if lane_id is None:
                            continue
                        dest_pos = traci.lane.getShape(lane_id)[-1]
                        vehicle_dests[v_id] = dest_pos
                        pos = traci.vehicle.getPosition(v_id)
                        vehicle_initial_dists[v_id] = np.linalg.norm(np.array(pos) - np.array(dest_pos))
                except (traci.TraCIException, IndexError):
                    continue

            active_vehicle_ids = [v for v in traci.vehicle.getIDList() if v in vehicle_dests]

            for v_id in active_vehicle_ids:
                try:
                    current_lane = traci.vehicle.getLaneID(v_id)
                    if current_lane.startswith(':'):
                        continue

                    lane_length = traci.lane.getLength(current_lane)
                    pos_on_lane = traci.vehicle.getLanePosition(v_id)
                    if lane_length - pos_on_lane <= DECISION_ZONE_LENGTH:
                        current_state = get_state(v_id, vehicle_dests[v_id])
                        if current_state is None:
                            continue

                        reward = get_reward(v_id, vehicle_initial_dists[v_id], vehicle_dests[v_id])
                        vehicle_episode_rewards[v_id] += reward

                        if last_decision_data[v_id] is not None:
                            prev_state, prev_action = last_decision_data[v_id]
                            agent.step(prev_state, prev_action, reward, current_state, False)

                        action = agent.act(current_state, eps=agent.epsilon)
                        execute_action(v_id, action)
                        last_decision_data[v_id] = (current_state, action)
                except traci.TraCIException:
                    continue

            # Handle arrivals
            for v_id in traci.simulation.getArrivedIDList():
                if v_id in last_decision_data:
                    prev_state, prev_action = last_decision_data[v_id]
                    final_reward = 10.0
                    vehicle_episode_rewards[v_id] += final_reward
                    agent.step(prev_state, prev_action, final_reward, prev_state, True)
                    del last_decision_data[v_id]

                vehicle_dests.pop(v_id, None)
                vehicle_initial_dists.pop(v_id, None)

            if traci.simulation.getMinExpectedNumber() == 0:
                print(f"[INFO] All vehicles arrived — ending early at step {step}")
                break

        agent.update_epsilon(episode)
        traci.close()

        episode_reward = sum(vehicle_episode_rewards.values())
        scores.append(episode_reward)

        print(f"\n[RESULT] ✅ Episode {episode} completed")
        print(f"[RESULT] Total Reward: {episode_reward:.2f}")
        print(f"[RESULT] Epsilon: {agent.epsilon:.3f}")
        print("=" * 70)

    return scores


if __name__ == "__main__":
    train()
