import os
import sys
import random
from collections import defaultdict
import numpy as np

# --- START OF ENVIRONMENT FIX ---
# Use Homebrew paths directly to avoid conflicts
HOMEBREW_SUMO_ROOT = "/opt/homebrew/opt/sumo"
SUMO_TOOLS_PATH = os.path.join(HOMEBREW_SUMO_ROOT, "share", "sumo", "tools")
sys.path.append(SUMO_TOOLS_PATH)

import traci
from dqn_agent import DQNAgent
# Use the exact binary from Homebrew
SUMO_BINARY = os.path.join(HOMEBREW_SUMO_ROOT, "bin", "sumo")
# --- END OF ENVIRONMENT FIX ---

SUMO_CFG = "simulation.sumocfg"
SUMO_CMD = [
    SUMO_BINARY,
    "-c", SUMO_CFG,
    "--tripinfo-output", "tripinfo.xml"
]
# scaling option (you said you used this)
SUMO_CMD += ["--scale", "0.5"]

STATE_SIZE = 20
ACTION_SIZE = 3  # Right, Straight, Left
NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 5400
DECISION_ZONE_LENGTH = 50

# --- Helper Functions ---
def get_state(vehicle_id, destination_pos):
    """
    Constructs the full 20-element state vector for a given AV agent.
    """
    road_ids = {}
    road_ids['current'] = traci.vehicle.getRoadID(vehicle_id)

    try:
        current_lane = traci.vehicle.getLaneID(vehicle_id)
        links = traci.lane.getLinks(current_lane)

        if not links:
            return None

        # link tuple indexing adjusted for SUMO 1.20.0
        for link in links:
            to_edge = link[7]          # target edge ID
            direction = link[4]        # 'r', 's', 'l'

            if isinstance(direction, str):
                if 'r' in direction:
                    road_ids['right'] = to_edge
                elif 's' in direction:
                    road_ids['straight'] = to_edge
                elif 'l' in direction:
                    road_ids['left'] = to_edge

    except traci.TraCIException:
        return None

    traffic_features = []
    road_order = ['current', 'right', 'straight', 'left']

    for road_key in road_order:
        edge_id = road_ids.get(road_key)

        av_count, hv_count = 0, 0
        av_speeds, hv_speeds = [], []

        if edge_id:
            try:
                vehicles_on_edge = traci.edge.getLastStepVehicleIDs(edge_id)
                active_vehicles = traci.vehicle.getIDList()
                for v_id in vehicles_on_edge:
                    if v_id in active_vehicles:
                        if traci.vehicle.getTypeID(v_id) == "AV":
                            av_count += 1
                            av_speeds.append(traci.vehicle.getSpeed(v_id))
                        else:
                            hv_count += 1
                            hv_speeds.append(traci.vehicle.getSpeed(v_id))
            except traci.TraCIException:
                pass

        avg_av_speed = np.mean(av_speeds) if av_speeds else 0.0
        avg_hv_speed = np.mean(hv_speeds) if hv_speeds else 0.0

        traffic_features.extend([av_count, hv_count, avg_av_speed, avg_hv_speed])

    try:
        current_pos = traci.vehicle.getPosition(vehicle_id)
        positional_features = [current_pos[0], current_pos[1], destination_pos[0], destination_pos[1]]
    except traci.TraCIException:
        return None

    state_vector = np.array(traffic_features + positional_features, dtype=np.float32)

    # --- Normalization ---
    state_vector[0:16:4] /= 50.0  # AV Counts
    state_vector[1:16:4] /= 50.0  # HV Counts
    state_vector[2:16:4] /= 15.0  # AV Speeds
    state_vector[3:16:4] /= 15.0  # HV Speeds
    state_vector[16] /= 200.0  # current x
    state_vector[17] /= 200.0  # current y
    state_vector[18] /= 200.0  # dest x
    state_vector[19] /= 200.0  # dest y

    return state_vector

def get_reward(vehicle_id, initial_dist_to_dest, destination_pos):
    """
    Calculates the reward based on the paper's multi-objective function.
    """
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
        return 0

def execute_action(vehicle_id, action):
    """
    Translates a discrete action (0: right, 1: straight, 2: left) into a
    SUMO command to change the vehicle's target edge.
    """
    action_to_direction = {0: 'r', 1: 's', 2: 'l'}
    chosen_direction = action_to_direction.get(action)

    if chosen_direction is None:
        return

    try:
        current_lane = traci.vehicle.getLaneID(vehicle_id)
        links = traci.lane.getLinks(current_lane)

        if not links:
            return

        for link in links:
            target_edge = link[7]
            direction = link[4]

            if isinstance(direction, str) and chosen_direction in direction:
                traci.vehicle.changeTarget(vehicle_id, target_edge)
                return

    except traci.TraCIException:
        pass

# Caching all lane IDs at the start of an episode
all_lane_ids = []
# Create a mapping from edgeID to list of laneIDs
edge_to_lanes_map = defaultdict(list)

def initialize_lane_map():
    """
    Builds a dictionary that maps edge IDs to their lane IDs.
    """
    global all_lane_ids, edge_to_lanes_map
    all_lane_ids = traci.lane.getIDList()
    edge_to_lanes_map.clear()
    for lane_id in all_lane_ids:
        try:
            edge_id = traci.lane.getEdgeID(lane_id)
            if not edge_id.startswith(':'):  # Ignore internal junction lanes
                edge_to_lanes_map[edge_id].append(lane_id)
        except traci.TraCIException:
            pass  # Skip internal lanes, etc.

def get_lane_from_edge(dest_edge):
    """
    Safely gets the first lane ID for a given edge ID using our cache.
    """
    if dest_edge in edge_to_lanes_map and edge_to_lanes_map[dest_edge]:
        return edge_to_lanes_map[dest_edge][0]
    return None

def train():
    agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    scores = []

    for episode in range(1, NUM_EPISODES + 1):
        # per-episode SUMO command (random seed + higher teleport during debugging)
        seed = random.randint(1, 2_000_000_000)
        episode_sumo_cmd = list(SUMO_CMD)  # copy base
        episode_sumo_cmd += ["--seed", str(seed), "--time-to-teleport", "600"]

        traci.start(episode_sumo_cmd)
        initialize_lane_map()

        step = 0
        episode_reward = 0

        vehicle_dests = {}
        vehicle_initial_dists = {}
        last_decision_data = defaultdict(lambda: None)
        vehicle_episode_rewards = defaultdict(float)

        # choose a single sample vehicle id for light tracing per episode
        sample_trace_vehicle = None
        sample_trace_count = 0

        while step < MAX_STEPS_PER_EPISODE:
            traci.simulationStep()

            # --- Handle newly departed vehicles ---
            departed_ids = traci.simulation.getDepartedIDList()
            for v_id in departed_ids:
                try:
                    if traci.vehicle.getTypeID(v_id) == "AV":
                        dest_edge = traci.vehicle.getRoute(v_id)[-1]

                        lane_id = get_lane_from_edge(dest_edge)
                        if lane_id is None:
                            continue

                        lane_shape = traci.lane.getShape(lane_id)
                        dest_pos = lane_shape[-1]
                        vehicle_dests[v_id] = dest_pos

                        pos = traci.vehicle.getPosition(v_id)
                        initial_dist = np.linalg.norm(np.array(pos) - np.array(dest_pos))
                        vehicle_initial_dists[v_id] = initial_dist
                except (traci.TraCIException, IndexError):
                    pass

            active_vehicle_ids = traci.vehicle.getIDList()
            active_av_ids = [v_id for v_id in active_vehicle_ids if v_id in vehicle_dests]

            for v_id in active_av_ids:
                try:
                    current_lane = traci.vehicle.getLaneID(v_id)
                    if current_lane.startswith(':'):
                        continue

                    lane_length = traci.lane.getLength(current_lane)
                    pos_on_lane = traci.vehicle.getLanePosition(v_id)
                    dist_to_intersection = lane_length - pos_on_lane

                    if dist_to_intersection <= DECISION_ZONE_LENGTH:
                        current_state = get_state(v_id, vehicle_dests[v_id])
                        if current_state is None:
                            continue

                        reward = get_reward(v_id, vehicle_initial_dists[v_id], vehicle_dests[v_id])
                        vehicle_episode_rewards[v_id] += reward

                        if last_decision_data[v_id] is not None:
                            prev_state, prev_action = last_decision_data[v_id]
                            done = False

                            agent.step(prev_state, prev_action, reward, current_state, done)
                        # ensure act uses current epsilon
                        action = agent.act(current_state, eps=agent.epsilon)
                        execute_action(v_id, action)

                        last_decision_data[v_id] = (current_state, action)

                        # record a sample trajectory for a single vehicle (very light)
                        if sample_trace_vehicle is None:
                            sample_trace_vehicle = v_id
                        if v_id == sample_trace_vehicle and sample_trace_count < 10:
                            pos = traci.vehicle.getPosition(v_id)
                            sample_trace_count += 1

                except traci.TraCIException:
                    if v_id in last_decision_data:
                        del last_decision_data[v_id]
                    if v_id in vehicle_dests:
                        del vehicle_dests[v_id]

            arrived_ids = traci.simulation.getArrivedIDList()
            for v_id in arrived_ids:
                if v_id in last_decision_data:
                    prev_state, prev_action = last_decision_data[v_id]

                    final_reward = 10.0
                    vehicle_episode_rewards[v_id] += final_reward

                    agent.step(prev_state, prev_action, final_reward, prev_state, True)

                    del last_decision_data[v_id]
                    if v_id in vehicle_dests:
                        del vehicle_dests[v_id]
                    if v_id in vehicle_initial_dists:
                        del vehicle_initial_dists[v_id]

            step += 1
            if traci.simulation.getMinExpectedNumber() == 0:
                break

        # close SUMO for this episode
        traci.close()

        episode_reward = sum(vehicle_episode_rewards.values())
        scores.append(episode_reward)

        print(f"Episode: {episode}/{NUM_EPISODES}, Total Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    # optionally: return scores for plotting
    return scores

if __name__ == "__main__":
    train()
