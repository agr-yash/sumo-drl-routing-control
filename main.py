import os
import sys
import random
import time
from collections import defaultdict
import numpy as np
import traci
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

# === GLOBAL LANE MAP ===
all_lane_ids = []
edge_to_lanes_map = defaultdict(list)


# =====================================================================
#  STATE CONSTRUCTION
# =====================================================================
def get_state(vehicle_id, destination_pos):
    print(f"\n[DEBUG] Enter get_state() for vehicle: {vehicle_id}")
    try:
        current_edge_id = traci.vehicle.getRoadID(vehicle_id)
        current_lane_id = traci.vehicle.getLaneID(vehicle_id)
        print(f"[DEBUG] current_edge_id={current_edge_id}, current_lane_id={current_lane_id}")
    except traci.TraCIException as e:
        print(f"[ERROR] Failed to fetch vehicle info: {e}")
        return None

    road_ids = {'current': current_edge_id}

    # --- Get connected edges ---
    try:
        links = traci.lane.getLinks(current_lane_id)
        print(f"[DEBUG] Found {len(links)} outgoing links from lane {current_lane_id}")
        if not links:
            print("[WARNING] No outgoing links found.")
            return None

        for link in links:
            to_lane = link[0]
            direction = link[6]
            to_edge = traci.lane.getEdgeID(to_lane)
            print(f"[DEBUG] Link to_lane={to_lane}, direction={direction}, to_edge={to_edge}")
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
                print(f"[DEBUG] {road_key} edge {edge_id} has {len(vehicles_on_edge)} vehicles")
                for v_id in vehicles_on_edge:
                    try:
                        v_type = traci.vehicle.getTypeID(v_id)
                        v_speed = traci.vehicle.getSpeed(v_id)
                    except traci.TraCIException:
                        # If individual vehicle read fails, skip it but warn
                        print(f"[WARNING] Could not read vehicle {v_id} on edge {edge_id}")
                        continue
                    if v_type == "AV":
                        av_count += 1
                        av_speeds.append(v_speed)
                    else:
                        hv_count += 1
                        hv_speeds.append(v_speed)
            except traci.TraCIException:
                print(f"[WARNING] Could not get vehicles for edge {edge_id}")

        avg_av_speed = np.mean(av_speeds) if av_speeds else 0.0
        avg_hv_speed = np.mean(hv_speeds) if hv_speeds else 0.0
        print(f"[DEBUG] {road_key} stats: AVs={av_count}, HVs={hv_count}, AV_avg={avg_av_speed:.2f}, HV_avg={avg_hv_speed:.2f}")
        traffic_features.extend([av_count, hv_count, avg_av_speed, avg_hv_speed])

    try:
        current_pos = traci.vehicle.getPosition(vehicle_id)
        positional_features = [current_pos[0], current_pos[1], destination_pos[0], destination_pos[1]]
        print(f"[DEBUG] current_pos={current_pos}, destination_pos={destination_pos}")
    except traci.TraCIException:
        print("[ERROR] Failed to get position.")
        return None

    state_vector = np.array(traffic_features + positional_features, dtype=np.float32)
    print(f"[DEBUG] Raw state vector: {state_vector}")

    # Normalization
    try:
        state_vector[0:16:4] /= 50.0
        state_vector[1:16:4] /= 50.0
        state_vector[2:16:4] /= 15.0
        state_vector[3:16:4] /= 15.0
        state_vector[16:] /= 200.0
    except Exception as e:
        print(f"[ERROR] Normalization failed: {e}")
        return None

    print(f"[DEBUG] Normalized state vector for {vehicle_id}: {state_vector}")
    print(f"[DEBUG] Exit get_state() for {vehicle_id}")
    return state_vector


# =====================================================================
#  REWARD FUNCTION
# =====================================================================
def get_reward(vehicle_id, initial_dist_to_dest, destination_pos):
    print(f"\n[DEBUG] Enter get_reward() for vehicle: {vehicle_id}")
    omega = 0.6
    try:
        current_speed = traci.vehicle.getSpeed(vehicle_id)
        max_speed = traci.vehicle.getMaxSpeed(vehicle_id)
        ds_n = current_speed / max_speed if max_speed > 0 else 0.0

        current_pos = np.array(traci.vehicle.getPosition(vehicle_id))
        current_dist_to_dest = np.linalg.norm(current_pos - np.array(destination_pos))
        dd_n = current_dist_to_dest / initial_dist_to_dest if initial_dist_to_dest > 0 else 0.0

        reward = (omega * ds_n) - ((1 - omega) * dd_n)
        print(f"[DEBUG] speed={current_speed:.2f}, max_speed={max_speed:.2f}, ds_n={ds_n:.3f}, dd_n={dd_n:.3f}, reward={reward:.3f}")
        print(f"[DEBUG] Exit get_reward() for {vehicle_id}")
        return reward
    except traci.TraCIException:
        print(f"[ERROR] TraCIException in get_reward() for {vehicle_id}")
        return 0.0
    except Exception as e:
        print(f"[ERROR] Unexpected error in get_reward(): {e}")
        return 0.0


# =====================================================================
#  ACTION EXECUTION
# =====================================================================
def execute_action(vehicle_id, action):
    print(f"\n[DEBUG] Enter execute_action() for {vehicle_id}, action={action}")
    action_to_direction = {0: 'r', 1: 's', 2: 'l'}
    chosen_direction = action_to_direction.get(action)
    if chosen_direction is None:
        print(f"[WARNING] Invalid action {action} for {vehicle_id}")
        return

    try:
        current_lane = traci.vehicle.getLaneID(vehicle_id)
        links = traci.lane.getLinks(current_lane)
        print(f"[DEBUG] Found {len(links)} links from {current_lane}")
        for link in links:
            to_lane, direction = link[0], link[6]
            print(f"[DEBUG] Checking link to_lane={to_lane}, direction={direction}")
            if direction == chosen_direction:
                target_edge = to_lane.split('_')[0]
                traci.vehicle.changeTarget(vehicle_id, target_edge)
                print(f"[INFO] Vehicle {vehicle_id} → {chosen_direction.upper()} to {target_edge}")
                print(f"[DEBUG] Exit execute_action() for {vehicle_id}")
                return
        print(f"[WARNING] No link found for desired direction {chosen_direction} from lane {current_lane}")
    except traci.TraCIException as e:
        print(f"[ERROR] Failed to execute action for {vehicle_id}: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error in execute_action(): {e}")


# =====================================================================
#  LANE MAP INITIALIZATION
# =====================================================================
def initialize_lane_map():
    global all_lane_ids, edge_to_lanes_map
    print("\n[INFO] ======== INITIALIZING LANE MAP ========")
    try:
        all_lane_ids = traci.lane.getIDList()
        print(f"[DEBUG] Retrieved {len(all_lane_ids)} lanes from SUMO")
    except traci.TraCIException as e:
        print(f"[ERROR] Failed to get lane ID list: {e}")
        all_lane_ids = []

    edge_to_lanes_map.clear()
    for lane_id in all_lane_ids:
        try:
            edge_id = traci.lane.getEdgeID(lane_id)
            if not edge_id.startswith(':'):
                edge_to_lanes_map[edge_id].append(lane_id)
        except traci.TraCIException:
            print(f"[WARNING] Could not resolve edge for lane {lane_id}")
            continue

    print(f"[INFO] Lane map built with {len(edge_to_lanes_map)} edges.")


def get_lane_from_edge(dest_edge):
    print(f"[DEBUG] get_lane_from_edge called for edge: {dest_edge}")
    if dest_edge in edge_to_lanes_map:
        lanes = edge_to_lanes_map[dest_edge]
        chosen = lanes[0] if lanes else None
        print(f"[DEBUG] Returning lane {chosen} for edge {dest_edge}")
        return chosen
    print(f"[WARNING] Edge {dest_edge} not found in lane map")
    return None


# =====================================================================
#  TRAINING LOOP
# =====================================================================
def train():
    print("[INFO] Starting training")
    agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    print(f"[DEBUG] Agent initialized with state_size={STATE_SIZE}, action_size={ACTION_SIZE}")
    scores = []

    for episode in range(1, NUM_EPISODES + 1):
        print(f"\n\n================== EPISODE {episode} ==================")
        seed = random.randint(1, 2_000_000_000)
        episode_sumo_cmd = list(SUMO_CMD) + ["--seed", str(seed), "--time-to-teleport", "600"]
        print(f"[DEBUG] Starting SUMO with seed {seed}")
        try:
            traci.start(episode_sumo_cmd)
        except Exception as e:
            print(f"[ERROR] traci.start failed: {e}")
            break

        initialize_lane_map()

        step, episode_reward = 0, 0.0
        vehicle_dests, vehicle_initial_dists = {}, {}
        last_decision_data = defaultdict(lambda: None)
        vehicle_episode_rewards = defaultdict(float)

        try:
            while step < MAX_STEPS_PER_EPISODE:
                traci.simulationStep()
                step += 1

                if step % 100 == 0:
                    print(f"[DEBUG] Episode {episode} step {step}")

                # Handle newly departed vehicles
                try:
                    departed = traci.simulation.getDepartedIDList()
                except traci.TraCIException:
                    departed = []
                if departed:
                    print(f"[DEBUG] Departed vehicles at step {step}: {departed}")

                for v_id in departed:
                    try:
                        if traci.vehicle.getTypeID(v_id) == "AV":
                            route = traci.vehicle.getRoute(v_id)
                            if not route:
                                print(f"[WARNING] Vehicle {v_id} has empty route")
                                continue
                            dest_edge = route[-1]
                            lane_id = get_lane_from_edge(dest_edge)
                            if lane_id is None:
                                print(f"[WARNING] No lane found for dest edge {dest_edge} for vehicle {v_id}")
                                continue
                            dest_pos = traci.lane.getShape(lane_id)[-1]
                            vehicle_dests[v_id] = dest_pos
                            pos = traci.vehicle.getPosition(v_id)
                            vehicle_initial_dists[v_id] = np.linalg.norm(np.array(pos) - np.array(dest_pos))
                            print(f"[INFO] AV {v_id} departed: dest_edge={dest_edge}, lane_id={lane_id}, dest_pos={dest_pos}, initial_dist={vehicle_initial_dists[v_id]:.2f}")
                    except (traci.TraCIException, IndexError) as e:
                        print(f"[WARNING] Error handling departed vehicle {v_id}: {e}")
                        continue

                # Active AVs with known destinations
                try:
                    all_vehicle_ids = traci.vehicle.getIDList()
                except traci.TraCIException:
                    all_vehicle_ids = []
                active_vehicle_ids = [v for v in all_vehicle_ids if v in vehicle_dests]
                if active_vehicle_ids:
                    print(f"[DEBUG] Active AVs at step {step}: {active_vehicle_ids}")

                for v_id in active_vehicle_ids:
                    try:
                        current_lane = traci.vehicle.getLaneID(v_id)
                        if current_lane.startswith(':'):
                            print(f"[DEBUG] Skipping vehicle {v_id} on internal lane {current_lane}")
                            continue

                        lane_length = traci.lane.getLength(current_lane)
                        pos_on_lane = traci.vehicle.getLanePosition(v_id)
                        distance_to_end = lane_length - pos_on_lane
                        # Check decision zone
                        if distance_to_end <= DECISION_ZONE_LENGTH:
                            print(f"[DEBUG] Vehicle {v_id} within decision zone (dist_to_end={distance_to_end:.2f})")
                            current_state = get_state(v_id, vehicle_dests[v_id])
                            if current_state is None:
                                print(f"[WARNING] state for {v_id} is None, skipping decision")
                                continue

                            reward = get_reward(v_id, vehicle_initial_dists[v_id], vehicle_dests[v_id])
                            vehicle_episode_rewards[v_id] += reward
                            print(f"[DEBUG] Computed reward for {v_id}: {reward:.3f}, cumulative for vehicle: {vehicle_episode_rewards[v_id]:.3f}")

                            if last_decision_data[v_id] is not None:
                                prev_state, prev_action = last_decision_data[v_id]
                                print(f"[DEBUG] Agent learning step for {v_id}: prev_action={prev_action}")
                                agent.step(prev_state, prev_action, reward, current_state, False)

                            action = agent.act(current_state, eps=agent.epsilon)
                            print(f"[INFO] Agent chose action {action} (epsilon={agent.epsilon:.3f}) for vehicle {v_id}")
                            execute_action(v_id, action)
                            last_decision_data[v_id] = (current_state, action)
                        else:
                            # Not in decision zone
                            pass
                    except traci.TraCIException as e:
                        print(f"[WARNING] TraCI error while processing vehicle {v_id}: {e}")
                        continue

                # Handle arrivals
                try:
                    arrived = traci.simulation.getArrivedIDList()
                except traci.TraCIException:
                    arrived = []
                if arrived:
                    print(f"[DEBUG] Arrived vehicles at step {step}: {arrived}")

                for v_id in arrived:
                    if v_id in last_decision_data:
                        prev_state, prev_action = last_decision_data[v_id]
                        final_reward = 10.0
                        vehicle_episode_rewards[v_id] += final_reward
                        print(f"[INFO] Vehicle {v_id} arrived: awarding final reward {final_reward:.2f}, cumulative: {vehicle_episode_rewards[v_id]:.2f}")
                        # Use terminal=True here
                        agent.step(prev_state, prev_action, final_reward, prev_state, True)
                        del last_decision_data[v_id]

                    vehicle_dests.pop(v_id, None)
                    vehicle_initial_dists.pop(v_id, None)

                # Check if simulation has no expected vehicles
                try:
                    min_expected = traci.simulation.getMinExpectedNumber()
                except traci.TraCIException:
                    min_expected = None

                if min_expected == 0:
                    print(f"[INFO] All vehicles arrived — ending episode early at step {step}")
                    break

            # End of episode loop
        except Exception as e:
            print(f"[ERROR] Unexpected error during episode {episode}: {e}")
        finally:
            # Update epsilon and close traci safely
            try:
                agent.update_epsilon(episode)
                print(f"[DEBUG] Agent epsilon updated to {agent.epsilon:.3f}")
            except Exception as e:
                print(f"[WARNING] Failed to update epsilon: {e}")

            try:
                traci.close()
                print(f"[INFO] traci closed for episode {episode}")
            except Exception as e:
                print(f"[WARNING] traci.close() failed or was already closed: {e}")

        episode_reward = sum(vehicle_episode_rewards.values())
        scores.append(episode_reward)

        print(f"\n[RESULT] ✅ Episode {episode} completed")
        print(f"[RESULT] Total Reward: {episode_reward:.2f}")
        print(f"[RESULT] Epsilon: {agent.epsilon:.3f}")
        print("=" * 70)

    print("[INFO] Training finished")
    return scores


if __name__ == "__main__":
    print("[INFO] Launching training script")
    scores = train()
    print(f"[INFO] Training produced {len(scores)} episode scores")
