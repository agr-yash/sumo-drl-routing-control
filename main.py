import os
import sys
import traci
import numpy as np
from collections import defaultdict

from dqn_agent import DQNAgent

if 'SUMO_HOME' in os.environ:
    sumo_home = os.environ['SUMO_HOME']  # ✅ fetch the value, not the whole dict
    tools = os.path.join(sumo_home, 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

SUMO_BINARY = "sumo"  # Use "sumo" for faster, non-GUI training
SUMO_CFG = "simulation.sumocfg"
SUMO_CMD = [
        SUMO_BINARY,
        "-c", "simulation.sumocfg",
        "--tripinfo-output", "tripinfo.xml"
    ]

STATE_SIZE = 20  
ACTION_SIZE = 3  # Right, Straight, Left
NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 5400 # Max simulation time

# --- Helper Functions ---
def get_state(vehicle_id, destination_pos):
    """
    Constructs the full 20-element state vector for a given AV agent.
    This function implements the state representation as defined in the paper,
    gathering data for the current road and the three potential outgoing roads.

    Args:
        vehicle_id (str): The ID of the autonomous vehicle.
        destination_pos (tuple): The (x, y) coordinates of the vehicle's destination.

    Returns:
        np.array: A 20-element NumPy array representing the state, or None if state cannot be formed.
    """
    # --- 1. Identify Relevant Roads (Current + 3 Outgoing) ---
    road_ids = {}  # Dict to store edge IDs for 'current', 'left', 'straight', 'right'
    road_ids['current'] = traci.vehicle.getRoadID(vehicle_id)

    try:
        # getNextLinks provides a list of possible connections from the current lane
        # Each link is a tuple: (viaLaneID, toLaneID, direction, length,...)
        links = traci.vehicle.getNextLinks(vehicle_id)
        if not links:
            return None # Cannot make a decision if there are no outgoing links

        # Map turn directions to outgoing edge IDs
        for link in links:
            direction = link[1] # 'r' for right, 's' for straight, 'l' for left, etc.
            to_edge = traci.lane.getEdgeID(link[2])
            if 'r' in direction:
                road_ids['right'] = to_edge
            elif 's' in direction:
                road_ids['straight'] = to_edge
            elif 'l' in direction:
                road_ids['left'] = to_edge
    except traci.TraCIException:
        return None # Vehicle might have left the network or is in a state with no links

    # --- 2. Extract Traffic Features for Each Road (4 roads x 4 features = 16 values) ---
    traffic_features = []
    # The order is critical for the DQN: current, right, straight, left
    road_order = ['current', 'right', 'straight', 'left']

    for road_key in road_order:
        edge_id = road_ids.get(road_key) # Use.get() to handle missing turns (e.g., no left turn)
        
        av_count, hv_count = 0, 0
        av_speeds, hv_speeds = 0.0, 0.0

        if edge_id:
            vehicles_on_edge = traci.edge.getLastStepVehicleIDs(edge_id)
            for v_id in vehicles_on_edge:
                if traci.vehicle.getTypeID(v_id) == "AV":
                    av_count += 1
                    av_speeds.append(traci.vehicle.getSpeed(v_id))
                else: # Assumes any non-AV is an HV
                    hv_count += 1
                    hv_speeds.append(traci.vehicle.getSpeed(v_id))
        
        # Calculate average speeds, handling the case of zero vehicles
        avg_av_speed = np.mean(av_speeds) if av_speeds else 0.0
        avg_hv_speed = np.mean(hv_speeds) if hv_speeds else 0.0

        # Append the 4 features for this road
        traffic_features.extend([av_count, hv_count, avg_av_speed, avg_hv_speed])

    # --- 3. Extract Positional Features (4 values) ---
    current_pos = traci.vehicle.getPosition(vehicle_id)
    positional_features = [current_pos, current_pos[2], destination_pos, destination_pos[2]]

    # --- 4. Combine and Normalize ---
    state_vector = np.array(traffic_features + positional_features, dtype=np.float32)
    
    # Apply normalization (example values, should be tuned based on network specifics)
    # Normalize counts by a reasonable maximum capacity per edge
    state_vector[0:16:4] /= 50.0  # AV Counts
    state_vector[1:16:4] /= 50.0  # HV Counts
    # Normalize speeds by the simulation's max speed
    state_vector[2:16:4] /= 15.0  # AV Speeds
    state_vector[3:16:4] /= 15.0  # HV Speeds
    # Normalizing positions can be complex; for a fixed map, it might not be necessary
    # if the network learns the scale. If using different maps, normalize by map dimensions.

    return state_vector

def get_reward(vehicle_id, initial_dist_to_dest, destination_pos):
    """
    Calculates the reward based on the paper's multi-objective function.
    r_n = omega * ds_n - (1 - omega) * dd_n

    Args:
        vehicle_id (str): The ID of the autonomous vehicle.
        initial_dist_to_dest (float): The initial Euclidean distance to the destination at spawn time.
        destination_pos (tuple): The (x, y) coordinates of the vehicle's destination.

    Returns:
        float: The calculated reward value.
    """
    omega = 0.6  # Weight parameter as specified in the paper 

    # --- Calculate Normalized Driving Speed (ds_n) ---
    current_speed = traci.vehicle.getSpeed(vehicle_id)
    # Use the vehicle's allowed max speed for normalization, which is more stable than lane speed
    max_speed = traci.vehicle.getMaxSpeed(vehicle_id)
    ds_n = current_speed / max_speed if max_speed > 0 else 0.0

    # --- Calculate Normalized Remaining Distance (dd_n) ---
    current_pos = np.array(traci.vehicle.getPosition(vehicle_id))
    current_dist_to_dest = np.linalg.norm(current_pos - np.array(destination_pos))
    
    # Normalize by the initial distance to provide a stable, decreasing signal
    dd_n = current_dist_to_dest / initial_dist_to_dest if initial_dist_to_dest > 0 else 0.0

    # --- Final Reward Calculation ---
    reward = (omega * ds_n) - ((1 - omega) * dd_n)
    return reward

def execute_action(vehicle_id, action):
    """
    Translates a discrete action (0: right, 1: straight, 2: left) into a
    SUMO command to change the vehicle's target edge.

    Args:
        vehicle_id (str): The ID of the autonomous vehicle.
        action (int): The action index chosen by the DQN agent.
    """
    # Map action indices to SUMO's turn directions
    action_to_direction = {0: 'r', 1: 's', 2: 'l'}
    chosen_direction = action_to_direction.get(action)

    if chosen_direction is None:
        return # Invalid action index

    try:
        links = traci.vehicle.getNextLinks(vehicle_id)
        if not links:
            return # No possible turns from here

        # Find the link that corresponds to the chosen action
        for link in links:
            direction = link[1]
            if chosen_direction in direction:
                target_lane = link[2]
                target_edge = traci.lane.getEdgeID(target_lane)
                
                # Command the vehicle to change its route to the new target edge
                traci.vehicle.changeTarget(vehicle_id, target_edge)
                return # Action executed successfully
    
    except traci.TraCIException:
        # This can happen if the vehicle is teleported or leaves the simulation
        # between the decision and execution step. It's safe to ignore.
        pass

def train():
    agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    scores =

    for episode in range(1, NUM_EPISODES + 1):
        traci.start(SUMO_CMD)
        
        step = 0
        episode_reward = 0
        
        # Dictionaries to track vehicle-specific data
        vehicle_dests = {}
        vehicle_initial_dists = {}
        # This dict now stores the state and action from the PREVIOUS decision zone
        last_decision_data = defaultdict(lambda: None)

        while step < MAX_STEPS_PER_EPISODE:
            traci.simulationStep()
            
            # --- Handle newly departed vehicles ---
            departed_ids = traci.simulation.getDepartedIDList()
            for v_id in departed_ids:
                if traci.vehicle.getTypeID(v_id) == "AV":
                    dest_edge = traci.vehicle.getRoute(v_id)[-1]
                    # A robust way to get destination coordinates (center of the destination edge)
                    lane_id = dest_edge + "_0"
                    lane_shape = traci.lane.getShape(lane_id)
                    dest_pos = lane_shape[1] # Use the endpoint of the lane
                    vehicle_dests[v_id] = dest_pos
                    
                    pos = traci.vehicle.getPosition(v_id)
                    initial_dist = np.linalg.norm(np.array(pos) - np.array(dest_pos))
                    vehicle_initial_dists[v_id] = initial_dist

            # --- Handle AVs that are currently active ---
            active_av_ids = {}
            
            for v_id in active_av_ids:
                # --- Decision Zone Logic ---
                current_lane = traci.vehicle.getLaneID(v_id)
                if current_lane.startswith(':'): # Skip if inside an intersection
                    continue

                lane_length = traci.lane.getLength(current_lane)
                pos_on_lane = traci.vehicle.getLanePosition(v_id)
                dist_to_intersection = lane_length - pos_on_lane

                if dist_to_intersection <= DECISION_ZONE_LENGTH:
                    # Vehicle is in a decision zone, get its current state
                    current_state = get_state(v_id, vehicle_dests[v_id])
                    if current_state is None:
                        continue

                    # If there's data from a previous decision, we can now form a complete experience tuple
                    if last_decision_data[v_id] is not None:
                        prev_state, prev_action = last_decision_data[v_id]
                        reward = get_reward(v_id, vehicle_initial_dists[v_id], vehicle_dests[v_id])
                        done = False # 'done' is only true when the vehicle arrives
                        
                        # This is the crucial learning step
                        agent.step(prev_state, prev_action, reward, current_state, done)
                        episode_reward += reward

                    # Agent chooses an action based on the new state
                    action = agent.act(current_state)
                    execute_action(v_id, action)
                    
                    # Store the current state and action for the *next* time this vehicle hits a decision zone
                    last_decision_data[v_id] = (current_state, action)

            step += 1
            if traci.simulation.getMinExpectedNumber() == 0:
                break # End episode if all vehicles have arrived

        traci.close()
        scores.append(episode_reward)
        print(f"Episode: {episode}/{NUM_EPISODES}, Total Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

if __name__ == "__main__":
    # Create a helpers.py file and put the get_state, get_reward, execute_action functions there
    # Or, paste them directly into this file above the train() function.
    train()