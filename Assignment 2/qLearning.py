import numpy as np

# Function to obtain node indices for given coordinates
def get_node_indices(coordinates, road_map):
    node_indices = []
    for (x, y) in coordinates:
        for i, row in enumerate(road_map):
            for j, (coord_x, coord_y) in enumerate(row):
                if (x, y) == (coord_x, coord_y):
                    node_indices.append(i * len(row) + j)
    return node_indices

# Function to initialize the Q-table
def initialize_q_table(num_states, num_actions):
    return np.zeros((num_states, num_actions))

# Function to choose an action using epsilon-greedy policy
def choose_action(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1])  # Explore
    else:
        return np.argmax(Q[state, :])  # Exploit

def update_q_value(Q, state, action, reward, next_state, alpha, gamma):
    Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]))
    return Q

def navigate_vehicle_Qlearning(Q, start, end, epsilon=0.1, alpha=0.1, gamma=0.6, num_episodes=1000):
    for episode in range(num_episodes):
        current_state = start
        while current_state != end:
            action = choose_action(Q, current_state, epsilon)
            
            # Simulate taking the action and get the next state and reward
            next_state = simulate_movement(current_state, action)
            reward = calculate_reward(next_state, end)
            
            # Update the Q-value
            Q = update_q_value(Q, current_state, action, reward, next_state, alpha, gamma)
            
            current_state = next_state

    return Q

def simulate_movement(current_state, action):
    return action  # For simplicity, assuming action is the next state

# Function to calculate the reward based on the current state and the goal state
def calculate_reward(current_state, goal_state):
    # Implement the logic to calculate the reward based on the current state and the goal state
    return -1 if current_state != goal_state else 0  # Negative reward until reaching the goal

# Function to choose the best path based on learned Q-values
def choose_best_path(Q, start, end):
    current_state = start
    path = [current_state]

    while current_state != end:
        action = np.argmax(Q[current_state, :])
        next_state = simulate_movement(current_state, action)
        path.append(next_state)
        current_state = next_state

    return path

def main():
    # Define the state space and action space
    road_map = [
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
        [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)],
        [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)],
        [(0, 3), (1, 3), (2, 3), (3, 3), (4, 3)],
        [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]
    ]

    # Define the vehicles as sources and destinations with coordinates. source = starting point, destination = end point
    vehicles = [{'source': (0, 0), 'destination': (3, 3)}, {'source': (4, 0), 'destination': (2, 3)}, {'source': (0, 3), 'destination': (4, 2)}]
        # Add more vehicles with source and destination coordinates as needed
    num_states = len(road_map) * len(road_map[0])
    num_actions = len(road_map) * len(road_map[0])

    # Initialize the Q-table
    Q = initialize_q_table(num_states, num_actions)

    # Loop through each vehicle and simulate navigation using Q-learning
    for i, vehicle in enumerate(vehicles):
        source_node = get_node_indices([vehicle['source']], road_map)[0]
        destination_node = get_node_indices([vehicle['destination']], road_map)[0]

        # Simulate navigation using Q-learning
        Q = navigate_vehicle_Qlearning(Q, source_node, destination_node)

        # Choose the best path based on learned Q-values
        best_path = choose_best_path(Q, source_node, destination_node)

        # Print the result for each vehicle
        if best_path is not None:
            print(f"Vehicle {i + 1}: Best Path = {best_path}")
        else:
            print(f"Vehicle {i + 1}: No path found")

if __name__ == "__main__":
    main()