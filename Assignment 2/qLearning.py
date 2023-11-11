import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
road_map = [
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
        [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)],
        [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)],
        [(0, 3), (1, 3), (2, 3), (3, 3), (4, 3)],
        [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]
    ]
# Function to obtain node indices for given coordinates
def get_node_indices(coordinates, road_map):
    node_indices = []
    for (x, y) in coordinates:
        for i, row in enumerate(road_map):
            for j, (coord_x, coord_y) in enumerate(row):
                if (x, y) == (coord_x, coord_y):
                    node_indices.append(i * len(row) + j)
    return node_indices

def get_coordinates(node_index, road_map):
    num_rows = len(road_map)
    num_cols = len(road_map[0])
    
    row = node_index // num_cols
    col = node_index % num_cols
    
    return road_map[row][col]
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

def navigate_vehicle_Qlearning(Q, start, end, road_map, epsilon=0.1, alpha=0.1, gamma=0.6, num_episodes=1000):
    episode_rewards = []

    for episode in range(num_episodes):
        current_state = start
        total_reward = 0  # Track the total reward for the episode

        while current_state != end:
            action = choose_action(Q, current_state, epsilon)

            # Simulate taking the action and get the next state and reward
            next_state = simulate_movement(current_state, action)
            reward = calculate_reward(next_state, end, road_map)
            
            # Update the Q-value
            Q = update_q_value(Q, current_state, action, reward, next_state, alpha, gamma)
            
            total_reward += reward  # Accumulate the reward for the episode

            current_state = next_state

        episode_rewards.append(total_reward)
        # Testing purposes 
        #print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    # Plot the learning curve
    import matplotlib.pyplot as plt
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

    return Q

def simulate_movement(current_state, action):
    row, col = divmod(current_state, len(road_map[0]))

    if action == 0:  # Move right
        col = min(col + 1, len(road_map[0]) - 1)
    elif action == 1:  # Move left
        col = max(col - 1, 0)
    elif action == 2:  # Move up
        row = max(row - 1, 0)
    elif action == 3:  # Move down
        row = min(row + 1, len(road_map) - 1)

    return row * len(road_map[0]) + col


def calculate_reward(current_state, goal_state, road_map):
    current_coordinates = get_coordinates(current_state, road_map)
    goal_coordinates = get_coordinates(goal_state, road_map)

    # Calculate Euclidean distance between current state and goal state
    distance = np.sqrt((current_coordinates[0] - goal_coordinates[0]) ** 2 +
                      (current_coordinates[1] - goal_coordinates[1]) ** 2)

    # Return negative distance as the reward
    return -distance

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

def convert_path_to_coordinates(path, road_map):
    coordinates_path = [get_coordinates(node_index, road_map) for node_index in path]
    return coordinates_path

def visualize_grid(road_map, vehicles):
    fig, ax = plt.subplots()

    # Plot grid lines
    for i in range(len(road_map)):
        for j in range(len(road_map[0])):
            ax.add_patch(patches.Rectangle((i, j), 1, 1, linewidth=1, edgecolor='black', facecolor='white'))

    # Plot vehicles
    for vehicle in vehicles:
        source = vehicle['source']
        destination = vehicle['destination']
        ax.add_patch(patches.Rectangle(source, 1, 1, linewidth=1, edgecolor='blue', facecolor='lightblue'))
        ax.add_patch(patches.Rectangle(destination, 1, 1, linewidth=1, edgecolor='red', facecolor='lightcoral'))

    plt.xlim(0, len(road_map))
    plt.ylim(0, len(road_map[0]))
    plt.gca().invert_yaxis()  # Invert y-axis to match the grid representation
    plt.show()


def visualize_grid_with_paths(road_map, vehicles, paths):
    fig, ax = plt.subplots()

    # Plot grid lines
    for i in range(len(road_map)):
        for j in range(len(road_map[0])):
            ax.add_patch(patches.Rectangle((i, j), 1, 1, linewidth=1, edgecolor='black', facecolor='white'))

    # Plot vehicles
    for vehicle, path in zip(vehicles, paths):
        source = vehicle['source']
        destination = vehicle['destination']
        ax.add_patch(patches.Rectangle(source, 1, 1, linewidth=1, edgecolor='blue', facecolor='lightblue'))
        ax.add_patch(patches.Rectangle(destination, 1, 1, linewidth=1, edgecolor='red', facecolor='lightcoral'))

        # Plot path
        path_coordinates = [get_coordinates(node_index, road_map) for node_index in path]
        ax.plot(*zip(*path_coordinates), marker='o', color='green', markersize=8, linewidth=2)

    plt.xlim(0, len(road_map))
    plt.ylim(0, len(road_map[0]))
    plt.gca().invert_yaxis()  # Invert y-axis to match the grid representation
    plt.show()



def main():
    # Define the state space and action space
    road_map = [
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
        [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)],
        [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)],
        [(0, 3), (1, 3), (2, 3), (3, 3), (4, 3)],
        [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]
    ]
    num_actions = 4  # Number of possible actions: right, left, up, down

    # Define the vehicles as sources and destinations with coordinates. source = starting point, destination = end point
    vehicles = [{'source': (0, 0), 'destination': (3, 3)}, 
                {'source': (4, 0), 'destination': (2, 3)}, 
                {'source': (0, 3), 'destination': (4, 1)}]
        # Add more vehicles with source and destination coordinates as needed

    # Visualize the grid and vehicles
    visualize_grid(road_map, vehicles)  

    num_states = len(road_map) * len(road_map[0])
    num_actions = len(road_map) * len(road_map[0])

    # Initialize the Q-table
    Q = initialize_q_table(num_states, num_actions)

    best_paths = [] 

    # Loop through each vehicle and simulate navigation using Q-learning
    for i, vehicle in enumerate(vehicles):
        source_node = get_node_indices([vehicle['source']], road_map)[0]
        destination_node = get_node_indices([vehicle['destination']], road_map)[0]

        # Simulate navigation using Q-learning
        Q = navigate_vehicle_Qlearning(Q, source_node, destination_node, road_map)

        # Choose the best path based on learned Q-values
        best_path = choose_best_path(Q, source_node, destination_node)
        best_path_coordinates = convert_path_to_coordinates(best_path, road_map)

        # Print the result for each vehicle
        if best_path is not None:
            best_paths.append(best_path)
            print(f"Vehicle {i + 1}: Best Path = {best_path}")
            print("Best Path (Coordinates):", best_path_coordinates)
        else:
            print(f"Vehicle {i + 1}: No path found")
        
    
    visualize_grid_with_paths(road_map, vehicles, best_paths)

if __name__ == "__main__":
    main()
