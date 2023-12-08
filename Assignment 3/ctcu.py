import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Algorithm that chooses the best neighbouring node for a vehicle to proceed to
def bnart(vehicle, road_map):
    source = vehicle['source']
    destination = vehicle['destination']

def get_coordinates(node_index, road_map):
    num_rows = len(road_map)
    num_cols = len(road_map[0])
    
    row = node_index // num_cols
    col = node_index % num_cols
    
    return road_map[row][col]

def convert_path_to_coordinates(path, road_map):
    coordinates_path = [get_coordinates(node_index, road_map) for node_index in path]
    return coordinates_path

# Calculating the Euclidean distance between two coordinates 
# Function to calculate the shortest time using Dijkstra's algorithm
def calculate_shortest_time_dijkstra(graph, start, end):
    shortest_times = {node: float('inf') for node in graph}
    shortest_times[start] = 0
    previous_nodes = {}
    nodes = set(graph)

    while nodes:
        current_node = min(nodes, key=lambda node: shortest_times[node])
        nodes.remove(current_node)
        if shortest_times[current_node] == float('inf'):
            break
        for neighbor, distance in graph[current_node].items():
            potential_time = shortest_times[current_node] + distance
            if potential_time < shortest_times[neighbor]:
                shortest_times[neighbor] = potential_time
                previous_nodes[neighbor] = current_node

    path, current_node = [], end
    while current_node in previous_nodes:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]
    if path:
        path.insert(0, start)
        return path, shortest_times[end]
    else:
        return None, None
    
# # Calculating the Euclidean distance between two coordinates 
def calculate_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to obtain node indices for given coordinates
def get_node_indices(coordinates, road_map):
    node_indices = []
    for (x, y) in coordinates:
        for i, row in enumerate(road_map):
            for j, (coord_x, coord_y) in enumerate(row):
                if (x, y) == (coord_x, coord_y):
                    node_indices.append(i * len(row) + j)
    return node_indices

def visualize_grid_with_points(road_map, vehicles):
    fig, ax = plt.subplots()

    # Plot grid lines
    for i in range(len(road_map)):
        for j in range(len(road_map[0])):
            ax.add_patch(patches.Rectangle((i, j), 1, 1, linewidth=1, edgecolor='black', facecolor='white'))

    # Scatter plot for points
    all_points = [(i, j) for row in road_map for i, j in row]
    x, y = zip(*all_points)
    ax.scatter(x, y, c='gray', marker='o', s=100, label='Grid Points')

    # Plot vehicles
    for vehicle in vehicles:
        source = vehicle['source']
        destination = vehicle['destination']

        # Add vehicle points to the scatter plot
        vehicle_points = [source]
        vx, vy = zip(*vehicle_points)
        ax.scatter(vx, vy, c='black', marker='X', s=100)

        destination_points = [destination]
        vx, vy = zip(*destination_points)
        ax.scatter(vx, vy, c='green', marker='s', s=100)

    ax.scatter(vx, vy, c='black', marker='X', s=100, label="Source")
    ax.scatter(vx, vy, c='green', marker='s', s=100, label="Destination")
    plt.xlim(0, len(road_map))
    plt.ylim(0, len(road_map[0]))
    plt.legend()
    plt.title('Initial State')
    plt.show()

def visualize_grid_with_arrows(road_map, vehicles, best_path_coordinates):
    fig, ax = plt.subplots()

    # Plot grid lines
    for i in range(len(road_map)):
        for j in range(len(road_map[0])):
            ax.add_patch(patches.Rectangle((i, j), 1, 1, linewidth=1, edgecolor='black', facecolor='white'))

    # Scatter plot for points
    all_points = [(i, j) for row in road_map for i, j in row]
    x, y = zip(*all_points)
    ax.scatter(x, y, c='gray', marker='o', s=100, label='Grid Points')

    # Plot vehicles
    for vehicle in vehicles:
        source = vehicle['source']
        destination = vehicle['destination']

        # Add vehicle points to the scatter plot
        source_points = [source]
        sx, sy = zip(*source_points)
        ax.scatter(sx, sy, c='black', marker='X', s=100)

        destination_points = [destination]
        dx, dy = zip(*destination_points)
        ax.scatter(dx, dy, c='green', marker='s', s=100)

    ax.scatter(sx, sy, c='black', marker='X', s=100, label="Source")
    ax.scatter(dx, dy, c='green', marker='s', s=100, label="Destination")

    # Plot best path
    path_coordinates = np.array(best_path_coordinates)
    ax.plot(path_coordinates[:, 0], path_coordinates[:, 1], marker='o', color='blue', markersize=8, linewidth=2, label='Best Path')

    # Plot arrows to represent movement
    for i in range(len(best_path_coordinates) - 1):
        dx = best_path_coordinates[i + 1][0] - best_path_coordinates[i][0]
        dy = best_path_coordinates[i + 1][1] - best_path_coordinates[i][1]
        ax.arrow(best_path_coordinates[i][0], best_path_coordinates[i][1], dx, dy, head_width=0.1, head_length=0.1, fc='red', ec='red')

    
    plt.xlim(0, len(road_map))
    plt.ylim(0, len(road_map[0]))
    plt.legend()
    plt.title("Best Path")
    plt.show()


def main():
    

    road_map = [
        [(0, 4), (1, 0), (2, 0), (3, 0), (4, 0)],
        [(0, 3), (1, 1), (2, 1), (3, 1), (4, 1)],
        [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)],
        [(0, 1), (1, 3), (2, 3), (3, 3), (4, 3)],
        [(0, 0), (1, 4), (2, 4), (3, 4), (4, 4)]
    ]

    # Define the vehicles as sources and destinations with coordinates. source = starting point, destination = end point
    vehicles = [
        {'source': (1, 1), 'destination': (3, 3)},
        {'source': (4, 1), 'destination': (2, 3)},
        {'source': (1, 3), 'destination': (4, 2)}

        # Add more vehicles with source and destination coordinates as needed
    ]

    # Visualize the grid and vehicles
    visualize_grid_with_points(road_map, vehicles) 
    
    # Create a distance matrix based on the Euclidean distances between coordinates
    num_nodes = len(road_map) * len(road_map[0])
    distance_matrix = np.zeros((num_nodes, num_nodes))

    graph = {}

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if i not in graph:
                graph[i] = {}
            if j not in graph:
                graph[j] = {}
            graph[i][j] = distance_matrix[i][j]
            graph[j][i] = distance_matrix[j][i]

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            coord1 = road_map[i // 5][i % 5]
            coord2 = road_map[j // 5][j % 5]
            # Calculate the distance
            distance = calculate_distance(coord1, coord2)
            # Round the distance to one decimal place
            rounded_distance = round(distance, 1)
            # Assign the rounded distance to the matrix
            distance_matrix[i][j] = rounded_distance
            distance_matrix[j][i] = rounded_distance
            # Update the graph with the distances
            graph[i][j] = rounded_distance
            graph[j][i] = rounded_distance

    # Print the road map
    for row in road_map:
        print(row)
        
    print(vehicles)

    # Loop through each vehicle and calculate the shortest time and path using Dijkstra's algorithm
    for i, vehicle in enumerate(vehicles):
        source_node = get_node_indices([vehicle['source']], road_map)[0]
        destination_node = get_node_indices([vehicle['destination']], road_map)[0]
        
        # Call the function to calculate shortest times using Dijkstra's algorithm
        path, shortest_time = calculate_shortest_time_dijkstra(graph, source_node, destination_node)
        
        best_path_coord = convert_path_to_coordinates(path, road_map)

        # Print the result for each vehicle
        if path is not None:
            print(f"Vehicle {i + 1}: Shortest Time = {shortest_time:.1f}")
            print("Path Taken:", best_path_coord)
        else:
            print(f"Vehicle {i + 1}: No path found")

        visualize_grid_with_arrows(road_map, vehicles, best_path_coord)

if __name__ == "__main__":
    main()