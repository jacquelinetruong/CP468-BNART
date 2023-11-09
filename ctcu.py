import numpy as np
import heapq

# Algorithm that chooses the best neighbouring node for a vehicle to proceed to
def bnart(vehicle, road_map):
    source = vehicle['source']
    destination = vehicle['destination']

# Calculating the Euclidean distance between two coordinates 
def calculate_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Define a function to find the shortest time using Dijkstra's algorithm
def shortest_time_dijkstra(source, destination, distance_matrix):
    num_nodes = len(distance_matrix)
    visited = [False] * num_nodes
    shortest_times = [float('inf')] * num_nodes
    shortest_times[source] = 0

    priority_queue = [(0, source)]

    while priority_queue:
        _, current_node = heapq.heappop(priority_queue)
        if visited[current_node]:
            continue

        visited[current_node] = True

        for neighbor in range(num_nodes):
            if not visited[neighbor]:
                new_distance = shortest_times[current_node] + distance_matrix[current_node][neighbor]
                if new_distance < shortest_times[neighbor]:
                    shortest_times[neighbor] = new_distance
                    heapq.heappush(priority_queue, (new_distance, neighbor))

    return shortest_times[destination]

# Function to calculate the shortest time for each vehicle
def calculate_shortest_times(vehicles, distance_matrix, road_map):
    for vehicle in vehicles:
        source_coord = vehicle['source']
        destination_coord = vehicle['destination']
        source_node = get_node_indices([source_coord], road_map)[0]
        destination_node = get_node_indices([destination_coord], road_map)[0]

        # Calculate the shortest time using the distance matrix
        shortest_time = distance_matrix[source_node][destination_node]

        vehicle['shortest_time'] = shortest_time

# Function to obtain node indices for given coordinates
def get_node_indices(coordinates, road_map):
    node_indices = []
    for (x, y) in coordinates:
        for i, row in enumerate(road_map):
            for j, (coord_x, coord_y) in enumerate(row):
                if (x, y) == (coord_x, coord_y):
                    node_indices.append(i * len(row) + j)
    return node_indices

def main():
    
    road_map = [
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
        [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)],
        [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)],
        [(0, 3), (1, 3), (2, 3), (3, 3), (4, 3)],
        [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]
    ]

    # Define the vehicles as sources and destinations with coordinates. source = starting point, destination = end point
    vehicles = [
        {'source': (0, 0), 'destination': (3, 3)},
        {'source': (4, 0), 'destination': (2, 3)},
        {'source': (0, 3), 'destination': (4, 2)},
        {'source': (0,0), 'destination': (1,1)}

        # Add more vehicles with source and destination coordinates as needed
    ]
    # Print the road map
    for row in road_map:
        print(row)
        
    print(vehicles)

    # Create a distance matrix based on the Euclidean distances between coordinates
    num_nodes = len(road_map) * len(road_map[0])
    distance_matrix = np.zeros((num_nodes, num_nodes))

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

    for row in distance_matrix:
        for value in row:
            print(f"{value:.1f}", end="\t")  # Prints with one decimal place and tab separation
        print()  # Move to the next row

    # Call the function to calculate shortest times
    calculate_shortest_times(vehicles, distance_matrix, road_map)

    # Print the result
    for i, vehicle in enumerate(vehicles):
        print(f"Vehicle {i + 1}: Shortest Time = {vehicle['shortest_time']}")

if __name__ == "__main__":
    main()