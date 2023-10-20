# These are all just basic potential functions

# Takes node (current state) of vehicle and determines neighbouring nodes (where to go)
def find_directions(node):
    return

# Calculates distance between current node and destination node
def get_distance(node, destination):
    return

# Estimates traffic time between two nodes based on current vehicles in ctcu
def get_traffic(node, neighbour, vehicle_locations):
    return

# Determines if the current node is an intersection
def is_intersection(node):
    return

# Calculate reward for a certain action (moving nodes)
def get_reward(vehicle, action, node):
    return

# Normalize traffic time and distance to ensure proportionate decision making
def normalize(value):
    return

# Algorithm that chooses the best neighbouring node for a vehicle to proceed to
def bnart(vehicle, current_location, destination):
    return best_neighbour

# Central Traffic Control Unit
def main():
    return