import math
import matplotlib.pyplot as plt
from collections import deque


def plot_circles(obs_dict, fig = None, ax = None, plot = True):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    for (x, y), radius in obs_dict.items():
        circle = plt.Circle((x, y), radius, color='blue', fill=False, zorder = 2)
        ax.add_patch(circle)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlim(-3.5, 3.5)
    ax.set_aspect('equal') # , adjustable='box')    
    
    if plot:
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Obstacle and Gate Circles')
        plt.grid(True)
        plt.show()

    return fig, ax

def plot_points(points, fig = None, ax = None, plot = True):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    
    xs = []
    ys = []
    for point in points:
        xs.append(point[0])
        ys.append(point[1])
    
    ax.scatter(xs, ys, s = 4, zorder = 1)
    if plot:
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Obstacle and Gate Points')
        plt.grid(True)
        plt.show()

    return fig, ax

def plot_graph(graph, fig = None, ax = None, plot = True):
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # Plot edges from the graph
    plotted_edges = set()
    for point, neighbours in graph.items():
        for neighbour in neighbours:
            neighbor_point = neighbour[0]
            edge = tuple(sorted([point, neighbor_point]))
            if edge not in plotted_edges:
                plotted_edges.add(edge)
                ax.plot([point[0], neighbor_point[0]], [point[1], neighbor_point[1]], linewidth=0.5, color = 'cyan', zorder = 0)
    
    if plot:
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Graph Edges')
        plt.grid(True)
        plt.show()

    return fig, ax

def plot_list(points, fig = None, ax = None, plot = True):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    
    xs = []
    ys = []
    for point in points:
        xs.append(point[0])
        ys.append(point[1])
    
    ax.plot(xs, ys, linewidth=0.5, color = 'red', zorder = 2)
    if plot:
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Path')
        plt.grid(True)
        plt.show()

    return fig, ax

"""
get dict of (x, y): obstacle radius given gates, obstacles list
consider a gate as 2 obstacles spaced apart
"""
def get_obs_dict(gate_list, obs_list, gate_spacing = 0.42, gate_rad = 0.04, obs_rad = 0.06):
    obs_dict = {}
    for obs in obs_list:
        obs_dict[(obs[0], obs[1])] = obs_rad
    
    for gate in gate_list:
        gate_x, gate_y, gate_rot = gate[0], gate[1], gate[5]

        gate_x1 = gate_x + (gate_spacing)/2 * math.cos(gate_rot)
        gate_y1 = gate_y + (gate_spacing)/2 * math.sin(gate_rot)

        gate_x2 = gate_x - (gate_spacing)/2 * math.cos(gate_rot)
        gate_y2 = gate_y - (gate_spacing)/2 * math.sin(gate_rot)

        obs_dict[(gate_x1, gate_y1)] = gate_rad
        obs_dict[(gate_x2, gate_y2)] = gate_rad

    return obs_dict

"""
generate a set of waypoints and a list of goals for bfs given both the gates and obstacles
goals are generated assuming start > gate_list (in order) > end
"""
def generate_waypoints_goal_list(gate_list, obs_list, start_pt, end_pt, 
                                 obs_waypoint_ct = 6, gate_waypoint_ct = 6, obs_waypont_dist = 0.50, 
                                 gate_dist = 0.5, gate_spacing = 0.42):
    # Generate waypoints around obstacles
    waypoints = []
    goals = [[start_pt]]
    goals_angles = {start_pt: None}

    for obs in obs_list:
        x, y = obs[0], obs[1]
        for i in range(obs_waypoint_ct):
            angle = 2 * math.pi * i / obs_waypoint_ct
            waypoint_x = x + obs_waypont_dist * math.cos(angle)
            waypoint_y = y + obs_waypont_dist * math.sin(angle)
            waypoints.append((waypoint_x, waypoint_y))

    # Generate waypoints around gates
    for gate in gate_list:
        gate_x, gate_y, gate_rot = gate[0], gate[1], gate[5]
        
        # determineing locations of "obstacles" around gates, aka the gate bars
        gate_x1_obs = gate_x + (gate_spacing)/2 * math.cos(gate_rot)
        gate_y1_obs = gate_y + (gate_spacing)/2 * math.sin(gate_rot)

        gate_x2_obs = gate_x - (gate_spacing)/2 * math.cos(gate_rot)
        gate_y2_obs = gate_y - (gate_spacing)/2 * math.sin(gate_rot)

        # extra waypoints around gate
        for i in range(gate_waypoint_ct):
            angle = 2 * math.pi * i / obs_waypoint_ct
            waypoint_x = gate_x1_obs + obs_waypont_dist * math.cos(angle)
            waypoint_y = gate_y1_obs + obs_waypont_dist * math.sin(angle)
            waypoints.append((waypoint_x, waypoint_y))

            waypoint_x = gate_x2_obs + obs_waypont_dist * math.cos(angle)
            waypoint_y = gate_y2_obs + obs_waypont_dist * math.sin(angle)
            waypoints.append((waypoint_x, waypoint_y))
        
        gate_x1 = gate_x + (gate_dist)/2 * math.sin(gate_rot)
        gate_y1 = gate_y + (gate_dist)/2 * math.cos(gate_rot)

        gate_x2 = gate_x - (gate_dist)/2 * math.sin(gate_rot)
        gate_y2 = gate_y - (gate_dist)/2 * math.cos(gate_rot)

        waypoints += [(gate_x1, gate_y1),(gate_x2, gate_y2)]
        goals += [[(gate_x1, gate_y1),(gate_x, gate_y),(gate_x2, gate_y2)]] 

        goals_angles[(gate_x1, gate_y1)] = gate_rot
        goals_angles[(gate_x, gate_y)] = gate_rot
        goals_angles[(gate_x2, gate_y2)] = gate_rot
        
    goals += [[end_pt]]
    goals_angles[end_pt] = None
    return waypoints, goals, goals_angles

def get_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1]-p2[1])**2)

"""
generate a graph given the waypoint list, obstacle list, current position, and target position.
graph should account for intersections between edges and obstacles
"""
def generate_graph(waypoints_raw, obs_dict, start, goal):
    waypoints = waypoints_raw.copy() + [start, goal]
    graph = {}
    
    for i, waypoint in enumerate(waypoints):
        if waypoint not in graph:
            graph[waypoint] = set()
        for j, other_waypoint in enumerate(waypoints):
            if i != j and other_waypoint not in graph[waypoint]:
                # Check if the line between the waypoints intersects with any obstacles
                if not intersects_obstacles(waypoint, other_waypoint, obs_dict):
                    edge_dist = get_dist(waypoint, other_waypoint)

                    # add the edge to the graph
                    graph[waypoint].add((other_waypoint, edge_dist))

                    # add the reverse edge to the graph
                    if other_waypoint not in graph:
                        graph[other_waypoint] = {(waypoint, edge_dist)}
                    else:
                        if waypoint not in graph[other_waypoint]:
                            graph[other_waypoint].add((waypoint, edge_dist))

    return graph

"""
iter through the obstacle dict and check if the line between p1 and p2 intersects with any obstacles
"""
def intersects_obstacles(p1, p2, obs_dict):
    # Check if the line between p1 and p2 intersects with any obstacles in obs_dict
    for (x, y), radius in obs_dict.items():
        if intersect_circle(p1, p2, (x, y), radius):
            return True
    return False

"""
check to see if the line between p1 and p2 intersects with a circle at (x, y) with radius r
"""
def intersect_circle(line_start, line_end, circle_center, r):
        # Calculate the distance from the point to the line segment
        px, py = circle_center
        x1, y1 = line_start
        x2, y2 = line_end

        # account for vertical or horizontal lines
        if x1 == x2:
            proj_x = x1
            proj_y = py
        elif y1 == y2:
            proj_x = px
            proj_y = y1
        else:
            # Project point onto the line segment
            m1 = ((y2 - y1) / (x2 - x1))
            m2 = -1/m1
            b1 = y1-m1*x1
            b2 = py-m2*px

            proj_x = (b2-b1)/(m1-m2)
            proj_y = m1*proj_x + b1

        # clip the point to line start/end
        proj_x = max(min(x1, x2), proj_x)
        proj_x = min(max(x1, x2), proj_x)
        proj_y = max(min(y1, y2), proj_y)
        proj_y = min(max(y1, y2), proj_y)

        # Distance squared from the point to the projection
        dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2

        # Check if the distance is within the radius
        return dist_sq <= r ** 2

"""
given a graph and a list of points that must be traversed, run BFS to find the shortest path
while traversing the points in order

no hairpin: prevent drone from going straight backwards through the same gate it came through
"""
def bfs_multiple_goals(graph, goal_list, no_hairpin, hairpin_blockrad = 0.2):
    total_path = []
    goal_starts = [] # bool array with True corresponding to path entry that indicates a start of a "goal sequence"
    total_len = 0
    prev_mid = None

    goal = goal_list[0][0]
    for i in range(0, len(goal_list)-1):
        start = goal
        goal_item = goal_list[i+1]

        # if 2x goals at a point (need to pass through), run bfs to find the nearest 
        # then go to that goal, then the paired goal, and then continue
        if len(goal_item) == 1:
            goal = goal_item[0]

            # preventing hairpins
            obs = None
            if no_hairpin:
                if prev_mid is not None:
                    obs = {prev_mid: hairpin_blockrad}

            path, length = bfs_shortest_path(graph, start, goal, obs)
            goal_starts += [False] * len(path)
        else:
            goal1 = goal_item[0]
            goal_mid = goal_item[1]
            goal2 = goal_item[2]

            # preventing hairpins
            obs = None
            if no_hairpin:
                if prev_mid is not None:
                    obs = {prev_mid: hairpin_blockrad}

            prev_mid = goal_mid

            print(obs)
            path1, length1 = bfs_shortest_path(graph, start, goal1, obs)
            path2, length2 = bfs_shortest_path(graph, start, goal2, obs)
            if length1 < length2:
                path = path1 + [goal_mid]
                goal_starts += [False] * (len(path)-2)
                goal_starts += [True, False]
                goal = goal2
                length = length1 + get_dist(goal1, goal2)
            else:
                path = path2 + [goal_mid]
                goal_starts += [False] * (len(path)-2)
                goal_starts += [True, False]
                goal = goal1
                length = length2 + get_dist(goal1, goal2)
        print(path)
        total_path += path
        total_len += length
    return total_path, total_len, goal_starts

"""
pretty standard bfs algo with weighted edges

use obs list (circular obstacles) to prevent hairpins or block certain paths
"""
def bfs_shortest_path(graph, start, end, obs = None):
    # Queue to store (current_node, path, total_edge_length)

    queue = deque([start])
    # dict of tuples (prev point, path length)
    visited = {start:(None, 0)}
    end_path = float('inf')
    end_found = False

    while queue:
        current_node = queue.popleft()
        path_len = visited[current_node][1]

        # if the path is longer than an already found path, give up this line
        if path_len > end_path:
            continue

        # if we reach the end node, record it and give up this line
        if current_node == end:
            end_found = True
            if path_len < end_path:
                end_path = path_len
            continue

        # Explore neighbors
        for neighbour in graph.get(current_node, []):
            neighbour_point, edge_length = neighbour[0], neighbour[1]

            # if we're considering obstacles make sure our point doesnt clash
            if obs is not None:
                if intersects_obstacles(current_node, neighbour_point, obs):
                    # print(current_node, neighbour_point)
                    continue
            
            # explore if it hasn't been seen before, or we found a shorter path
            if neighbour_point not in visited:
                queue.append(neighbour_point)
                visited[neighbour_point] = (current_node, path_len + edge_length)
            elif visited[neighbour_point][1] > path_len + edge_length:
                queue.append(neighbour_point)
                visited[neighbour_point] = (current_node, path_len + edge_length)

    if end_found:
        # build path by reverse traversing the graph
        path = deque()
        node = end
        while node in visited:
            path.appendleft(node)
            prev_node = visited[node][0]
            if prev_node is None:
                break
            node = prev_node
        return list(path), end_path
    else:
        # If no path is found, return None
        return None, float('inf')
    
# returns arctan results but up to pi, and negatives go up to -pi
# from point1 -> point2
def atan_2pi(point_1, point_2):
    x_diff = point_2[0] - point_1[0]
    y_diff = point_2[1] - point_1[1]

    if x_diff == 0 and y_diff > 0:
        return 1.57 # vertical line
    elif x_diff == 0 and y_diff < 0:
        return 4.71
    
    arctan = math.atan(y_diff/x_diff)
    if x_diff >= 0 and y_diff >= 0: # sector 1
        return arctan
    elif x_diff <= 0 and y_diff >= 0: # sector 2
        return math.pi + arctan
    elif x_diff <= 0 and y_diff <= 0:  # sector 3
        return math.pi + arctan
    elif x_diff >= 0 and y_diff <= 0:  # sector 4
        return 2*math.pi + arctan
    
    return arctan # angle_to_2pi(arctan)

# convert radians to pi -pi range
# TODO validate
def angle_to_2pi(angle):
    # if angle > 2*math.pi:
    #     angle = angle % 2*math.pi
    # elif angle < -2*math.pi:
    #     angle = angle % 2*math.pi 

    # if angle >= math.pi:
    #     return angle - 2*math.pi
    # if angle <= -math.pi:
    #     return angle + 2*math.pi
    if angle > 2*math.pi:
        return angle - 2*math.pi
    elif angle < 0:
        return 2*math.pi + angle
    else:
        return angle

def wrap_angle(gate_angle, new_angle):
    if gate_angle > math.pi:
        if new_angle < gate_angle - math.pi:
            gate_angle = gate_angle - 2*math.pi
    elif gate_angle < math.pi:
        if new_angle > gate_angle + math.pi:
            new_angle = new_angle - 2*math.pi
    return gate_angle, new_angle

def postproc_angle_diff(angle_diff):
    if angle_diff > math.pi/2 or angle_diff < -math.pi/2:
        return 0
    return angle_diff
    # convert to range -pi, pi
    # if angle_diff_1 > math.pi:
    #     angle_diff_1 = 0
    #     # angle_diff_1 = 2*math.pi - angle_diff_1
    # if angle_diff_2 > math.pi:
    #     angle_diff_2 = 0
    #     # angle_diff_2 = 2*math.pi - angle_diff_2
    # if angle_diff_1 < -math.pi:
    #     angle_diff_1 = 0
    #     #angle_diff_1 = 2*math.pi + angle_diff_1
    # if angle_diff_2 < -math.pi:
    #     angle_diff_2 = 0
    #     # angle_diff_2 = 2*math.pi + angle_diff_2

def tweak_path(path, goal_starts, goals_angles, scale = 0.1, mid_scale = 0.1, max_offset = 0.5):
    # drop path duplicate points
    filt_path = path
    tuned_path = []
    skip = 0
    # for all non start/end points, tune the goal points to align with the former/prior graph lines
    for i, point in enumerate(filt_path):
        if goal_starts[i]:
            prev_point = filt_path[i-1]
            goal_1 = filt_path[i]
            midpoint = filt_path[i+1]
            goal_2 = filt_path[i+2]
            next_point = filt_path[i+3]

            skip = 2

            # get angles of gates in 0 to 2pi scale
            gate_angle_1 = atan_2pi(goal_2, goal_1)
            gate_angle_2 = atan_2pi(goal_1, goal_2)

            angle_1 = atan_2pi(goal_1, prev_point)
            gate_angle_1, angle_1 = wrap_angle(gate_angle_1, angle_1)
            angle_diff_1 = gate_angle_1 - angle_1

            angle_2 = atan_2pi(goal_2, next_point) 
            gate_angle_2, angle_2 = wrap_angle(gate_angle_2, angle_2) 
            angle_diff_2 = gate_angle_2 - angle_2

            angle_diff_1 = postproc_angle_diff(angle_diff_1)
            angle_diff_2 = postproc_angle_diff(angle_diff_2)

            # need to adjust goal point 1, goal point 2, and mid goal
            norm_vect = [math.cos(gate_angle_1 + 1.57), math.sin(gate_angle_1 + 1.57)]
            norm_length = math.sqrt(norm_vect[0]**2 + norm_vect[1]**2)

            scale_1 = min(max((scale * angle_diff_1), -max_offset), max_offset)
            scale_2 = min(max((scale * angle_diff_2), -max_offset), max_offset)
            
            norm_vect = [norm_vect[0] / norm_length, norm_vect[1] / norm_length]

            norm_vect_1 = [norm_vect[0] *scale_1, norm_vect[1] *scale_1]
            norm_vect_2 = [norm_vect[0] *scale_2, norm_vect[1] *scale_2]

            mid_vect = [norm_vect_2[0] - norm_vect_1[0], norm_vect_2[1] - norm_vect_1[1]]
            # print(goal_1, midpoint, goal_2)
            # print(norm_vect)
            # print(gate_angle_1, gate_angle_2)
            # print(angle_1, angle_2)
            # print(angle_diff_1, angle_diff_2)

            # norm_vect_1 = (0,0)
            # norm_vect_2 = (0,0)

            goal_1_adj = (goal_1[0] - norm_vect_1[0], goal_1[1] - norm_vect_1[1])
            goal_2_adj = (goal_2[0] + norm_vect_2[0], goal_2[1] + norm_vect_2[1])
            if scale == 0:
                ref_scale = 1
            else:
                ref_scale = scale
            midpoint_adj = (midpoint[0] + mid_vect[0]/2 * mid_scale/ref_scale, midpoint[1] + mid_vect[1]/2 * mid_scale/ref_scale)
            
            tuned_path += [goal_1_adj, midpoint_adj, goal_2_adj]
            # print([goal_1_adj, midpoint_adj, goal_2_adj])

        else:
            if skip > 0:
                skip -= 1
            else:
                tuned_path += [point]

    # tuned_path += [filt_path[-1]]

    # filt_path = []
    # prev_point = None
    # for i, point in enumerate(path):
    #     if prev_point is not None:
    #         if point != prev_point:
    #             filt_path += [point]
    #     else:
    #         filt_path += [point]
    #     prev_point = point

    return tuned_path


def get_path(start, end, gate_rad, obs_rad, gates, obstacles, no_hairpin = True):
    # start = (-2, -2)
    # end = (2, 2)
    # gate_rad = 0.08 #0.04 default
    # obs_rad = 0.12 #0.06 default
    obs_dict = get_obs_dict(gates, obstacles, gate_rad = gate_rad, obs_rad = obs_rad)
    fig, ax = plot_circles(obs_dict, plot = False)
    waypoints, goals, goals_angles = generate_waypoints_goal_list(gates, obstacles, start, end)
    fig, ax = plot_points(waypoints, fig = fig, ax = ax, plot = False)

    graph = generate_graph(waypoints, obs_dict, start, end)
    path, length, goal_starts = bfs_multiple_goals(graph, goals, no_hairpin = no_hairpin)
    path = tweak_path(path, goal_starts, goals_angles, scale = 0.1, mid_scale = 0)

    fig, ax = plot_graph(graph, fig = fig, ax = ax, plot = False)
    plot_list(path, fig = fig, ax = ax, plot = True)
    return path

if __name__ == "__main__":
    gates= [  # x, y, z, r, p, y, type 
    # s[-0.5,  1.5, 0, 0, 0, 0,     0] ,
    [ 0.5, -2.5, 0, 0, 0, -1.57, 0],      # gate 1
    [ 0.0,  0.2, 0, 0, 0, 1.57,  0],      # gate 3  
    [-0.5,  1.5, 0, 0, 0, 0,     0],      # gate 4
    [ 2.0, -1.5, 0, 0, 0, 0,     0],      # gate 2
    ]
    obstacles= [  # x, y, z, r, p, y
        [ 1.5, -2.5, 0, 0, 0, 0],             # obstacle 1
        [ 0.5, -1.0, 0, 0, 0, 0],             # obstacle 2
        [ 1.5,    0, 0, 0, 0, 0],             # obstacle 3
        [-1.0,    0, 0, 0, 0, 0]              # obstacle 4
    ]

    start = (-1, -3)
    end = (0.5, 2)

    # gate spacing 0.42
    gate_rad = 0.2
    obs_rad = 0.2

    get_path(start, end, gate_rad, obs_rad, gates, obstacles)