import numpy as np
import imageio
import os
import copy
from skimage.morphology import label

from parameter import *


def get_cell_position_from_coords(coords, map_info):
    coords = coords.reshape(-1, 2)
    coords_x = coords[:, 0]
    coords_y = coords[:, 1]
    cell_x = ((coords_x - map_info.map_origin_x) / map_info.cell_size)
    cell_y = ((coords_y - map_info.map_origin_y) / map_info.cell_size)

    cell_position = np.around(np.stack((cell_x, cell_y), axis=-1)).astype(int)
    if cell_position.shape[0] == 1:
        return cell_position[0]
    else:
        return cell_position


def get_coords_from_cell_position(cell_position, map_info):
    cell_position = cell_position.reshape(-1, 2)
    cell_x = cell_position[:, 0]
    cell_y = cell_position[:, 1]
    coords_x = cell_x * map_info.cell_size + map_info.map_origin_x
    coords_y = cell_y * map_info.cell_size + map_info.map_origin_y
    coords = np.stack((coords_x, coords_y), axis=-1)
    coords = np.around(coords, 1)
    if coords.shape[0] == 1:
        return coords[0]
    else:
        return coords


def get_free_area_coords(map_info):
    free_indices = np.where(map_info.map == 255)
    free_cells = np.asarray([free_indices[1], free_indices[0]]).T
    free_coords = get_coords_from_cell_position(free_cells, map_info)
    return free_coords


def get_free_and_connected_map(location, map_info, connected=True):
    # a binary map for free and connected areas
    free = (map_info.map == 255).astype(float)
    labeled_free = label(free, connectivity=2) if connected else free
    cell = get_cell_position_from_coords(location, map_info)
    label_number = labeled_free[cell[1], cell[0]]
    connected_free_map = (labeled_free == label_number)
    return connected_free_map


def get_local_node_coords(location, local_map_info, connected=True):
    x_min = (local_map_info.map_origin_x // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    y_min = (local_map_info.map_origin_y // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    x_max = ((local_map_info.map_origin_x + local_map_info.map.shape[1] * CELL_SIZE) // NODE_RESOLUTION) * NODE_RESOLUTION
    y_max = ((local_map_info.map_origin_y + local_map_info.map.shape[0] * CELL_SIZE) // NODE_RESOLUTION) * NODE_RESOLUTION

    x_coords = np.arange(x_min, x_max, NODE_RESOLUTION)
    y_coords = np.arange(y_min, y_max, NODE_RESOLUTION)
    t1, t2 = np.meshgrid(x_coords, y_coords)
    nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    nodes = np.around(nodes, 1)

    free_connected_map = get_free_and_connected_map(location, local_map_info, connected)

    nodes_cells = get_cell_position_from_coords(nodes, local_map_info)
    nodes_cells_value = free_connected_map[nodes_cells[:, 1], nodes_cells[:, 0]]
    indices = np.where(nodes_cells_value == True)[0]
    nodes = nodes[indices]

    return nodes, free_connected_map


def get_explore_frontier(map_info):
    map_info = copy.deepcopy(map_info)
    x_len = map_info.map.shape[1]
    y_len = map_info.map.shape[0]
    unknown = (map_info.map == 127) * 1
    unknown = np.lib.pad(unknown, ((1, 1), (1, 1)), 'constant', constant_values=0)
    unknown_neighbor = unknown[2:][:, 1:x_len + 1] + unknown[:y_len][:, 1:x_len + 1] + unknown[1:y_len + 1][:, 2:] \
                       + unknown[1:y_len + 1][:, :x_len] + unknown[:y_len][:, 2:] + unknown[2:][:, :x_len] + \
                       unknown[2:][:, 2:] + unknown[:y_len][:, :x_len]
    free_cell_indices = np.where(map_info.map.ravel(order='F') == 255)[0]
    frontier_cell_1 = np.where(1 < unknown_neighbor.ravel(order='F'))[0]
    frontier_cell_2 = np.where(unknown_neighbor.ravel(order='F') < 8)[0]
    frontier_cell_indices = np.intersect1d(frontier_cell_1, frontier_cell_2)
    frontier_cell_indices = np.intersect1d(free_cell_indices, frontier_cell_indices)

    x = np.linspace(0, x_len - 1, x_len)
    y = np.linspace(0, y_len - 1, y_len)
    t1, t2 = np.meshgrid(x, y)
    cells = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    frontier_cell = cells[frontier_cell_indices]

    if frontier_cell.shape[0] > 0:
        frontier_coords = get_coords_from_cell_position(frontier_cell, map_info)
        frontier_coords = frontier_coords.reshape(-1, 2)
        frontier_coords = frontier_down_sample(frontier_coords)
    else:
        frontier_coords = frontier_cell
    return frontier_coords


def get_safe_zone_frontier(safe_info, map_info):
    safe_info = copy.deepcopy(safe_info)
    map_info = copy.deepcopy(map_info)

    x_len = safe_info.map.shape[1]
    y_len = safe_info.map.shape[0]

    safe_free_cell_indices = np.where(safe_info.map.ravel(order='F') > 0)[0]
    safe_zone = (safe_info.map == 255) * 1
    safe_zone = np.lib.pad(safe_zone, ((1, 1), (1, 1)), 'constant', constant_values=0)
    safe_neighbor = safe_zone[2:][:, 1:x_len + 1] + safe_zone[:y_len][:, 1:x_len + 1] + safe_zone[1:y_len + 1][:, 2:] \
                    + safe_zone[1:y_len + 1][:, :x_len] + safe_zone[:y_len][:, 2:] + safe_zone[2:][:, :x_len] + \
                    safe_zone[2:][:, 2:] + safe_zone[:y_len][:, :x_len]

    safe_frontier_cell_1 = np.where(1 < safe_neighbor.ravel(order='F'))[0]
    safe_frontier_cell_2 = np.where(safe_neighbor.ravel(order='F') < 8)[0]
    safe_frontier_cell_indices = np.intersect1d(safe_frontier_cell_1, safe_frontier_cell_2)
    safe_frontier_cell_indices = np.intersect1d(safe_free_cell_indices, safe_frontier_cell_indices)  # border of safe zone

    x_len = map_info.map.shape[1]
    y_len = map_info.map.shape[0]
    obstacle = (map_info.map == 1) * 1
    obstacle = np.lib.pad(obstacle, ((1, 1), (1, 1)), 'constant', constant_values=1)  # add obstacle around extended map
    obstacle_neighbor = obstacle[2:][:, 1:x_len + 1] + obstacle[:y_len][:, 1:x_len + 1] + obstacle[1:y_len + 1][:, 2:] \
                        + obstacle[1:y_len + 1][:, :x_len] + obstacle[:y_len][:, 2:] + obstacle[2:][:, :x_len] + \
                        obstacle[2:][:, 2:] + obstacle[:y_len][:, :x_len]
    obstacle_frontier_cell = np.where(1 <= obstacle_neighbor.ravel(order='F'))[0]  # border of map

    valid_safe_frontier_cell_indices = np.setdiff1d(safe_frontier_cell_indices, obstacle_frontier_cell)  # border of safe zone in free area

    x = np.linspace(0, x_len - 1, x_len)
    y = np.linspace(0, y_len - 1, y_len)
    t1, t2 = np.meshgrid(x, y)
    cells = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    frontier_cell = cells[valid_safe_frontier_cell_indices]

    if frontier_cell.shape[0] > 0:
        frontier_coords = get_coords_from_cell_position(frontier_cell, safe_info)
        frontier_coords = frontier_down_sample(frontier_coords.reshape(-1, 2), safe_info)
    else:
        frontier_coords = frontier_cell
    return frontier_coords.reshape(-1, 2)


def frontier_down_sample(data, map_info=None, voxel_size=FRONTIER_CELL_SIZE):
    voxel_indices = np.array(data / voxel_size, dtype=int).reshape(-1, 2)

    voxel_dict = {}
    for i, point in enumerate(data):
        voxel_index = tuple(voxel_indices[i])

        if voxel_index not in voxel_dict:
            voxel_dict[voxel_index] = point
        else:
            current_point = voxel_dict[voxel_index]
            if np.linalg.norm(point - np.array(voxel_index) * voxel_size) < np.linalg.norm(
                    current_point - np.array(voxel_index) * voxel_size):
                voxel_dict[voxel_index] = point

    downsampled_data = np.array(list(voxel_dict.values()))

    if map_info is not None:
        downsampled_data = push_frontier_to_free(downsampled_data, map_info)

    downsampled_data = remove_isolate_frontiers(downsampled_data, voxel_size)

    return downsampled_data


def push_frontier_to_free(data, map_info):
    downsampled_cells = get_cell_position_from_coords(data, map_info)
    neighbor_cells = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
    for cell in downsampled_cells:
        delta = np.array([0, 0])
        for neighbor_cell in neighbor_cells:
            neighbor = cell + neighbor_cell
            try:
                if map_info.map[neighbor[1], neighbor[0]] == 255:
                    delta += neighbor_cell
            except IndexError:
                pass
        cell += delta
    downsampled_data = get_coords_from_cell_position(downsampled_cells, map_info)
    return downsampled_data


def remove_isolate_frontiers(data, voxel_size):
    try:
        differences = data[:, np.newaxis, :] - data[np.newaxis, :, :]
    except IndexError:
        return data
    distance_to_otherfrontiers = np.sqrt(np.sum(differences ** 2, axis=2))
    np.fill_diagonal(distance_to_otherfrontiers, np.inf)
    min_distance = np.min(distance_to_otherfrontiers, axis=1)
    data = data[min_distance < voxel_size * 2]
    return data


def check_collision(start, end, map_info, max_collision=1):
    # Bresenham line algorithm checking
    collision = False

    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    map = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    collision_flag = 0

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k in [0, 1, 127]:
            collision_flag += 1
            if collision_flag >= max_collision:
                collision = True
                break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return collision


def make_gif(path, n, frame_files, rate):
    with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, rate), mode='I', duration=1) as writer:
        for frame in frame_files:
            image = imageio.imread(frame)
            writer.append_data(image)
    print('gif complete\n')

    # Remove files
    for filename in frame_files[:-1]:
        os.remove(filename)


class Map_info:
    def __init__(self, map, map_origin_x, map_origin_y, cell_size):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
        self.cell_size = cell_size

    def update_map_info(self, map, map_origin_x, map_origin_y):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y

