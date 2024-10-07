import numpy as np


def collision_check(x0, y0, x1, y1, ground_truth, robot_belief):
    x0 = x0.round()
    y0 = y0.round()
    x1 = x1.round()
    y1 = y1.round()
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    collision_flag = 0
    max_collision = 3

    while 0 <= x < ground_truth.shape[1] and 0 <= y < ground_truth.shape[0]:
        k = ground_truth.item(y, x)
        if k == 1 and collision_flag < max_collision:
            collision_flag += 1
            if collision_flag >= max_collision:
                break

        if k != 1 and collision_flag > 0:
            break

        if x == x1 and y == y1:
            break

        robot_belief.itemset((y, x), k)

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return robot_belief


def safety_check(x0, y0, x1, y1, sub_safe_zone, sub_belief, sub_intersection):
    x1 = x1.round()
    y1 = y1.round()
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < sub_safe_zone.shape[1] and 0 <= y < sub_safe_zone.shape[0]:
        k1 = sub_safe_zone.item(y, x)
        k2 = sub_belief.item(y, x)
        k3 = sub_intersection.item(y, x) if sub_intersection is not None else 0

        if k2 == 1:
            break

        if k3 == 255:
            break

        if x == x1 and y == y1:
            break

        sub_safe_zone.itemset((y, x), 0)

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return sub_safe_zone


def update_coverage(x0, y0, x1, y1, ground_truth, safe_zone):
    x0 = x0.round()
    y0 = y0.round()
    x1 = x1.round()
    y1 = y1.round()
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    collision_flag = 0
    max_collision = 1

    while 0 <= x < ground_truth.shape[1] and 0 <= y < ground_truth.shape[0]:
        k = ground_truth.item(y, x)
        if k == 1 and collision_flag < max_collision:
            collision_flag += 1
            if collision_flag >= max_collision:
                break

        if k != 1 and collision_flag > 0:
            break

        if x == x1 and y == y1:
            break

        safe_zone.itemset((y, x), 255)

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return safe_zone


def exploration_sensor(robot_position, sensor_range, robot_belief, ground_truth):
    sensor_angle_inc = 0.5 / 180 * np.pi
    sensor_angle = 0
    x0 = robot_position[0]
    y0 = robot_position[1]
    while sensor_angle < 2 * np.pi:
        x1 = x0 + np.cos(sensor_angle) * sensor_range
        y1 = y0 + np.sin(sensor_angle) * sensor_range
        robot_belief = collision_check(x0, y0, x1, y1, ground_truth, robot_belief)
        sensor_angle += sensor_angle_inc
    return robot_belief


def coverage_sensor(robot_position, sensor_range, robot_belief, ground_truth):
    sensor_angle_inc = 0.5 / 180 * np.pi
    sensor_angle = 0
    x0 = robot_position[0]
    y0 = robot_position[1]
    while sensor_angle < 2 * np.pi:
        x1 = x0 + np.cos(sensor_angle) * sensor_range
        y1 = y0 + np.sin(sensor_angle) * sensor_range
        robot_belief = update_coverage(x0, y0, x1, y1, ground_truth, robot_belief)
        sensor_angle += sensor_angle_inc
    return robot_belief


def decrease_safety_by_frontier(center, safety_range, sub_safe_zone, sub_belief, sub_intersection=None):
    angle_inc = 0.5 / 180 * np.pi
    angle = 0
    x0 = center[0]
    y0 = center[1]
    while angle < 2 * np.pi:
        x1 = x0 + np.cos(angle) * safety_range
        y1 = y0 + np.sin(angle) * safety_range
        sub_safe_zone[:, :] = safety_check(x0, y0, x1, y1, sub_safe_zone, sub_belief, sub_intersection)
        angle += angle_inc
    return sub_safe_zone
