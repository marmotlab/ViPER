import matplotlib.pyplot as plt
from copy import deepcopy
from env import Env
from agent import Agent
from model import PolicyNet
from utils.utils import *
from utils.node_manager_quadtree import NodeManager

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class Multi_agent_worker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, explore=EXPLORATION, plot=self.save_image)
        self.n_agent = N_AGENTS
        self.node_manager = NodeManager(self.env.ground_truth_coords, self.env.ground_truth_info, explore=EXPLORATION, plot=self.save_image)

        self.robot_list = [Agent(i, policy_net, self.node_manager, self.device, self.save_image) for i in range(self.n_agent)]

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(24):
            self.episode_buffer.append([])

    def run_episode(self):
        done = False
        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
        for robot in self.robot_list:
            robot.update_safe_graph(self.env.safe_info, self.env.uncovered_safe_frontiers, self.env.counter_safe_info)
        for robot in self.robot_list:
            robot.update_planning_state(self.env.robot_locations)
            robot.update_underlying_state()

        safe_increase_log = []
        max_travel_dist = 0
        for i in range(MAX_EPISODE_STEP):
            selected_locations = []
            dist_list = []
            next_node_index_list = []
            for robot in self.robot_list:
                observation = robot.get_observation()
                state = robot.get_state()
                robot.save_observation(observation)
                robot.save_state(state)

                next_location, next_node_index, action_index = robot.select_next_waypoint(observation)
                robot.save_action(action_index)

                selected_locations.append(next_location)
                dist_list.append(np.linalg.norm(next_location - robot.location))
                next_node_index_list.append(next_node_index)

            selected_locations = self.solve_path_confict(selected_locations, dist_list)

            curr_node_indices = np.array([robot.current_local_index for robot in self.robot_list])

            self.env.decrease_safety(selected_locations)

            self.env.step(selected_locations)

            self.env.classify_safe_frontier(selected_locations)

            for robot in self.robot_list:
                robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
            for robot in self.robot_list:
                robot.update_safe_graph(self.env.safe_info, self.env.uncovered_safe_frontiers, self.env.counter_safe_info)

            done = self.env.check_done()

            indiv_reward, safety_increase = self.env.calculate_reward(dist_list)

            max_travel_dist += np.max(dist_list)
            if safety_increase > 0:
                safe_increase_log.append(1)
            else:
                safe_increase_log.append(0)

            for robot, reward in zip(self.robot_list, indiv_reward):
                robot.save_all_indices(np.array(curr_node_indices))
                robot.save_reward(reward)
                robot.save_done(done)
                robot.update_planning_state(self.env.robot_locations)
                robot.update_underlying_state()

            if self.save_image:
                self.plot_local_env(i)

            if done:
                break

        # save metrics
        self.perf_metrics['travel_dist'] = max([robot.travel_dist for robot in self.robot_list])
        self.perf_metrics['max_travel_dist'] = max_travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['safe_rate'] = self.env.safe_rate
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['safe_increase_rate'] = np.mean(safe_increase_log)

        # save episode buffer
        for robot in self.robot_list:
            observation = robot.get_observation()
            state = robot.get_state()
            robot.save_next_observations(observation, next_node_index_list)
            robot.save_next_state(state)

            for i in range(len(self.episode_buffer)):
                self.episode_buffer[i] += robot.episode_buffer[i]

        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.safe_rate)

    def solve_path_confict(self, selected_locations, dist_list):
        selected_locations = np.array(selected_locations).reshape(-1, 2)
        arriving_sequence = np.argsort(np.array(dist_list))
        selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]

        for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
            solved_locations = selected_locations_in_arriving_sequence[:j]
            while selected_location[0] + selected_location[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                id = arriving_sequence[j]
                nearby_nodes = self.robot_list[id].node_manager.local_nodes_dict.nearest_neighbors(
                    selected_location.tolist(), 25)
                for node in nearby_nodes:
                    coords = node.data.coords
                    if coords[0] + coords[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                        continue
                    selected_location = coords
                    break

                selected_locations_in_arriving_sequence[j] = selected_location
                selected_locations[id] = selected_location

        return selected_locations

    def plot_local_env(self, step):
        plt.switch_backend('agg')
        plt.figure(figsize=(9, 4))
        plt.subplot(1, 2, 2)
        plt.imshow(self.env.robot_belief, cmap='gray', vmin=0)
        plt.axis('off')
        color_list = ['r', 'b', 'g', 'y', 'm', 'c', 'k', 'w', (1,0.5,0.5), (0.2,0.5,0.7)]
        robot = self.robot_list[0]
        nodes = get_cell_position_from_coords(robot.local_node_coords, robot.safe_zone_info)
        plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.safe_utility, s=5, zorder=2)
        for i in range(nodes.shape[0]):
            for j in range(i + 1, nodes.shape[0]):
                if robot.local_adjacent_matrix[i, j] == 0:
                    plt.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], c=(0.988, 0.557, 0.675), linewidth=1.5, zorder=1)

        plt.subplot(1, 2, 1)
        plt.imshow(self.env.robot_belief, cmap='gray')

        self.env.classify_safe_frontier(self.env.robot_locations)
        covered_safe_frontier_cells = get_cell_position_from_coords(self.env.covered_safe_frontiers, self.env.safe_info).reshape(-1, 2)
        uncovered_safe_frontier_cells = get_cell_position_from_coords(self.env.uncovered_safe_frontiers, self.env.safe_info).reshape(-1, 2)
        if covered_safe_frontier_cells.shape[0] != 0:
            plt.scatter(covered_safe_frontier_cells[:, 0], covered_safe_frontier_cells[:, 1], c='g', s=1, zorder=6)
        if uncovered_safe_frontier_cells.shape[0] != 0:
            plt.scatter(uncovered_safe_frontier_cells[:, 0], uncovered_safe_frontier_cells[:, 1], c='r', s=1, zorder=6)

        n_segments = len(self.robot_list[0].trajectory_x) - 1
        alpha_values = np.linspace(0.3, 1, n_segments)
        for robot in self.robot_list:
            c = color_list[robot.id]
            if robot.id == 0:
                alpha_mask = robot.safe_zone_info.map / 255 / 3
                plt.imshow(robot.safe_zone_info.map, cmap='Greens', alpha=alpha_mask)
                plt.axis('off')

            robot_cell = get_cell_position_from_coords(robot.location, robot.safe_zone_info)
            plt.plot(robot_cell[0], robot_cell[1], c=c, marker='o', markersize=10, zorder=5)

            for i in range(n_segments):
                plt.plot((np.array(robot.trajectory_x[i:i + 2]) - robot.global_map_info.map_origin_x) / robot.cell_size,
                         (np.array(robot.trajectory_y[i:i + 2]) - robot.global_map_info.map_origin_y) / robot.cell_size,
                         c,
                         linewidth=2, alpha=alpha_values[i], zorder=3)

        plt.axis('off')
        plt.suptitle('Explored rate: {:.4g} | Cleared rate: {:.4g} | Trajectory length: {:.4g}'.format(self.env.explored_rate,
                                                                                                self.env.safe_rate,
                                                                                                max([robot.travel_dist for robot in self.robot_list])))
        plt.tight_layout()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step), dpi=150)
        plt.close()
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step)
        self.env.frame_files.append(frame)


if __name__ == '__main__':
    from parameter import *
    policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
    # ckp = torch.load('model/viper/checkpoint.pth', map_location='cpu')
    # policy_net.load_state_dict(ckp['policy_model'])
    worker = Multi_agent_worker(0, policy_net, 0, 'cpu', False)
    worker.run_episode()
