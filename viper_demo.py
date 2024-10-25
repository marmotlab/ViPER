import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Button, Slider
import pickle
from skimage.measure import block_reduce
from skimage import io
from copy import deepcopy
import torch

from test_worker import TestWorker
from env import Env
from agent import Agent
from utils.utils import *
from model import PolicyNet
from test_parameter import model_path, UNBOUND_SPEED, MAX_EPISODE_STEP


class MapEditor:
    def __init__(self, map_size=500, boundary_width=20, map_file_name='map.png', save_dir='maps_spec'):
        # Configuration Parameters
        self.map_size = map_size
        self.boundary_width = boundary_width
        self.map_file_name = map_file_name
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Map Values
        self.free_space_value = 195
        self.obstacle_value = 127
        self.start_position_value = 208

        # Initialize Maps
        self.map_array = np.full((self.map_size, self.map_size), self.free_space_value, dtype=int)
        self._add_boundaries(self.map_array)
        self.temp_map_array = np.copy(self.map_array)

        # Initialize State Variables
        self.play_event = False
        self.drawing_mode = None
        self.line_start = None
        self.reset_state = False
        self.placing_agents = False
        self.robot_cells_user = []
        self.agent_patches = []
        self.agent_colors = ['r', 'b', 'g', 'y', 'm', 'c', 'k', 'w', (1, 0.5, 0.5), (0.2, 0.5, 0.7)]

        # Setup Plot
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.ax.set_title("Draw your map", fontsize=12, fontweight='bold')
        self.ax.axis('off')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.3)

        # UI
        self.map_display = self.ax.imshow(self.map_array, cmap='gray', vmin=0, vmax=255)
        self.cursor_square = self.ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=None, edgecolor='red', visible=False))
        self.tooltip = self.fig.text(0.5, 0.01, '', ha='center', va='bottom', fontsize=10, color='gray', alpha=0)
        self._add_ui_components()
        self._connect_events()

    def _add_boundaries(self, array):
        array[:self.boundary_width, :] = self.obstacle_value
        array[-self.boundary_width:, :] = self.obstacle_value
        array[:, :self.boundary_width] = self.obstacle_value
        array[:, -self.boundary_width:] = self.obstacle_value

    def _add_ui_components(self):
        button_width = 0.15
        button_height = 0.05
        spacing_x = 0.0
        spacing_y = 0.01

        row1_y = 0.2
        row2_y = row1_y - (button_height + spacing_y)

        # Row 1: Obstacle | Free Space | Reset | Random Map
        total_width_row1 = 4 * button_width + 3 * spacing_x
        start_x_row1 = (1 - total_width_row1) / 2

        ax_button_obstacle = plt.axes([start_x_row1, row1_y, button_width, button_height])
        self.button_obstacle = Button(ax_button_obstacle, 'Obstacle')
        self.button_obstacle.on_clicked(lambda _: self.toggle_drawing_mode('obstacle'))

        ax_button_free_space = plt.axes([start_x_row1 + button_width + spacing_x, row1_y, button_width, button_height])
        self.button_free_space = Button(ax_button_free_space, 'Free Space')
        self.button_free_space.on_clicked(lambda _: self.toggle_drawing_mode('free_space'))

        ax_button_reset = plt.axes([start_x_row1 + 2 * (button_width + spacing_x), row1_y, button_width, button_height])
        self.button_reset = Button(ax_button_reset, 'Reset')
        self.button_reset.on_clicked(self.reset_map)

        ax_button_random_map = plt.axes(
            [start_x_row1 + 3 * (button_width + spacing_x), row1_y, button_width, button_height])
        self.button_random_map = Button(ax_button_random_map, 'Random Map')
        self.button_random_map.on_clicked(self.load_random_map)

        # Row 2: Place Agents | Play | Start | Save Map
        total_width_row2 = 4 * button_width + 3 * spacing_x
        start_x_row2 = (1 - total_width_row2) / 2

        ax_button_place_agents = plt.axes([start_x_row2, row2_y, button_width, button_height])
        self.button_place_agents = Button(ax_button_place_agents, 'Place Agents')
        self.button_place_agents.on_clicked(self.toggle_place_agents)

        ax_button_play = plt.axes([start_x_row2 + button_width + spacing_x, row2_y, button_width, button_height])
        self.button_play = Button(ax_button_play, 'Play')
        self.button_play.color = mcolors.CSS4_COLORS['lightgreen']
        self.button_play.hovercolor = mcolors.CSS4_COLORS['palegreen']
        self.button_play.on_clicked(self.play)

        ax_button_start = plt.axes([start_x_row2 + 2 * (button_width + spacing_x), row2_y, button_width, button_height])
        self.button_start = Button(ax_button_start, 'Start Position')
        self.button_start.on_clicked(lambda _: self.toggle_drawing_mode('start'))

        ax_button_save = plt.axes([start_x_row2 + 3 * (button_width + spacing_x), row2_y, button_width, button_height])
        self.button_save = Button(ax_button_save, 'Save Map')
        self.button_save.on_clicked(self.save_map)

        # Slider
        ax_slider_thickness = plt.axes([0.2, 0.09, 0.6, 0.03])
        self.slider_thickness = Slider(ax_slider_thickness, 'Thickness', 5, 120, valinit=50, valstep=1)

        # Cursor Position Text
        self.cursor_pos = self.fig.text(0.43, 0.28, 'Cursor: (0, 0)', fontsize=10, color='gray')

    def _connect_events(self):
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)

    def update_display(self):
        self.map_display.set_data(self.temp_map_array)
        self.fig.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        if self.placing_agents:
            self.place_agent(event)
            return

        if self.drawing_mode is None:
            return

        if self.drawing_mode in ['obstacle', 'free_space', 'start']:
            self.line_start = (int(event.ydata), int(event.xdata))

    def on_release(self, event):
        if event.inaxes != self.ax:
            return

        if self.placing_agents:
            return

        if self.drawing_mode is None or self.line_start is None:
            return

        line_end = (int(event.ydata), int(event.xdata))
        dx = abs(line_end[1] - self.line_start[1])
        dy = abs(line_end[0] - self.line_start[0])

        if self.drawing_mode in ['obstacle', 'free_space']:
            if dx > dy:
                # Make the line horizontal
                line_end = (self.line_start[0], line_end[1])
            else:
                # Make the line vertical
                line_end = (line_end[0], self.line_start[1])

            self.draw_line_on_map(self.line_start, line_end, self.map_array, self.get_drawing_value())
        elif self.drawing_mode == 'start':
            self.draw_start_position(self.line_start)

        self.line_start = None
        self.temp_map_array = np.copy(self.map_array)
        self.update_display()

    def on_move(self, event):
        if event.inaxes == self.ax:
            self.cursor_pos.set_text(f'Cursor: ({int(event.xdata)}, {int(event.ydata)})')
            self.update_cursor_square(event)
            if self.drawing_mode and self.line_start is not None and not self.placing_agents:
                line_end = (int(event.ydata), int(event.xdata))
                temp_array = np.copy(self.map_array)
                self.draw_line_on_map(self.line_start, line_end, temp_array, self.get_drawing_value())
                self.temp_map_array = temp_array
                self.update_display()
            self.fig.canvas.draw_idle()

    def on_hover(self, event):
        if event.inaxes is not None:
            if self.button_obstacle.ax.contains(event)[0]:
                self.show_tooltip("Draw obstacles.")
            elif self.button_free_space.ax.contains(event)[0]:
                self.show_tooltip("Draw free space.")
            elif self.button_reset.ax.contains(event)[0]:
                self.show_tooltip("Clear the canvas, set to obstacle or free space (click again).")
            elif self.button_random_map.ax.contains(event)[0]:
                self.show_tooltip("Load a random map from /maps_test.")
            elif self.button_place_agents.ax.contains(event)[0]:
                self.show_tooltip("Place agents in the free space.")
            elif self.button_play.ax.contains(event)[0]:
                self.show_tooltip("Start ViPER simulation.")
            elif self.button_start.ax.contains(event)[0]:
                self.show_tooltip("Set the start position (if you want to save the map).")
            elif self.button_save.ax.contains(event)[0]:
                self.show_tooltip("Save the current map to /maps_spec/map.png.")
            else:
                self.hide_tooltip()
        else:
            self.hide_tooltip()

    def show_tooltip(self, message):
        self.tooltip.set_text(message)
        self.tooltip.set_alpha(1)
        self.fig.canvas.draw_idle()

    def hide_tooltip(self):
        self.tooltip.set_alpha(0)
        self.fig.canvas.draw_idle()

    def update_cursor_square(self, event):
        if self.drawing_mode == 'obstacle':
            edge_color = 'black'
        elif self.drawing_mode == 'free_space':
            edge_color = 'white'
        elif self.drawing_mode == 'start':
            edge_color = 'gray'
        elif self.placing_agents:
            edge_color = 'red'
        else:
            edge_color = 'black'

        size = self.get_cursor_square_size()
        self.cursor_square.set_width(size)
        self.cursor_square.set_height(size)
        self.cursor_square.set_xy((event.xdata - size / 2 - 1, event.ydata - size / 2 - 1))
        self.cursor_square.set_edgecolor(edge_color)
        self.cursor_square.set_visible(True)

    def get_cursor_square_size(self):
        if self.drawing_mode in ['obstacle', 'free_space']:
            return self.slider_thickness.val
        elif self.drawing_mode == 'start':
            return 25
        elif self.placing_agents:
            return 10
        else:
            return 0

    def get_drawing_value(self):
        if self.drawing_mode == 'obstacle':
            return self.obstacle_value
        elif self.drawing_mode == 'free_space':
            return self.free_space_value
        else:
            return None

    def draw_line_on_map(self, start, end, array, value):
        if value is None:
            return
        thickness = int(self.slider_thickness.val) // 2
        y0, x0 = start
        y1, x1 = end
        num = max(abs(x1 - x0), abs(y1 - y0)) + 1
        x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
        for xi, yi in zip(x, y):
            xi, yi = int(xi), int(yi)
            x_min = max(xi - thickness, 0)
            x_max = min(xi + thickness, self.map_size)
            y_min = max(yi - thickness, 0)
            y_max = min(yi + thickness, self.map_size)
            array[y_min:y_max, x_min:x_max] = value

    def draw_start_position(self, position):
        y, x = position
        self.map_array[max(y - 12, 0):min(y + 13, self.map_size), max(x - 12, 0):min(x + 13, self.map_size)] = self.start_position_value

    def toggle_drawing_mode(self, mode):
        self.drawing_mode = mode
        self.placing_agents = False

    def toggle_place_agents(self, _=None):
        if len(self.robot_cells_user) >= 10:
            return
        self.placing_agents = True
        self.drawing_mode = None

    def place_agent(self, event):
        if len(self.robot_cells_user) >= 10:
            print("Warning: Maximum of 10 agents already placed.")
            return

        x, y = int(event.xdata), int(event.ydata)

        if self.map_array[y, x] != self.free_space_value:
            print(f"Warning: Cell ({x}, {y}) is not free space. Cannot place an agent here.")
            return

        for robot_cell in self.robot_cells_user:
            distance = ((x - robot_cell[0]) ** 2 + (y - robot_cell[1]) ** 2) ** 0.5
            if distance < 20:
                print(f"Warning: Cell ({x}, {y}) is too close to an existing agent at ({robot_cell[0]}, {robot_cell[1]}).")
                return

        color = self.agent_colors[len(self.robot_cells_user) % len(self.agent_colors)]
        agent_circle = plt.Circle((x, y), 10, color=color, fill=True, alpha=0.7)
        self.ax.add_patch(agent_circle)
        self.agent_patches.append(agent_circle)
        self.robot_cells_user.append([x, y])
        self.fig.canvas.draw_idle()

    def load_random_map(self, _=None):
        map_list = os.listdir('maps_test')
        map_path = os.path.join('maps_test', np.random.choice(map_list))
        if os.path.exists(map_path):
            self.map_array = (io.imread(map_path, 1)).astype(int)
            self.map_array[self.map_array == self.start_position_value] = self.free_space_value
            self.temp_map_array = np.copy(self.map_array)
            self.update_display()
        else:
            print(f"Map {map_path} does not exist.")

    def reset_map(self, _=None):
        if self.reset_state:
            self.map_array.fill(self.free_space_value)
        else:
            self.map_array.fill(self.obstacle_value)
        self._add_boundaries(self.map_array)
        self.temp_map_array = np.copy(self.map_array)

        for patch in self.agent_patches:
            patch.remove()
        self.agent_patches.clear()
        self.robot_cells_user.clear()

        self.reset_state = not self.reset_state
        self.update_display()

    def save_map(self, _=None):
        save_path = os.path.join(self.save_dir, self.map_file_name)
        cv2.imwrite(save_path, self.map_array)
        print(f'Map saved to {save_path}/{self.map_file_name}.')

    def play(self, _=None):
        save_path = os.path.join(self.save_dir, 'tmpdata.viper')
        map_data = {'map': self.map_array, 'loc': self.robot_cells_user}
        with open(save_path, 'wb') as file:
            pickle.dump(map_data, file)
        if self.robot_cells_user.__len__() == 0:
            print("Please place at least one agent.")
            return
        self.play_event = True
        plt.close()

    @staticmethod
    def run():
        plt.show()


class InteractiveEnv(Env):
    def __init__(self, episode_index, n_agent, explore, map_array, robot_cells):
        self.map_array = map_array
        self.robot_cells_user = np.array(robot_cells) // 2
        super().__init__(episode_index, n_agent, explore)

    def import_ground_truth(self, episode_index):
        ground_truth = block_reduce(self.map_array, 2, np.min)
        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * 254 + 1
        robot_cell = self.robot_cells_user[0]
        return ground_truth, robot_cell

    def set_initial_location(self):
        robot_locations = []
        for robot_cell_user in self.robot_cells_user:
            robot_loc_user = get_coords_from_cell_position(np.array(robot_cell_user), self.ground_truth_info)
            nearest_index = np.argmin(np.sum((self.ground_truth_coords - robot_loc_user) ** 2, axis=1))
            robot_loc = self.ground_truth_coords[nearest_index]
            robot_locations.append(robot_loc)
        return np.array(robot_locations)


class InteractiveWorker(TestWorker):
    def __init__(self, meta_agent_id, policy_net, global_step):
        super().__init__(meta_agent_id, policy_net, global_step, save_image=True, greedy=True)
        self.save_dir = 'maps_spec'
        self.exploration = True
        self.map_array, self.robot_cells = self.load_map()
        self.env = InteractiveEnv(0, n_agent=len(self.robot_cells), explore=self.exploration,
                                  map_array=self.map_array, robot_cells=self.robot_cells)
        self.robot_list = [Agent(i, policy_net, self.node_manager, self.device, self.save_image) for i in range(self.env.n_agent)]

        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(8, 4))
        self.fig.show()
        self.fig.canvas.draw()

    def load_map(self):
        save_path = os.path.join(self.save_dir, 'tmpdata.viper')
        with open(save_path, 'rb') as file:
            map_data = pickle.load(file)
        return map_data['map'], map_data['loc']

    def run_episode(self):
        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
        for robot in self.robot_list:
            robot.update_safe_graph(self.env.safe_info, self.env.uncovered_safe_frontiers, self.env.counter_safe_info)
        for robot in self.robot_list:
            robot.update_planning_state(self.env.robot_locations)
        if self.save_image:
            self.plot_local_env(-1)

        max_travel_dist = 0

        length_history = [max_travel_dist]
        safe_rate_history = [self.env.safe_rate]
        explored_rate_history = [self.env.explored_rate]

        for i in range(MAX_EPISODE_STEP):
            selected_locations = []
            dist_list = []

            for robot in self.robot_list:
                local_observation = robot.get_observation(pad=False)
                next_location, _, _ = robot.select_next_waypoint(local_observation, self.greedy)
                selected_locations.append(next_location)
                dist_list.append(np.linalg.norm(next_location - robot.location))

            selected_locations = np.array(selected_locations).reshape(-1, 2)
            arriving_sequence = np.argsort(np.array(dist_list))
            selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]

            for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
                solved_locations = selected_locations_in_arriving_sequence[:j]
                while selected_location[0] + selected_location[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                    id = arriving_sequence[j]
                    nearby_nodes = self.robot_list[id].node_manager.local_nodes_dict.nearest_neighbors(selected_location.tolist(), 25)
                    for node in nearby_nodes:
                        coords = node.data.coords
                        if coords[0] + coords[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                            continue
                        selected_location = coords
                        break

                    selected_locations_in_arriving_sequence[j] = selected_location
                    selected_locations[id] = selected_location

            if not UNBOUND_SPEED:
                self.env.decrease_safety(selected_locations)
            else:
                tmp_safe_zone_frontier = copy.deepcopy(self.env.safe_zone_frontiers)
                for _ in range(8):
                    self.env.decrease_safety(selected_locations)
                    self.env.safe_zone_frontiers = get_safe_zone_frontier(self.env.safe_info, self.env.belief_info)
                    if np.array_equal(tmp_safe_zone_frontier, self.env.safe_zone_frontiers):
                        break
                    else:
                        tmp_safe_zone_frontier = copy.deepcopy(self.env.safe_zone_frontiers)

            self.env.step(selected_locations)

            self.env.classify_safe_frontier(selected_locations)

            for robot in self.robot_list:
                robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
            for robot in self.robot_list:
                robot.update_safe_graph(self.env.safe_info, self.env.uncovered_safe_frontiers, self.env.counter_safe_info)
            for robot in self.robot_list:
                robot.update_planning_state(self.env.robot_locations)

            max_travel_dist += np.max(dist_list)

            done = self.env.check_done()

            length_history.append(max_travel_dist)
            safe_rate_history.append(self.env.safe_rate)
            explored_rate_history.append(self.env.explored_rate)

            if self.save_image:
                self.plot_local_env(i)

            if max_travel_dist >= 1000 or i == MAX_EPISODE_STEP - 1:
                print(f"Over 1000 steps or reached maximum steps.")
                break

            if done:
                print(f"Environment cleared in {i + 1} steps.")
                break

        if self.save_image:
            plt.ioff()
            plt.show()


    def plot_local_env(self, step):
        self.ax1.clear()
        self.ax2.clear()

        self.ax2.imshow(self.env.robot_belief, cmap='gray', vmin=0, alpha=1)
        self.ax2.axis('off')
        color_list = ['r', 'b', 'g', 'y', 'm', 'c', 'k', 'w', (1, 0.5, 0.5), (0.2, 0.5, 0.7)]
        robot = self.robot_list[0]
        nodes = get_cell_position_from_coords(robot.local_node_coords, robot.safe_zone_info)
        self.ax2.scatter(nodes[:, 0], nodes[:, 1], c=robot.safe_utility, s=5, zorder=2)

        # for i in range(nodes.shape[0]):
        #     for j in range(i + 1, nodes.shape[0]):
        #         if robot.local_adjacent_matrix[i, j] == 0:
        #             self.ax2.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], c=(0.988, 0.557, 0.675), linewidth=1.5, zorder=1)

        self.ax1.imshow(self.env.robot_belief, cmap='gray', vmin=0)

        self.env.classify_safe_frontier(self.env.robot_locations)
        covered_safe_frontier_cells = get_cell_position_from_coords(self.env.covered_safe_frontiers, self.env.safe_info).reshape(-1, 2)
        uncovered_safe_frontier_cells = get_cell_position_from_coords(self.env.uncovered_safe_frontiers, self.env.safe_info).reshape(-1, 2)

        if covered_safe_frontier_cells.size != 0:
            self.ax1.scatter(covered_safe_frontier_cells[:, 0], covered_safe_frontier_cells[:, 1], c='g', s=1, zorder=6)
        if uncovered_safe_frontier_cells.size != 0:
            self.ax1.scatter(uncovered_safe_frontier_cells[:, 0], uncovered_safe_frontier_cells[:, 1], c='r', s=1, zorder=6)

        n_segments = len(self.robot_list[0].trajectory_x) - 1
        alpha_values = np.linspace(0.3, 1, n_segments)
        for robot in self.robot_list:
            c = color_list[robot.id]
            if robot.id == 0:
                alpha_mask = robot.safe_zone_info.map / 255 / 3
                self.ax1.imshow(robot.safe_zone_info.map, cmap='Greens', alpha=alpha_mask)
                self.ax1.axis('off')

            robot_cell = get_cell_position_from_coords(robot.location, robot.safe_zone_info)
            self.ax1.plot(robot_cell[0], robot_cell[1], c=c, marker='o', markersize=10, zorder=5)

            for i in range(n_segments):
                traj_x = (np.array(robot.trajectory_x[i:i + 2]) - robot.global_map_info.map_origin_x) / robot.cell_size
                traj_y = (np.array(robot.trajectory_y[i:i + 2]) - robot.global_map_info.map_origin_y) / robot.cell_size
                self.ax1.plot(traj_x, traj_y, c=c, linewidth=2, alpha=alpha_values[i], zorder=3)

        self.ax1.axis('off')
        suptitle = 'Explored rate: {:.4g} | Cleared rate: {:.4g} | Trajectory length: {:.4g}'.format(self.env.explored_rate,
                                                                                                     self.env.safe_rate,
                                                                                                     max([robot.travel_dist for robot in self.robot_list]))
        self.fig.suptitle(suptitle)
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05, wspace=0.01)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


if __name__ == "__main__":
    editor = MapEditor()
    editor.run()

    if editor.play_event:
        net = PolicyNet(8, 128)
        ckp = torch.load(f'{model_path}/checkpoint.pth', weights_only=True)
        net.load_state_dict(ckp['policy_model'])
        worker = InteractiveWorker(0, net, 0)
        worker.run_episode()
