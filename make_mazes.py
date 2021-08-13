""" make_mazes.py
    Create a datset of mazes using the depth-first algorithm described at
    https://scipython.com/blog/making-a-maze/
    This file includes code borrowed from Christian Hill, April 2017.
    First moodified in July 2020 for DeepThinking.
    Avi Schwarzschild and Arjun Gupta
    July 2021
"""

import collections as col
import heapq
import os
import random

import matplotlib.pyplot as plt
import numpy as np


class Cell:
    """A cell in the maze.
    A maze "Cell" is a point in the grid which may be surrounded by walls to
    the north, east, south or west.
    """

    # A wall separates a pair of cells in the N-S or W-E directions.
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

    def __init__(self, x, y):
        """Initialize the cell at (x, y). At first it is surrounded by walls."""

        self.x, self.y = x, y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}

    def has_all_walls(self):
        """Does this cell still have all its walls?"""

        return all(self.walls.values())

    def knock_down_wall(self, other, wall):
        """Knock down the wall between cells self and other."""

        self.walls[wall] = False
        other.walls[Cell.wall_pairs[wall]] = False


class Maze:
    """A Maze, represented as a grid of cells."""

    def __init__(self, nx, ny, ix=0, iy=0):
        """Initialize the maze grid.
        The maze consists of nx x ny cells and will be constructed starting
        at the cell indexed at (ix, iy).
        """

        self.nx, self.ny = nx, ny
        self.ix, self.iy = ix, iy
        self.maze_map = [[Cell(x, y) for y in range(ny)] for x in range(nx)]

    def cell_at(self, x, y):
        """Return the Cell object at (x, y)."""

        return self.maze_map[x][y]

    def __str__(self):
        """Return a (crude) string representation of the maze."""

        nx = self.nx
        ny = self.ny
        maze_rows = ['-' * nx*2]
        for y in range(ny):
            maze_row = ['|']
            for x in range(nx):
                if self.maze_map[x][y].walls['E']:
                    maze_row.append(' |')
                else:
                    maze_row.append('  ')
            maze_rows.append(''.join(maze_row))
            maze_row = ['|']
            for x in range(nx):
                if self.maze_map[x][y].walls['S']:
                    maze_row.append('-+')
                else:
                    maze_row.append(' +')
            maze_rows.append(''.join(maze_row))
        return '\n'.join(maze_rows)

    def write_np(self):
        """Write an SVG image of the maze to filename."""
        beta = 2
        my_numpy_array = np.zeros((beta * (self.ny) + 1, beta * (self.nx) + 1))
        # Make all the nodes white and make the edges
        # connecting these nodes as white
        # according to the graph
        for x in range(self.nx):
            for y in range(self.ny):
                my_numpy_array[beta * (y) + 1, beta * (x) + 1] = 1
                if not (self.cell_at(x, y).walls['S']):
                    my_numpy_array[beta * (y + 1), beta * (x) + 1] = 1
                if not (self.cell_at(x, y).walls['E']):
                    my_numpy_array[beta * (y) + 1, beta * (x + 1)] = 1

        return np.dstack([my_numpy_array, my_numpy_array, my_numpy_array])

    def find_valid_neighbours(self, cell):
        """Return a list of unvisited neighbours to cell."""

        delta = [('W', (-1, 0)),
                 ('E', (1, 0)),
                 ('S', (0, 1)),
                 ('N', (0, -1))]
        neighbours = []
        for direction, (dx, dy) in delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.cell_at(x2, y2)
                if neighbour.has_all_walls():
                    neighbours.append((direction, neighbour))
        return neighbours

    def make_maze(self):
        """make a single maze"""
        # Total number of cells.
        n = self.nx * self.ny
        cell_stack = []
        current_cell = self.cell_at(self.ix, self.iy)
        # Total number of visited cells during maze construction.
        nv = 1

        while nv < n:
            neighbours = self.find_valid_neighbours(current_cell)

            if not neighbours:
                # We've reached a dead end: backtrack.
                current_cell = cell_stack.pop()
                continue

            # Choose a random neighbouring cell and move to it.
            direction, next_cell = random.choice(neighbours)
            current_cell.knock_down_wall(next_cell, direction)
            cell_stack.append(current_cell)
            current_cell = next_cell
            nv += 1


def create_maze(n=20, ix=0, iy=0):
    """Returns numpy array of maze with n columns/rows and a starting point of (0,0)."""
    maze = Maze(n, n, ix, iy)
    maze.make_maze()
    arr = maze.write_np()
    return arr


class Node:
    """A node in the graph"""
    def __init__(self, x_coord, y_coord, cost, parentID):

        self.x = x_coord
        self.y = y_coord
        self.cost = cost
        self.parentID = parentID

    def __lt__(self, other):
        return self.cost < other.cost


def possible_steps():
    """get possible steps"""
    steps_with_cost = np.array([[0, 1, 1],              # Move_up
                                [1, 0, 1],              # Move_right
                                [0, -1, 1],             # Move_down
                                [-1, 0, 1],             # Move_left
                                ])
    return steps_with_cost


def is_valid(point_x, point_y, grid, width, height):
    """see if a point is valid"""
    if not grid[int(point_y)][int(point_x)]:
        return False
    if point_y < 0 or point_x < 0:
        return False
    if point_y > height or point_x > width:
        return False
    return True


def is_goal(current, goal):
    """see if we are at the goal"""
    return (current.x == goal.x) and (current.y == goal.y)


def path_search_algo(start_node, end_node, grid, width, height):
    """path search function"""
    current_node = start_node
    goal_node = end_node
    steps_with_cost = possible_steps()

    if is_goal(current_node, goal_node):
        return 1

    open_nodes = {}
    open_nodes[start_node.x * width + start_node.y] = start_node
    closed_nodes = {}
    cost = []
    all_nodes = []
    heapq.heappush(cost, [start_node.cost, start_node])

    while len(cost) != 0:

        current_node = heapq.heappop(cost)[1]
        all_nodes.append([current_node.x, current_node.y])
        current_id = current_node.x * width + current_node.y

        if is_goal(current_node, end_node):
            end_node.parentID = current_node.parentID
            end_node.cost = current_node.cost
            return 1, all_nodes

        if current_id in closed_nodes:
            continue
        else:
            closed_nodes[current_id] = current_node

        del open_nodes[current_id]

        for i in range(steps_with_cost.shape[0]):

            new_node = Node(current_node.x + steps_with_cost[i][0],
                            current_node.y + steps_with_cost[i][1],
                            current_node.cost + steps_with_cost[i][2],
                            current_node)

            new_node_id = new_node.x*width + new_node.y

            if not is_valid(new_node.x, new_node.y, grid, width, height):
                continue
            elif new_node_id in closed_nodes:
                continue

            if new_node_id in open_nodes:
                if new_node.cost < open_nodes[new_node_id].cost:
                    open_nodes[new_node_id].cost = new_node.cost
                    open_nodes[new_node_id].parentID = new_node.parentID
            else:
                open_nodes[new_node_id] = new_node

            heapq.heappush(cost, [open_nodes[new_node_id].cost, open_nodes[new_node_id]])

    return 0, all_nodes


def find_path(end_node):
    """Function to find path to the end node"""
    x_coord = [end_node.x]
    y_coord = [end_node.y]

    node_id = end_node.parentID
    while node_id != -1:
        # current_node = id.parentID
        x_coord.append(node_id.x)
        y_coord.append(node_id.y)
        node_id = node_id.parentID

    x_coord.reverse()
    y_coord.reverse()
    coords = np.vstack((x_coord, y_coord))
    return coords


def gen_sample(num, ix, iy, ex, ey):
    """Returns a numpy array corresponding to an nxn maze and the length
     of the shortest path from (ix,iy) to (ex,ey)"""
    maze = create_maze(num, ix, iy)
    start_node = Node(2 * ix + 1, 2 * iy + 1, 0.0, -1)
    end_node = Node(2 * ex + 1, 2 * ey + 1, 0.0, -1)
    path_search_algo(start_node, end_node, maze[:, :, 0], maze.shape[0], maze.shape[1])
    coords = find_path(end_node)
    solution = np.zeros((maze.shape[0], maze.shape[1], 1))
    solution[coords[1, :], coords[0, :]] = 1
    return maze, len(coords[0]) - 1, solution


def get_final_maze_array(arr, ix, iy, ex, ey):
    """Add the start and end points to a maze and cast as uint8"""
    maze_array = arr.copy()
    maze_array[2 * iy + 1, 2 * ix + 1, :] = [0, 1, 0]
    maze_array[2 * ey + 1, 2 * ex + 1, :] = [1, 0, 0]
    return maze_array


def gen_dataset(num_images=60000, maze_size=7):
    """Function to generate a whole dataset"""
    num_images = int(num_images)
    data_array = np.zeros((num_images, 2 * maze_size + 1, 2 * maze_size + 1, 3))
    targets_array = np.zeros(num_images)
    solution_array = np.zeros((num_images, 2 * maze_size + 1, 2 * maze_size + 1, 1))
    start_and_end_array = np.zeros((num_images, 4))
    x_points, y_points = np.meshgrid(np.arange(0, maze_size), np.arange(0, maze_size))
    x_points = x_points.flatten()
    y_points = y_points.flatten()
    for j in range(num_images):
        start, end = np.random.choice(maze_size ** 2, 2, replace=False)
        ix = x_points[start]
        iy = y_points[start]
        ex = x_points[end]
        ey = y_points[end]
        maze, length, solution = gen_sample(maze_size, ix, iy, ex, ey)
        maze_array = get_final_maze_array(maze, ix, iy, ex, ey)
        data_array[j] = maze_array
        targets_array[j] = length
        solution_array[j] = solution
        start_and_end_array[j] = [ix, iy, ex, ey]
        if j % 5000 == 0:
            print(f"Done making {j} mazes.")

    img_size = 2 * (2 * (maze_size) + 1) + 2
    border = (img_size - 4 * maze_size) // 2

    big_data_array = np.zeros((num_images, img_size, img_size, 3))
    big_data_array[:, border-1:-border:2, border-1:-border:2, :] = data_array
    big_data_array[:, border:-border+1:2, border:-border+1:2, :] = data_array
    big_data_array[:, border:-border+1:2, border-1:-border:2, :] = data_array
    big_data_array[:, border-1:-border:2, border:-border+1:2, :] = data_array

    big_solution_array = np.zeros((num_images, img_size, img_size, 3))
    big_solution_array[:, border-1:-border:2, border-1:-border:2, :] = solution_array
    big_solution_array[:, border:-border+1:2, border:-border+1:2, :] = solution_array
    big_solution_array[:, border:-border+1:2, border-1:-border:2, :] = solution_array
    big_solution_array[:, border-1:-border:2, border:-border+1:2, :] = solution_array
    return big_data_array, targets_array, start_and_end_array, big_solution_array


if __name__ == "__main__":
    num_mazes = 5
    size = 9
    # for size in range(9, 18, 2):
    data_name = f"data/maze_data_train_{size}"
    inputs, targets, start_and_end, solutions = gen_dataset(num_mazes, (size+1) // 2)
    # unique, frequency = np.unique(targets, return_counts=True)
    # fig, ax = plt.subplots()
    # ax.hist(targets, bins=len(unique))
    # plt.savefigig(f"historgram_of_labels{data_name}.pdf")
    if not os.path.isdir(f"{data_name}"):
        os.makedirs(f"{data_name}")
    inputs, solutions = inputs.transpose((0, 3, 1, 2)), solutions.transpose((0, 3, 1, 2))[:, 0]
    print(f"Mazes of size {size}, inputs.shape = {inputs.shape}, targets.shape = {solutions.shape}")
    np.save(os.path.join(data_name, "inputs.npy"), inputs)
    np.save(os.path.join(data_name, "solutions.npy"), solutions)

    # Check for repeats
    t_dict = {}
    t_dict = col.defaultdict(lambda: 0)     # t_dict = {*:0}
    for t in inputs:
        t_dict[t.tobytes()] += 1            # t_dict[input] += 1

    repeats = 0
    for i in inputs:
        if t_dict[i.tobytes()] > 1:
            repeats += 1

    print(f"Maze size: {size} \n There are {repeats} mazes repeated in the dataset. ({repeats/num_mazes*100} %)")
