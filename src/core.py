"""
Core pathfinding algorithms and maze generation

Author: gumocimo
Date: 20/08/2025
"""

import heapq
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


# ========== Node and State Classes ==========
class Node:
    """Represents a cell in the grid"""

    def __init__(self, *coords):
        """
        Initialize a node with coordinates.
        Supports both 2D (x, y) and 3D (x, y, z) coordinates.
        """
        if len(coords) == 2:
            self.x, self.y = coords
            self.z = None
            self.coords = (self.x, self.y)
        elif len(coords) == 3:
            self.x, self.y, self.z = coords
            self.coords = (self.x, self.y, self.z)
        else:
            raise ValueError("Node requires 2 or 3 coordinates")

        self.g = float('inf') # Cost from start
        self.h = 0 # Heuristic cost to goal
        self.f = float('inf') # Total cost (g + h)
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.coords == other.coords

    def __hash__(self):
        return hash(self.coords)


@dataclass
class AlgorithmState:
    """Captures the state of the algorithm at a specific step"""
    explored_nodes: List[Tuple[int, ...]]
    frontier: Dict[Tuple[int, ...], Tuple[float, float, float]] # coords -> (g, h, f)
    current_node: Optional[Tuple[int, ...]] = None


# ========== A* Pathfinding Algorithm ==========
class AStar:
    """A* pathfinding algorithm for 2D and 3D grids"""

    def __init__(self, grid):
        """
        Initialize A* pathfinder
        grid: 2D or 3D list where 0 = walkable, 1 = obstacle
        """
        self.grid = grid
        self._determine_dimensions()

    def _determine_dimensions(self):
        """Determine if grid is 2D or 3D and set dimensions"""
        if isinstance(self.grid[0][0], list):
            # 3D grid
            self.is_3d = True
            self.depth = len(self.grid)
            self.rows = len(self.grid[0]) if self.depth > 0 else 0
            self.cols = len(self.grid[0][0]) if self.rows > 0 else 0
        else:
            # 2D grid
            self.is_3d = False
            self.rows = len(self.grid)
            self.cols = len(self.grid[0]) if self.rows > 0 else 0
            self.depth = None

    def heuristic(self, node1, node2):
        """Manhattan distance heuristic"""
        if self.is_3d:
            return abs(node1.x - node2.x) + abs(node1.y - node2.y) + abs(node1.z - node2.z)
        else:
            return abs(node1.x - node2.x) + abs(node1.y - node2.y)

    def get_neighbors(self, node):
        """Get valid neighboring nodes"""
        neighbors = []

        if self.is_3d:
            # 6-directional movement for 3D
            directions = [
                (1, 0, 0), (-1, 0, 0),
                (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1)
            ]

            for dx, dy, dz in directions:
                nx, ny, nz = node.x + dx, node.y + dy, node.z + dz

                if (0 <= nx < self.depth and
                        0 <= ny < self.rows and
                        0 <= nz < self.cols and
                        self.grid[nx][ny][nz] == 0):
                    neighbors.append(Node(nx, ny, nz))
        else:
            # 4-directional movement for 2D
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

            for dx, dy in directions:
                nx, ny = node.x + dx, node.y + dy

                if (0 <= nx < self.rows and
                        0 <= ny < self.cols and
                        self.grid[nx][ny] == 0):
                    neighbors.append(Node(nx, ny))

        return neighbors

    def reconstruct_path(self, node):
        """Reconstruct path from start to goal"""
        path = []
        current = node
        while current:
            path.append(current.coords)
            current = current.parent
        return path[::-1] # Reverse to get path from start to goal

    def find_path(self, start, goal):
        """
        Find shortest path from start to goal using A*
        start: tuple (x, y) or (x, y, z)
        goal: tuple (x, y) or (x, y, z)
        Returns: list of tuples representing the path
        """
        path, _, _, _ = self.find_path_with_stats(start, goal)
        return path

    def find_path_with_stats(self, start, goal):
        """
        Find shortest path from start to goal using A* and return statistics
        start: tuple (x, y) or (x, y, z)
        goal: tuple (x, y) or (x, y, z)
        Returns: (path, explored_nodes, max_frontier_size, nodes_evaluated)
        """
        path, _, _, max_frontier_size, nodes_evaluated = self.find_path_with_animation_data(start, goal)
        # Extract just the explored nodes for backward compatibility
        explored_nodes = []
        if path is not None:
            # Get all explored nodes from animation states
            astar_temp = AStar(self.grid)
            _, animation_states, _, _ = astar_temp.find_path_with_animation_data(start, goal)
            for state in animation_states:
                explored_nodes.extend(state.explored_nodes)
            # Remove duplicates while preserving order
            seen = set()
            explored_nodes = [x for x in explored_nodes if not (x in seen or seen.add(x))]
        return path, explored_nodes, max_frontier_size, nodes_evaluated

    def find_path_with_animation_data(self, start, goal):
        """
        Find shortest path from start to goal using A* and return animation data
        Returns: (path, animation_states, max_frontier_size, nodes_evaluated)
        """
        # Create start and goal nodes
        start_node = Node(*start)
        goal_node = Node(*goal)

        # Check if start and goal are valid
        if self.is_3d:
            if (self.grid[start[0]][start[1]][start[2]] == 1 or
                    self.grid[goal[0]][goal[1]][goal[2]] == 1):
                return None, [], 0, 0
        else:
            if (self.grid[start[0]][start[1]] == 1 or
                    self.grid[goal[0]][goal[1]] == 1):
                return None, [], 0, 0

        # Initialize start node
        start_node.g = 0
        start_node.h = self.heuristic(start_node, goal_node)
        start_node.f = start_node.g + start_node.h

        # Open set (nodes to be evaluated)
        open_set = []
        heapq.heappush(open_set, start_node)
        open_set_dict = {start: start_node} # For quick lookup

        # Closed set (nodes already evaluated)
        closed_set = set()

        # Keep track of all nodes
        all_nodes = {start: start_node}

        # Animation states
        animation_states = []

        # Statistics
        max_frontier_size = 1
        nodes_evaluated = 0

        while open_set:
            # Capture state BEFORE popping the current node
            frontier_data = {}
            for coords, node in open_set_dict.items():
                if coords not in closed_set:
                    frontier_data[coords] = (node.g, node.h, node.f)

            # Get the node with lowest f value
            current = heapq.heappop(open_set)
            current_coords = current.coords
            del open_set_dict[current_coords]
            nodes_evaluated += 1

            # Capture current state with the node about to be evaluated
            current_state = AlgorithmState(
                explored_nodes=list(closed_set),
                frontier=frontier_data.copy(),
                current_node=current_coords
            )
            animation_states.append(current_state)

            # Check if we reached the goal
            if current == goal_node:
                path = self.reconstruct_path(current)
                return path, animation_states, max_frontier_size, nodes_evaluated

            # Mark as explored
            closed_set.add(current_coords)

            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                neighbor_tuple = neighbor.coords

                # Skip if already evaluated
                if neighbor_tuple in closed_set:
                    continue

                # Calculate tentative g score
                tentative_g = current.g + 1 # Cost to move to neighbor is 1

                # Get or create neighbor node
                if neighbor_tuple not in all_nodes:
                    all_nodes[neighbor_tuple] = neighbor
                neighbor = all_nodes[neighbor_tuple]

                # Update neighbor if we found a better path
                if tentative_g < neighbor.g:
                    neighbor.parent = current
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor, goal_node)
                    neighbor.f = neighbor.g + neighbor.h

                    # Add to open set if not already there
                    if neighbor_tuple not in open_set_dict:
                        heapq.heappush(open_set, neighbor)
                        open_set_dict[neighbor_tuple] = neighbor

            # Update max frontier size
            max_frontier_size = max(max_frontier_size, len(open_set))

        return None, animation_states, max_frontier_size, nodes_evaluated


# ========== Maze Generation ==========
class MazeGenerator:
    """Generate random mazes using recursive backtracking (2D) or sparse obstacles (3D)"""

    def __init__(self, width, height, depth=None):
        """
        Initialize maze generator
        width: maze width (should be odd for proper 2D generation)
        height: maze height (should be odd for proper 2D generation)
        depth: maze depth for 3D (None for 2D)
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.is_3d = depth is not None

        if self.is_3d:
            # Initialize 3D grid with all empty (0s)
            self.grid = [[[0 for _ in range(width)] for _ in range(height)] for _ in range(depth)]
        else:
            # Initialize 2D grid with all walls (1s)
            self.grid = [[1 for _ in range(width)] for _ in range(height)]

    def generate(self):
        """Generate a random maze"""
        if self.is_3d:
            return self._generate_3d()
        else:
            return self._generate_2d()

    def _generate_2d(self):
        """Generate a 2D maze using recursive backtracking"""
        # Start from a random cell
        start_x = random.randint(0, self.height-1)
        start_y = random.randint(0, self.width-1)

        # Carve out the starting cell
        self.grid[start_x][start_y] = 0

        # Stack for backtracking
        stack = [(start_x, start_y)]

        while stack:
            current_x, current_y = stack[-1]

            # Get unvisited neighbors
            neighbors = self._get_unvisited_neighbors_2d(current_x, current_y)

            if neighbors:
                # Choose a random neighbor
                next_x, next_y = random.choice(neighbors)

                # Remove wall between current and chosen neighbor
                wall_x = (current_x + next_x)//2
                wall_y = (current_y + next_y)//2
                self.grid[wall_x][wall_y] = 0
                self.grid[next_x][next_y] = 0

                # Add neighbor to stack
                stack.append((next_x, next_y))
            else:
                # Backtrack
                stack.pop()

        # Ensure start and goal areas are clear
        self._clear_area_2d(0, 0)
        self._clear_area_2d(self.height-1, self.width-1)

        return self.grid

    def _generate_3d(self):
        """Generate a simple 3D maze with sparse obstacles"""
        # Start with all empty space (done in __init__)

        # Add some pillar obstacles (vertical columns)
        num_pillars = min(8, (self.height * self.width)//20)
        for _ in range(num_pillars):
            x = random.randint(2, self.depth-3)
            y = random.randint(2, self.height-3)
            # Create a pillar through multiple z levels
            pillar_height = random.randint(3, min(7, self.width-2))
            start_z = random.randint(0, self.width - pillar_height)
            for z in range(start_z, start_z + pillar_height):
                self.grid[x][y][z] = 1

        # Add some floating blocks
        num_blocks = min(10, (self.depth * self.height * self.width)//100)
        for _ in range(num_blocks):
            x = random.randint(1, self.depth-2)
            y = random.randint(1, self.height-2)
            z = random.randint(1, self.width-2)
            # Create a small 2x2x2 or 3x3x3 block
            block_size = random.choice([2, 3])
            for dx in range(block_size):
                for dy in range(block_size):
                    for dz in range(block_size):
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if (0 < nx < self.depth-1 and
                                0 < ny < self.height-1 and
                                0 < nz < self.width-1):
                            self.grid[nx][ny][nz] = 1

        # Add some wall segments
        num_walls = min(6, (self.height + self.width)//4)
        for _ in range(num_walls):
            if random.random() < 0.5:
                # Horizontal wall (along x)
                y = random.randint(1, self.height-2)
                z = random.randint(1, self.width-2)
                length = random.randint(3, self.depth//2)
                start_x = random.randint(0, self.depth - length)
                for x in range(start_x, start_x + length):
                    self.grid[x][y][z] = 1
            else:
                # Horizontal wall (along y)
                x = random.randint(1, self.depth-2)
                z = random.randint(1, self.width-2)
                length = random.randint(3, self.height//2)
                start_y = random.randint(0, self.height - length)
                for y in range(start_y, start_y + length):
                    self.grid[x][y][z] = 1

        # Ensure start and goal areas are clear
        self._clear_area_3d(0, 0, 0, radius=1)
        self._clear_area_3d(self.depth-1, self.height-1, self.width-1, radius=1)

        return self.grid

    def _get_unvisited_neighbors_2d(self, x, y):
        """Get unvisited neighbors that are 2 cells away"""
        neighbors = []
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check if neighbor is within bounds and unvisited (still a wall)
            if 0 <= nx < self.height and 0 <= ny < self.width and self.grid[nx][ny] == 1:
                neighbors.append((nx, ny))

        return neighbors

    def _clear_area_2d(self, x, y, radius=1):
        """Clear a small area around a point in 2D"""
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.height and 0 <= ny < self.width:
                    self.grid[nx][ny] = 0

    def _clear_area_3d(self, x, y, z, radius=1):
        """Clear a small area around a point in 3D"""
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                for dz in range(-radius, radius+1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if (0 <= nx < self.depth and
                            0 <= ny < self.height and
                            0 <= nz < self.width):
                        self.grid[nx][ny][nz] = 0

    def add_random_paths(self, percentage=0.1):
        """Add some random paths to make the maze less perfect"""
        if self.is_3d:
            cells_to_clear = int(self.depth * self.height * self.width * percentage)

            for _ in range(cells_to_clear):
                x = random.randint(1, self.depth-2)
                y = random.randint(1, self.height-2)
                z = random.randint(1, self.width-2)
                self.grid[x][y][z] = 0
        else:
            cells_to_clear = int(self.height * self.width * percentage)

            for _ in range(cells_to_clear):
                x = random.randint(1, self.height-2)
                y = random.randint(1, self.width-2)
                self.grid[x][y] = 0

    @staticmethod
    def create_sample_maze():
        """Create a sample 2D maze for testing"""
        return [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]
