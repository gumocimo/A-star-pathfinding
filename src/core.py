"""
Core A* pathfinding algorithm implementation

Author: gumocimo
Date: 14/08/2025
"""

import heapq
import random


class Node:
    """Represents a cell in the grid"""

    def __init__(self, *coords):
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
        self.f = float('inf') # Total cost
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.coords == other.coords

    def __hash__(self):
        return hash(self.coords)


class AStar:
    """A* pathfinding algorithm for 2D and 3D grids"""

    def __init__(self, grid):
        self.grid = grid
        self._determine_dimensions()

    def _determine_dimensions(self):
        """Determine if grid is 2D or 3D"""
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
        return path[::-1]

    def find_path(self, start, goal):
        """Find shortest path using A*"""
        path, _, _, _ = self.find_path_with_stats(start, goal)
        return path

    def find_path_with_stats(self, start, goal):
        """
        Find shortest path using A* and return statistics
        Returns: (path, explored_nodes, max_frontier_size, nodes_evaluated)
        """
        start_node = Node(*start)
        goal_node = Node(*goal)

        # Validate start and goal
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

        open_set = []
        heapq.heappush(open_set, start_node)
        closed_set = set()
        explored_nodes = [] # Track order of exploration
        all_nodes = {start: start_node}

        # Statistics
        max_frontier_size = 1
        nodes_evaluated = 0

        while open_set:
            current = heapq.heappop(open_set)
            nodes_evaluated += 1

            if current == goal_node:
                return self.reconstruct_path(current), explored_nodes, max_frontier_size, nodes_evaluated

            closed_set.add(current.coords)
            explored_nodes.append(current.coords)

            for neighbor in self.get_neighbors(current):
                neighbor_tuple = neighbor.coords

                if neighbor_tuple in closed_set:
                    continue

                tentative_g = current.g + 1

                if neighbor_tuple not in all_nodes:
                    all_nodes[neighbor_tuple] = neighbor
                neighbor = all_nodes[neighbor_tuple]

                if tentative_g < neighbor.g:
                    neighbor.parent = current
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor, goal_node)
                    neighbor.f = neighbor.g + neighbor.h

                    if neighbor not in open_set:
                        heapq.heappush(open_set, neighbor)

            max_frontier_size = max(max_frontier_size, len(open_set))

        return None, explored_nodes, max_frontier_size, nodes_evaluated


class MazeGenerator:
    """Generate random mazes using recursive backtracking"""

    def __init__(self, width, height, depth=None):
        """
        Initialize maze generator
        Dimensions should be odd for proper maze generation
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.is_3d = depth is not None

        if self.is_3d:
            # Initialize 3D grid with all walls
            self.grid = [[[1 for _ in range(width)] for _ in range(height)] for _ in range(depth)]
        else:
            # Initialize 2D grid with all walls
            self.grid = [[1 for _ in range(width)] for _ in range(height)]

    def generate(self):
        """Generate a random maze using recursive backtracking"""
        if self.is_3d:
            raise NotImplementedError("3D maze generation not yet implemented")

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
            neighbors = self._get_unvisited_neighbors(current_x, current_y)

            if neighbors:
                # Choose a random neighbor
                next_x, next_y = random.choice(neighbors)

                # Remove wall between current and chosen neighbor
                wall_x = (current_x + next_x) // 2
                wall_y = (current_y + next_y) // 2
                self.grid[wall_x][wall_y] = 0
                self.grid[next_x][next_y] = 0

                # Add neighbor to stack
                stack.append((next_x, next_y))
            else:
                # Backtrack
                stack.pop()

        # Ensure start and goal areas are clear
        self._clear_area(0, 0)
        self._clear_area(self.height-1, self.width-1)

        return self.grid

    def _get_unvisited_neighbors(self, x, y):
        """Get unvisited neighbors that are 2 cells away"""
        neighbors = []
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check if neighbor is within bounds and unvisited
            if 0 <= nx < self.height and 0 <= ny < self.width and self.grid[nx][ny] == 1:
                neighbors.append((nx, ny))

        return neighbors

    def _clear_area(self, x, y, radius=1):
        """Clear a small area around a point"""
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.height and 0 <= ny < self.width:
                    self.grid[nx][ny] = 0

    def add_random_paths(self, percentage=0.1):
        """Add random paths to make the maze less perfect"""
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
