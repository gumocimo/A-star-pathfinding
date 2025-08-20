#!/usr/bin/env python3
"""
A* Pathfinding Algorithm - Main Entry Point with Configuration

Usage:
    python main.py [options]

Options:
    --width WIDTH       Grid width (default: from config)
    --height HEIGHT     Grid height (default: from config)
    --depth DEPTH       Grid depth for 3D mode (default: from config)
    --no-visualize      Disable console visualization
    --no-plot           Disable matplotlib plot
    --static            Use static plot instead of animated
    --learning-mode     Enable learning mode to see g/h values
    --no-learning-mode  Disable learning mode (overrides config)
    --save-plot PATH    Save plot to file (png for static, gif/mp4 for animated)
    --interval MS       Animation interval in milliseconds (default: 50)
    --mode {2d,3d}      Choose between 2D and 3D mode (default: 2d)
    --seed SEED         Random seed for maze generation
    --sample-maze       Use the sample maze instead of generating random
    --random-paths PCT  Percentage of random paths to add (0.0-1.0)

Author: gumocimo
Date: 20/08/2025
"""

import argparse
import random
import sys

from src.core import AStar, MazeGenerator
from src.visualization import ConsoleVisualizer, MatplotlibVisualizer


# ========== Configuration Settings ==========
# Default mode
DEFAULT_MODE = '2d' # '2d' or '3d'

# Default dimensions for 2D mode
DEFAULT_WIDTH_2D = 25
DEFAULT_HEIGHT_2D = 15

# Default dimensions for 3D mode (smaller for performance)
DEFAULT_WIDTH_3D = 8
DEFAULT_HEIGHT_3D = 8
DEFAULT_DEPTH_3D = 8

# Maze generation settings
MAZE_RANDOM_SEED = None # Set to integer for reproducible mazes
MAZE_RANDOM_PATHS_PERCENTAGE = 0.25 # Percentage of additional paths (0.0 - 1.0)

# Start and goal positions
DEFAULT_START_2D = (0, 0)
DEFAULT_GOAL_2D = None # Will be set to (height-1, width-1) if None
DEFAULT_START_3D = (0, 0, 0)
DEFAULT_GOAL_3D = None # Will be set to (depth-1, height-1, width-1) if None

# Visualization settings
SHOW_VISUALIZATION = True
SHOW_PLOT = True # Show matplotlib plot if available
PLOT_SAVE_PATH = None # Set to filename to save plot instead of showing

# Animation settings
ANIMATE_PLOT = True # Show animated visualization by default
ANIMATION_INTERVAL = 50 # Milliseconds between frames (50ms = 20fps)
ANIMATION_INTERVAL_3D = 10 # Faster for 3D to handle more frames

# Learning mode settings
LEARNING_MODE = False # Set to True to show g/h values during animation
LEARNING_MODE_INTERVAL = 250 # Slower animation for learning mode
LEARNING_MODE_INTERVAL_3D = 500 # Even slower for 3D learning mode

# Console visualization
CONSOLE_SYMBOLS = {
    'walkable': '.',
    'obstacle': '#',
    'path': '*',
    'start': 'S',
    'goal': 'G'
}

# Plot colors (RGB values 0-1)
PLOT_COLORS = {
    'wall': [0.2, 0.2, 0.2], # Dark gray
    'empty': [0.95, 0.95, 0.95], # White
    'explored': [0.8, 0.9, 1.0], # Light blue
    'frontier': [1.0, 1.0, 0.6], # Light yellow
    'current': [0.4, 1.0, 0.4], # Light green
    'path': [1.0, 0.3, 0.3], # Red
    'start': [0, 0.8, 0], # Green
    'goal': [0.5, 0.2, 0.8] # Purple
}

# 3D specific settings
VOXEL_SIZE = 0.8 # Size of voxels in 3D visualization
VOXEL_EDGE_WIDTH = 0.5 # Edge width for voxels


# ========== Main Execution ==========
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='A* Pathfinding Algorithm Implementation'
    )
    parser.add_argument(
        '--width', type=int, default=None,
        help='Grid width (should be odd for proper 2D maze generation)'
    )
    parser.add_argument(
        '--height', type=int, default=None,
        help='Grid height (should be odd for proper 2D maze generation)'
    )
    parser.add_argument(
        '--depth', type=int, default=None,
        help='Grid depth for 3D mode'
    )
    parser.add_argument(
        '--no-visualize', action='store_true',
        help='Disable console visualization'
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Disable matplotlib plot'
    )
    parser.add_argument(
        '--static', action='store_true',
        help='Use static plot instead of animated'
    )

    # Learning mode arguments
    learning_group = parser.add_mutually_exclusive_group()
    learning_group.add_argument(
        '--learning-mode', action='store_true',
        help='Enable learning mode to see g/h values during animation'
    )
    learning_group.add_argument(
        '--no-learning-mode', action='store_true',
        help='Disable learning mode (overrides config file setting)'
    )

    parser.add_argument(
        '--save-plot', type=str, default=PLOT_SAVE_PATH,
        help='Save plot to file (png for static, gif/mp4 for animated)'
    )
    parser.add_argument(
        '--interval', type=int, default=None,
        help='Animation interval in milliseconds'
    )
    parser.add_argument(
        '--mode', choices=['2d', '3d'], default=DEFAULT_MODE,
        help='Choose between 2D and 3D mode (default: from config)'
    )
    parser.add_argument(
        '--seed', type=int, default=MAZE_RANDOM_SEED,
        help='Random seed for maze generation'
    )
    parser.add_argument(
        '--sample-maze', action='store_true',
        help='Use the sample maze instead of generating random (2D only)'
    )
    parser.add_argument(
        '--random-paths', type=float, default=MAZE_RANDOM_PATHS_PERCENTAGE,
        help='Percentage of random paths to add (0.0-1.0)'
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()

    # Determine learning mode setting
    if args.learning_mode:
        learning_mode = True
    elif args.no_learning_mode:
        learning_mode = False
    else:
        learning_mode = LEARNING_MODE

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Determine dimensions based on mode
    if args.mode == '3d':
        # Use 3D defaults if not specified
        width = args.width if args.width else DEFAULT_WIDTH_3D
        height = args.height if args.height else DEFAULT_HEIGHT_3D
        depth = args.depth if args.depth else DEFAULT_DEPTH_3D

        # Set default interval for 3D if not specified
        if args.interval is None:
            interval = LEARNING_MODE_INTERVAL_3D if learning_mode else ANIMATION_INTERVAL_3D
        else:
            interval = args.interval
    else:
        # 2D mode
        if args.sample_maze:
            print("Using sample maze...")
            grid = MazeGenerator.create_sample_maze()
            height, width = len(grid), len(grid[0])
        else:
            # For learning mode, use smaller default size if not specified
            if learning_mode and args.width is None and args.height is None:
                width, height = 21, 21
                print("Note: Using 21x21 maze for better learning mode visibility")
            else:
                # Use specified or default dimensions
                width = args.width if args.width else DEFAULT_WIDTH_2D
                height = args.height if args.height else DEFAULT_HEIGHT_2D
                # Ensure odd dimensions for proper maze generation
                width = width if width % 2 == 1 else width+1
                height = height if height % 2 == 1 else height+1

        depth = None

        # Set default interval for 2D if not specified
        if args.interval is None:
            interval = LEARNING_MODE_INTERVAL if learning_mode else ANIMATION_INTERVAL
        else:
            interval = args.interval

    # Create or generate maze
    if args.mode == '3d':
        print(f"Generating 3D maze ({depth}x{height}x{width})...")
        print("Obstacles include: pillars, floating blocks, and wall segments")
        maze_gen = MazeGenerator(width, height, depth)
        grid = maze_gen.generate()

        # No additional random paths for 3D
    elif not args.sample_maze:
        print(f"Generating random maze ({height}x{width})...")
        maze_gen = MazeGenerator(width, height)
        grid = maze_gen.generate()

        # Add random paths if specified
        if args.random_paths > 0:
            print(f"Adding random paths ({args.random_paths * 100:.1f}% of cells)...")
            maze_gen.add_random_paths(args.random_paths)

    # Define start and goal
    if args.mode == '3d':
        start = DEFAULT_START_3D
        goal = DEFAULT_GOAL_3D if DEFAULT_GOAL_3D else (depth-1, height-1, width-1)
    else:
        start = DEFAULT_START_2D
        goal = DEFAULT_GOAL_2D if DEFAULT_GOAL_2D else (height-1, width-1)

    # Create pathfinder and find path with animation data
    print(f"\nFinding path with {'3D' if args.mode == '3d' else '2D'} A* algorithm...")
    if learning_mode:
        print("LEARNING MODE is ON - Animation will show g (cost from start) and h (heuristic to goal) values")
        if args.mode == '3d':
            print("Cost values shown: g/h format")

    astar = AStar(grid)
    path, animation_states, max_frontier, nodes_evaluated = astar.find_path_with_animation_data(start, goal)

    # Calculate statistics
    if path:
        path_length = len(path)
        nodes_explored = len(animation_states)
        efficiency = (path_length / nodes_explored) * 100 if nodes_explored > 0 else 0

        print(f"Path found! Length: {path_length}")
        print(f"Nodes explored: {nodes_explored}")
        print(f"Nodes evaluated: {nodes_evaluated}")
        print(f"Max frontier size: {max_frontier}")
        print(f"Efficiency: {efficiency:.1f}%")
    else:
        print("No path found!")
        path_length = 0
        nodes_explored = len(animation_states)
        efficiency = 0

    # Show maze statistics
    if args.mode == '3d':
        total_cells = width * height * depth
        walkable_cells = sum(sum(sum(1 for cell in row if cell == 0) for row in layer) for layer in grid)
    else:
        total_cells = width * height
        walkable_cells = sum(row.count(0) for row in grid)

    print(f"\nMaze statistics:")
    if args.mode == '3d':
        print(f"- Size: {depth}x{height}x{width}")
    else:
        print(f"- Size: {height}x{width}")
    print(f"- Total cells: {total_cells}")
    print(f"- Walkable cells: {walkable_cells} ({walkable_cells / total_cells * 100:.1f}%)")
    print(f"- Obstacles: {total_cells - walkable_cells} ({(total_cells - walkable_cells) / total_cells * 100:.1f}%)")

    # Console visualization
    if not args.no_visualize:
        print("\nConsole visualization:")
        if args.mode == '2d':
            print("Legend: S=Start, G=Goal, #=Wall, .=Empty, *=Path")
            print("-"*(width*2 -1))
        ConsoleVisualizer.visualize_path(grid, path, start, goal)

    # Matplotlib visualization
    if not args.no_plot and SHOW_PLOT:
        if MatplotlibVisualizer.is_available():
            stats = {
                'path_length': path_length,
                'nodes_explored': nodes_explored,
                'nodes_evaluated': nodes_evaluated,
                'max_frontier': max_frontier,
                'efficiency': efficiency
            }

            if args.static and args.mode == '2d':
                print("\nGenerating static visualization plot...")
                # For static plot, need to get explored nodes list
                _, explored_nodes, _, _ = astar.find_path_with_stats(start, goal)
                MatplotlibVisualizer.plot_solution(
                    grid, path, explored_nodes, start, goal, stats,
                    save_path=args.save_plot
                )
            else:
                print(f"\nGenerating {'3D' if args.mode == '3d' else '2D'} animated visualization...")
                if learning_mode:
                    print(
                        "Cost values shown: g (top), h (bottom)" if args.mode == '2d' else "Cost values shown: g/h format")
                print(f"Animation speed: {interval}ms per frame")
                if args.mode == '3d':
                    print("Use mouse to rotate, zoom, and pan the view!")

                MatplotlibVisualizer.plot_solution_animated(
                    grid, path, animation_states, start, goal, stats,
                    interval=interval, learning_mode=learning_mode,
                    save_path=args.save_plot
                )
        else:
            print("\nMatplotlib not available. Install with: pip install matplotlib numpy")


if __name__ == "__main__":
    main()
