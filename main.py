#!/usr/bin/env python3
"""
A* Pathfinding Algorithm - Main Entry Point

Implementation of the A* pathfinding algorithm with animated visualization
showing the algorithm's exploration process in real-time.

Usage:
    python main.py [options]

Options:
    --width WIDTH       Grid width (default: 45, should be odd)
    --height HEIGHT     Grid height (default: 27, should be odd)
    --no-visualize      Disable console visualization
    --no-plot           Disable matplotlib plot
    --static            Use static plot instead of animated
    --save-plot PATH    Save plot to file (png for static, gif/mp4 for animated)
    --interval MS       Animation interval in milliseconds (default: 50)
    --mode {2d,3d}      Choose between 2D and 3D mode (default: 2d)
    --seed SEED         Random seed for maze generation
    --sample-maze       Use the sample maze instead of generating random
    --random-paths PCT  Percentage of random paths to add (0.0-1.0)

Author: gumocimo
Date: 14/08/2025
"""

import argparse
import random
import sys
from src import AStar, MazeGenerator, ConsoleVisualizer, MatplotlibVisualizer

# Configuration settings
DEFAULT_WIDTH = 45
DEFAULT_HEIGHT = 27
DEFAULT_DEPTH = 8 # For 3D mode - NOT IMPLEMENTED YET

# Maze generation settings
MAZE_RANDOM_SEED = None # Set to integer for reproducible mazes
MAZE_RANDOM_PATHS_PERCENTAGE = 0.25 # Percentage of additional paths

# Start and goal positions
DEFAULT_START_2D = (0, 0)
DEFAULT_GOAL_2D = None # Will be set to (height-1, width-1) if None
DEFAULT_START_3D = (0, 0, 0)
DEFAULT_GOAL_3D = None # Will be set to (depth-1, height-1, width-1) if None

# Visualization settings
SHOW_VISUALIZATION = True
SHOW_PLOT = True # Show matplotlib plot if available
PLOT_SAVE_PATH = None # Set to filename to save plot instead of showing
ANIMATE_PLOT = True # Show animated visualization by default
ANIMATION_INTERVAL = 50 # Milliseconds between frames (50ms = 20fps)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='A* Pathfinding Algorithm Implementation'
    )
    parser.add_argument(
        '--width', type=int, default=DEFAULT_WIDTH,
        help='Grid width (should be odd for proper maze generation)'
    )
    parser.add_argument(
        '--height', type=int, default=DEFAULT_HEIGHT,
        help='Grid height (should be odd for proper maze generation)'
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
    parser.add_argument(
        '--save-plot', type=str, default=PLOT_SAVE_PATH,
        help='Save plot to file (png for static, gif/mp4 for animated)'
    )
    parser.add_argument(
        '--interval', type=int, default=ANIMATION_INTERVAL,
        help='Animation interval in milliseconds'
    )
    parser.add_argument(
        '--mode', choices=['2d', '3d'], default='2d',
        help='Choose between 2D and 3D mode'
    )
    parser.add_argument(
        '--seed', type=int, default=MAZE_RANDOM_SEED,
        help='Random seed for maze generation'
    )
    parser.add_argument(
        '--sample-maze', action='store_true',
        help='Use the sample maze instead of generating random'
    )
    parser.add_argument(
        '--random-paths', type=float, default=MAZE_RANDOM_PATHS_PERCENTAGE,
        help='Percentage of random paths to add (0.0-1.0)'
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Only 2D visualization is supported
    if args.mode == '3d':
        print("3D visualization not yet implemented. Using 2D mode.")
        args.mode = '2d'

    # Create or generate maze
    if args.sample_maze:
        print("Using sample maze...")
        grid = MazeGenerator.create_sample_maze()
        height, width = len(grid), len(grid[0])
    else:
        # Ensure odd dimensions for proper maze generation
        width = args.width if args.width % 2 == 1 else args.width + 1
        height = args.height if args.height % 2 == 1 else args.height + 1

        print(f"Generating random maze ({height}x{width})...")
        maze_gen = MazeGenerator(width, height)
        grid = maze_gen.generate()

        # Add random paths if specified
        if args.random_paths > 0:
            print(f"Adding random paths ({args.random_paths * 100:.1f}% of cells)...")
            maze_gen.add_random_paths(args.random_paths)

    # Define start and goal
    start = DEFAULT_START_2D
    goal = DEFAULT_GOAL_2D if DEFAULT_GOAL_2D else (height - 1, width - 1)

    # Find path with statistics
    print("\nFinding path with A* algorithm...")
    astar = AStar(grid)
    path, explored_nodes, max_frontier, nodes_evaluated = astar.find_path_with_stats(start, goal)

    # Calculate statistics
    if path:
        path_length = len(path)
        nodes_explored = len(explored_nodes)
        efficiency = (path_length / nodes_explored) * 100 if nodes_explored > 0 else 0

        print(f"Path found! Length: {path_length}")
        print(f"Nodes explored: {nodes_explored}")
        print(f"Nodes evaluated: {nodes_evaluated}")
        print(f"Max frontier size: {max_frontier}")
        print(f"Efficiency: {efficiency:.1f}%")

        if path_length <= 10:
            print(f"Path: {path}")
        else:
            print(f"Path: {path[:5]}...{path[-5:]}")
    else:
        print("No path found!")
        path_length = 0
        nodes_explored = len(explored_nodes)
        nodes_evaluated = 0
        max_frontier = 0
        efficiency = 0

    # Show maze statistics
    total_cells = width * height
    walkable_cells = sum(row.count(0) for row in grid)
    print(f"\nMaze statistics:")
    print(f"- Size: {height}x{width}")
    print(f"- Total cells: {total_cells}")
    print(f"- Walkable cells: {walkable_cells} ({walkable_cells / total_cells * 100:.1f}%)")
    print(f"- Obstacles: {total_cells - walkable_cells} ({(total_cells - walkable_cells) / total_cells * 100:.1f}%)")

    # Console visualization
    if not args.no_visualize:
        print("\nConsole visualization:")
        print("Legend: S=Start, G=Goal, #=Wall, .=Empty, *=Path")
        print("-" * (width * 2 - 1))
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

            if args.static:
                print("\nGenerating static visualization plot...")
                MatplotlibVisualizer.plot_solution(
                    grid, path, explored_nodes, start, goal, stats,
                    save_path=args.save_plot
                )
            else:
                print("\nGenerating animated visualization...")
                print(f"Animation speed: {args.interval}ms per frame")
                MatplotlibVisualizer.plot_solution_animated(
                    grid, path, explored_nodes, start, goal, stats,
                    interval=args.interval, save_path=args.save_plot
                )
        else:
            print("\nMatplotlib not available. Install with: pip install matplotlib numpy")


if __name__ == "__main__":
    main()
