# A* Pathfinding Algorithm

Implementation of the A* pathfinding algorithm with animated visualization and comprehensive analytics.

## Overview

A* is an informed search algorithm that finds the optimal path between nodes by combining:
- **g-score**: Actual distance from start
- **h-score**: Heuristic estimate to goal (Manhattan distance)
- **f-score**: Total cost (g+h)

## Features

- **Live animated visualization** showing algorithm progression
- **Real-time statistics update** during exploration
- **Configurable animation speed**
- **Save animations** as GIF or video files
- **Choice between static and animated** visualization
- Matplotlib visualization with color-coded elements
- Performance metrics display
- Algorithm exploration tracking
- Random maze generation using recursive backtracking
- Adjustable maze complexity
- Seeded generation for reproducible mazes

## Project Structure

```
A-star-pathfinding/
├── src/
│   ├── __init__.py         # Package initialization
│   ├── core.py             # Core algorithms (Node, AStar, MazeGenerator)
│   └── visualization.py    # Visualization utilities (Console & Matplotlib)
├── main.py                 # Main entry point with configuration
├── requirements.txt
└── README.md
```

## Usage

### Basic Usage
```bash
# Run with default settings and animated visualization
python main.py
```

### Static Visualization
```bash
# Use static plot instead of animation
python main.py --static
```

### Animation Control
```bash
# Slower animation for better observation
python main.py --interval 200

# Fast animation for large mazes
python main.py --width 71 --height 41 --interval 10

# Save animation as GIF
python main.py --save-plot solution.gif

# Save as video (requires ffmpeg)
python main.py --save-plot solution.mp4
```

### Maze Generation
```bash
# Custom size (should be odd numbers for best results)
python main.py --width 31 --height 21

# Reproducible maze with seed
python main.py --seed 42

# Add more random paths (0-100% of cells)
python main.py --random-paths 0.1

# Use sample maze
python main.py --sample-maze
```

### Visualization Options
```bash
# Disable console visualization
python main.py --no-visualize

# Disable matplotlib plot
python main.py --no-plot

# Save static plot
python main.py --static --save-plot solution.png
```

## Visual Analytics

### Animation Features
- **Real-time exploration**: Watch the algorithm explore the maze
- **Dynamic statistics**: See metrics update as the search progresses
- **Path reveal**: Final optimal path highlighted after exploration
- **Adjustable speed**: Control animation pace with interval setting

### Color Coding
- 🟩 **Green**: Start position
- 🟪 **Purple**: Goal position
- 🔴 **Red**: Optimal path
- 🔵 **Light Blue**: Explored nodes
- ⬛ **Dark Gray**: Walls
- ⬜ **White**: Empty cells

### Performance Metrics
- **Path Length**: Steps in optimal path
- **Nodes Explored**: Total cells examined
- **Nodes Evaluated**: Nodes popped from priority queue
- **Max Frontier Size**: Largest open set size
- **Efficiency**: Path length / nodes explored ratio

## Algorithm Details

### Maze Generation (Recursive Backtracking)
1. Start with a grid of walls
2. Pick a random starting cell
3. Recursively carve paths to unvisited neighbors
4. Backtrack when no unvisited neighbors remain
5. Optionally add random paths for multiple solutions

### A* Pathfinding
1. Initialize start node with g=0, calculate h using Manhattan distance
2. Pop lowest f-score node from priority queue
3. Check all valid neighbors (4-directional movement)
4. Update costs and parent if better path found
5. Repeat until goal reached or no nodes left
6. Reconstruct path by tracing parent pointers

### Animation Process
1. **Exploration Phase**: Nodes are explored one by one
2. **Statistics Update**: Metrics update in real-time
3. **Path Drawing**: Final path is revealed after exploration
4. **Final Display**: Complete solution with all statistics

## Animation Formats

- **PNG**: Static plots (`--static --save-plot solution.png`)
- **GIF**: Animated plots (`--save-plot solution.gif`)
- **MP4**: Video animations (`--save-plot solution.mp4`) - requires ffmpeg
