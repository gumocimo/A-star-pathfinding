# A* Pathfinding Algorithm

Implementation of the A* pathfinding algorithm with random maze generation and visual analytics.

## Overview

A* is an informed search algorithm that finds the optimal path between nodes by combining:
- **g-score**: Actual distance from start
- **h-score**: Heuristic estimate to goal (Manhattan distance)
- **f-score**: Total cost (g+h)

## Features
- **Matplotlib visualization** with color-coded elements
- **Performance metrics display** in the plot
- **Algorithm exploration tracking** showing which nodes were explored
- **Save plot to file** option
- Random maze generation using recursive backtracking
- Adjustable maze complexity with random path addition

## Project Structure

```
A-star-pathfinding/
├── core.py           # Core algorithms (Node, AStar, MazeGenerator, Visualizers)
├── main.py           # Main entry point with configuration
├── requirements.txt
└── README.md  
```

The core algorithm works without external dependencies, but matplotlib and numpy are needed for graphical visualization.

## Usage

### Basic Usage
```bash
python main.py
```

### Generate Random Maze
```bash
# Custom size (should be odd numbers for best results)
python main.py --width 31 --height 21

# With reproducible seed
python main.py --seed 42

# Add more random paths (0-100% of cells)
python main.py --random-paths 0.1
```

### Visualization Options
```bash
# Disable console visualization
python main.py --no-visualize

# Disable matplotlib plot
python main.py --no-plot

# Save plot to file
python main.py --save-plot solution.png
```

## Visual Analytics

The matplotlib visualization provides:
- **Color-coded elements**:
  - 🟩 Green: Start position
  - 🟪 Purple: Goal position
  - 🔴 Red: Optimal path
  - 🔵 Light Blue: Explored nodes
  - ⬛ Dark Gray: Walls
  - ⬜ White: Empty cells

- **Performance Metrics**:
  - Path Length: Steps in optimal path
  - Nodes Explored: Total cells examined
  - Nodes Evaluated: Nodes popped from priority queue
  - Max Frontier Size: Largest open set size
  - Efficiency: Path length / nodes explored ratio

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

### Performance Analysis
The implementation tracks:
- Exploration order to visualize algorithm progress
- Frontier size evolution
- Node evaluation count
- Efficiency metrics for algorithm performance
