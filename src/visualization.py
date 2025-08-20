"""
Visualization utilities for pathfinding

Author: gumocimo
Date: 20/08/2025
"""

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ConsoleVisualizer:
    """Text-based visualization for grids and paths"""

    @staticmethod
    def visualize_path(grid, path, start, goal):
        """Visualize the grid and path in console"""
        # For 3D grids, show layer by layer
        if isinstance(grid[0][0], list): # 3D grid
            depth = len(grid)
            for z in range(depth):
                print(f"\nLayer {z}:")
                # Create a 2D slice
                layer = grid[z]
                visual = [row[:] for row in layer]

                # Mark the path in this layer
                if path:
                    for coords in path:
                        if len(coords) == 3 and coords[0] == z:
                            x, y, z_coord = coords
                            if coords != start and coords != goal:
                                visual[y][z_coord] = 2 # Path marker

                # Mark start and goal if in this layer
                if len(start) == 3 and start[0] == z:
                    visual[start[1]][start[2]] = 3 # Start marker
                if len(goal) == 3 and goal[0] == z:
                    visual[goal[1]][goal[2]] = 4 # Goal marker

                # Print the layer
                symbols = {
                    0: '.', # Walkable
                    1: '#', # Obstacle
                    2: '*', # Path
                    3: 'S', # Start
                    4: 'G'  # Goal
                }

                for row in visual:
                    print(' '.join(symbols.get(cell, '?') for cell in row))
        else:
            # 2D visualization (existing code)
            visual = [row[:] for row in grid]

            # Mark the path
            if path:
                for coords in path:
                    if len(coords) == 2: # 2D
                        x, y = coords
                        if (x, y) != start and (x, y) != goal:
                            visual[x][y] = 2 # Path marker

            # Mark start and goal for 2D
            if len(start) == 2:
                visual[start[0]][start[1]] = 3 # Start marker
                visual[goal[0]][goal[1]] = 4 # Goal marker

            # Print the grid
            symbols = {
                0: '.', # Walkable
                1: '#', # Obstacle
                2: '*', # Path
                3: 'S', # Start
                4: 'G'  # Goal
            }

            for row in visual:
                print(' '.join(symbols.get(cell, '?') for cell in row))


class MatplotlibVisualizer:
    """Matplotlib-based visualization for grids and paths"""

    @staticmethod
    def is_available():
        """Check if matplotlib is available"""
        return MATPLOTLIB_AVAILABLE

    @staticmethod
    def plot_solution(grid, path, explored_nodes, start, goal, stats, save_path=None):
        """Create a static plot showing the solution and algorithm evolution"""
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available. Install with: pip install matplotlib numpy")
            return

        # Check if 3D
        if isinstance(grid[0][0], list):
            # For now, just show a message
            print("Static 3D visualization not implemented. Use animated mode.")
            return

        # 2D visualization (existing code)
        maze = np.array(grid)
        height, width = maze.shape

        # Create figure with proper size
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create display array
        display = np.ones((height, width, 3)) # RGB array

        # Define colors (RGB)
        colors = {
            'wall': [0.2, 0.2, 0.2], # Dark gray
            'empty': [0.95, 0.95, 0.95], # White
            'explored': [0.8, 0.9, 1.0], # Light blue
            'path': [1.0, 0.3, 0.3], # Red
            'start': [0, 0.8, 0], # Green
            'goal': [0.5, 0.2, 0.8] # Purple
        }

        # Fill maze structure
        for i in range(height):
            for j in range(width):
                if maze[i][j] == 1: # Wall
                    display[i, j] = colors['wall']
                else: # Empty
                    display[i, j] = colors['empty']

        # Mark explored nodes
        for coords in explored_nodes:
            x, y = coords[:2] # Handle both 2D and 3D
            if (x, y) != start[:2] and (x, y) != goal[:2]:
                display[x, y] = colors['explored']

        # Mark path
        if path:
            for coords in path:
                x, y = coords[:2] # Handle both 2D and 3D
                if (x, y) != start[:2] and (x, y) != goal[:2]:
                    display[x, y] = colors['path']

        # Mark start and goal
        display[start[0], start[1]] = colors['start']
        display[goal[0], goal[1]] = colors['goal']

        # Display the image
        ax.imshow(display, interpolation='nearest', aspect='equal')

        # Remove axes
        ax.axis('off')

        # Add title
        ax.set_title('A* Pathfinding Solution', fontsize=16, fontweight='bold', pad=20)

        # Add statistics text
        stats_text = f"Path Length: {stats['path_length']}\n"
        stats_text += f"Nodes Explored: {stats['nodes_explored']}\n"
        stats_text += f"Nodes Evaluated: {stats['nodes_evaluated']}\n"
        stats_text += f"Max Frontier Size: {stats['max_frontier']}\n"
        stats_text += f"Efficiency: {stats['efficiency']:.1f}%"

        # Add statistics box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(1.02, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

        # Add legend
        legend_elements = [
            patches.Rectangle((0, 0), 1, 1, facecolor=colors['wall'], label='Wall'),
            patches.Rectangle((0, 0), 1, 1, facecolor=colors['empty'], label='Empty'),
            patches.Rectangle((0, 0), 1, 1, facecolor=colors['explored'], label='Explored'),
            patches.Rectangle((0, 0), 1, 1, facecolor=colors['path'], label='Path'),
            patches.Rectangle((0, 0), 1, 1, facecolor=colors['start'], label='Start'),
            patches.Rectangle((0, 0), 1, 1, facecolor=colors['goal'], label='Goal')
        ]

        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                  frameon=True, fancybox=True, shadow=True)

        # Adjust layout to prevent legend cutoff
        plt.tight_layout()

        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

    @staticmethod
    def plot_solution_animated(grid, path, animation_states, start, goal, stats,
                               interval=50, learning_mode=False, save_path=None):
        """Create an animated plot showing the algorithm evolution"""
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available. Install with: pip install matplotlib numpy")
            return

        # Check if 3D
        if isinstance(grid[0][0], list):
            MatplotlibVisualizer._plot_solution_3d_animated(
                grid, path, animation_states, start, goal, stats,
                interval, learning_mode, save_path
            )
        else:
            MatplotlibVisualizer._plot_solution_2d_animated(
                grid, path, animation_states, start, goal, stats,
                interval, learning_mode, save_path
            )

    @staticmethod
    def _plot_solution_2d_animated(grid, path, animation_states, start, goal, stats,
                                   interval=50, learning_mode=False, save_path=None):
        """2D animated visualization (original simple style)"""
        # Import AlgorithmState if using animation states
        from src.core import AlgorithmState

        # Convert grid to numpy array
        maze = np.array(grid)
        height, width = maze.shape

        # Create figure with proper size
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create base display array
        display = np.ones((height, width, 3)) # RGB array

        # Define colors (RGB)
        colors = {
            'wall': [0.2, 0.2, 0.2], # Dark gray
            'empty': [0.95, 0.95, 0.95], # White
            'explored': [0.8, 0.9, 1.0], # Light blue
            'frontier': [1.0, 1.0, 0.6], # Light yellow
            'current': [0.4, 1.0, 0.4], # Light green
            'path': [1.0, 0.3, 0.3], # Red
            'start': [0, 0.8, 0], # Green
            'goal': [0.5, 0.2, 0.8] # Purple
        }

        # Fill initial maze structure
        for i in range(height):
            for j in range(width):
                if maze[i][j] == 1: # Wall
                    display[i, j] = colors['wall']
                else: # Empty
                    display[i, j] = colors['empty']

        # Mark start and goal
        display[start[0], start[1]] = colors['start']
        display[goal[0], goal[1]] = colors['goal']

        # Create the image
        im = ax.imshow(display, interpolation='nearest', aspect='equal')

        # Remove axes
        ax.axis('off')

        # Add title
        title_text = 'A* Pathfinding Algorithm Visualization'
        if learning_mode:
            title_text += ' (Learning Mode)'
        ax.set_title(title_text, fontsize=15, fontweight='bold', pad=20)

        # Add statistics text (will be updated during animation)
        stats_text = ax.text(1.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Add legend - always include frontier in legend
        legend_elements = [
            patches.Rectangle((0, 0), 1, 1, facecolor=colors['wall'], label='Wall'),
            patches.Rectangle((0, 0), 1, 1, facecolor=colors['empty'], label='Empty'),
            patches.Rectangle((0, 0), 1, 1, facecolor=colors['explored'], label='Explored'),
            patches.Rectangle((0, 0), 1, 1, facecolor=colors['frontier'], label='Frontier'),
            patches.Rectangle((0, 0), 1, 1, facecolor=colors['current'], label='Current'),
            patches.Rectangle((0, 0), 1, 1, facecolor=colors['path'], label='Path'),
            patches.Rectangle((0, 0), 1, 1, facecolor=colors['start'], label='Start'),
            patches.Rectangle((0, 0), 1, 1, facecolor=colors['goal'], label='Goal')
        ]

        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                  frameon=True, fancybox=True, shadow=True)

        # Adjust layout to prevent legend cutoff
        plt.tight_layout()

        # Store text annotations for cells
        cell_texts = {}

        # Animation state
        current_step = 0
        exploration_done = False

        def update_stats_text(state_idx):
            """Update the statistics text"""
            if state_idx < len(animation_states):
                explored_count = len(animation_states[state_idx].explored_nodes)
                frontier_size = len(animation_states[state_idx].frontier)
            else:
                explored_count = len(animation_states[-1].explored_nodes) if animation_states else 0
                frontier_size = 0

            text = f"Path Length: {'-' if not exploration_done else stats['path_length']}\n"
            text += f"Nodes Explored: {explored_count}\n"
            text += f"Nodes Evaluated: {'-' if not exploration_done else stats['nodes_evaluated']}\n"
            text += f"Current Frontier Size: {frontier_size}\n"
            text += f"Max Frontier Size: {'-' if not exploration_done else stats['max_frontier']}\n"
            efficiency_str = '-' if not exploration_done else f"{stats['efficiency']:.1f}%"
            text += f"Efficiency: {efficiency_str}"
            stats_text.set_text(text)

        def animate(frame):
            """Animation function"""
            nonlocal current_step, exploration_done, display

            # Phase 1: Exploration
            if current_step < len(animation_states):
                state = animation_states[current_step]

                # Always reset display to base state (for frontier visualization)
                for i in range(height):
                    for j in range(width):
                        if maze[i][j] == 1: # Wall
                            display[i, j] = colors['wall']
                        else: # Empty
                            display[i, j] = colors['empty']

                # Mark explored nodes
                for coords in state.explored_nodes:
                    x, y = coords[:2]
                    if (x, y) != start[:2] and (x, y) != goal[:2]:
                        display[x, y] = colors['explored']

                # ALWAYS mark frontier nodes (not just in learning mode)
                for coords in state.frontier.keys():
                    x, y = coords[:2]
                    if (x, y) != start[:2] and (x, y) != goal[:2]:
                        display[x, y] = colors['frontier']

                # Mark current node
                if state.current_node and state.current_node[:2] != start[:2] and state.current_node[:2] != goal[:2]:
                    display[state.current_node[0], state.current_node[1]] = colors['current']

                # Handle text annotations only in learning mode
                if learning_mode:
                    # Clear previous texts
                    for text in cell_texts.values():
                        text.remove()
                    cell_texts.clear()

                    # Add text for frontier nodes showing g and h values
                    for coords, (g, h, f) in state.frontier.items():
                        x, y = coords[:2]
                        text = ax.text(y, x, f'{int(g)}\n{int(h)}',
                                       ha='center', va='center', fontsize=8,
                                       color='black', weight='bold')
                        cell_texts[coords] = text

                # Mark start and goal
                display[start[0], start[1]] = colors['start']
                display[goal[0], goal[1]] = colors['goal']

                # Update statistics
                update_stats_text(current_step)
                current_step += 1

            # Phase 2: Path drawing
            elif path and not exploration_done:
                exploration_done = True

                # Clear any remaining texts in learning mode
                if learning_mode:
                    for text in cell_texts.values():
                        text.remove()
                    cell_texts.clear()

                # Draw the final path
                for coords in path:
                    x, y = coords[:2]
                    if (x, y) != start[:2] and (x, y) != goal[:2]:
                        display[x, y] = colors['path']

                # Update final statistics
                update_stats_text(len(animation_states))

            # Update the image
            im.set_array(display)
            return [im, stats_text] + list(cell_texts.values())

        # Calculate total frames needed
        total_frames = len(animation_states) + (1 if path else 0)

        # Set interval based on learning mode
        if learning_mode:
            interval = max(interval, 250) # Minimum 250ms for learning mode

        # Create animation
        anim = FuncAnimation(fig, animate, frames=total_frames,
                             interval=interval, blit=False, repeat=False)

        # Save or show animation
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=1000 / interval)
                print(f"Animation saved to: {save_path}")
            else:
                # For video formats, need ffmpeg
                try:
                    anim.save(save_path, writer='ffmpeg', fps=1000 / interval)
                    print(f"Animation saved to: {save_path}")
                except:
                    print("Warning: Could not save animation. Install ffmpeg for video support.")
                    plt.show()
        else:
            plt.show()

    @staticmethod
    def _plot_solution_3d_animated(grid, path, animation_states, start, goal, stats,
                                   interval=50, learning_mode=False, save_path=None):
        """Create an animated 3D plot with consistent style to 2D"""
        # Import AlgorithmState
        from src.core import AlgorithmState

        # Convert grid to numpy array
        maze = np.array(grid)
        depth, height, width = maze.shape

        # Create figure with same size as 2D
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Set background color
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Define colors with alpha for transparency (same as 2D conceptually)
        colors = {
            'wall': [0.2, 0.2, 0.2, 0.9], # Dark gray obstacles
            'empty': [0.95, 0.95, 0.95, 0.0], # Transparent
            'explored': [0.8, 0.9, 1.0, 0.5], # Light blue
            'frontier': [1.0, 1.0, 0.6, 0.7], # Light yellow
            'current': [0.4, 1.0, 0.4, 0.8], # Light green
            'path': [1.0, 0.3, 0.3, 0.9], # Red
            'start': [0, 0.8, 0, 1.0], # Green
            'goal': [0.5, 0.2, 0.8, 1.0] # Purple
        }

        # Add title
        title_text = 'A* Pathfinding Algorithm Visualization (3D)'
        if learning_mode:
            title_text += ' (Learning Mode)'
        ax.set_title(title_text, fontsize=15, fontweight='bold', pad=20)

        # Create a text box for statistics (positioned similarly to 2D)
        stats_text = fig.text(0.82, 0.85, '', fontsize=10,
                              verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Animation state
        current_step = 0
        exploration_done = False

        # Store collections
        wall_collections = [] # Persistent walls
        dynamic_collections = [] # Changing elements
        text_objects = []

        def draw_voxel(x, y, z, color, size=0.9):
            """Draw a single voxel (cube) at position (x, y, z)"""
            # Define the vertices of a cube
            r = size/2
            vertices = [
                [x - r, y - r, z - r], [x + r, y - r, z - r],
                [x + r, y + r, z - r], [x - r, y + r, z - r],
                [x - r, y - r, z + r], [x + r, y - r, z + r],
                [x + r, y + r, z + r], [x - r, y + r, z + r]
            ]
            vertices = np.array(vertices)

            # Define the 6 faces of the cube
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]], # Bottom
                [vertices[4], vertices[5], vertices[6], vertices[7]], # Top
                [vertices[0], vertices[1], vertices[5], vertices[4]], # Front
                [vertices[2], vertices[3], vertices[7], vertices[6]], # Back
                [vertices[0], vertices[3], vertices[7], vertices[4]], # Left
                [vertices[1], vertices[2], vertices[6], vertices[5]]  # Right
            ]

            # Create the 3D polygon collection
            poly = Poly3DCollection(faces, facecolors=[color]*6,
                                    edgecolors='black',
                                    linewidths=0.5, alpha=color[3])
            return poly

        # Draw walls once at the beginning
        print("Drawing maze obstacles...")
        wall_count = 0
        for x in range(depth):
            for y in range(height):
                for z in range(width):
                    if maze[x][y][z] == 1:
                        poly = draw_voxel(x, y, z, colors['wall'], size=0.98)
                        ax.add_collection3d(poly)
                        wall_collections.append(poly)
                        wall_count += 1
        print(f"Rendered {wall_count} obstacle voxels")

        def update_stats_text(state_idx):
            """Update the statistics text (consistent with 2D)"""
            if state_idx < len(animation_states):
                explored_count = len(animation_states[state_idx].explored_nodes)
                frontier_size = len(animation_states[state_idx].frontier)
            else:
                explored_count = len(animation_states[-1].explored_nodes) if animation_states else 0
                frontier_size = 0

            text = f"Path Length: {'-' if not exploration_done else stats['path_length']}\n"
            text += f"Nodes Explored: {explored_count}\n"
            text += f"Nodes Evaluated: {'-' if not exploration_done else stats['nodes_evaluated']}\n"
            text += f"Current Frontier Size: {frontier_size}\n"
            text += f"Max Frontier Size: {'-' if not exploration_done else stats['max_frontier']}\n"
            efficiency_str = '-' if not exploration_done else f"{stats['efficiency']:.1f}%"
            text += f"Efficiency: {efficiency_str}"

            stats_text.set_text(text)

        def animate(frame):
            """Animation function"""
            nonlocal current_step, exploration_done

            # Clear only dynamic elements (not walls)
            for coll in dynamic_collections:
                coll.remove()
            dynamic_collections.clear()

            for text in text_objects:
                text.remove()
            text_objects.clear()

            # Phase 1: Exploration
            if current_step < len(animation_states):
                state = animation_states[current_step]

                # Draw explored nodes
                for x, y, z in state.explored_nodes:
                    if (x, y, z) != start and (x, y, z) != goal:
                        poly = draw_voxel(x, y, z, colors['explored'], size=0.8)
                        ax.add_collection3d(poly)
                        dynamic_collections.append(poly)

                # Draw frontier nodes
                for (x, y, z), (g, h, f) in state.frontier.items():
                    if (x, y, z) != start and (x, y, z) != goal:
                        poly = draw_voxel(x, y, z, colors['frontier'], size=0.85)
                        ax.add_collection3d(poly)
                        dynamic_collections.append(poly)

                        # Add text in learning mode
                        if learning_mode:
                            # Position text slightly above the voxel center
                            text = ax.text(x, y, z+0.6, f'{int(g)}/{int(h)}',
                                           ha='center', va='center', fontsize=9,
                                           color='black', weight='bold',
                                           bbox=dict(boxstyle='round,pad=0.2',
                                                     facecolor='yellow', alpha=0.8))
                            text_objects.append(text)

                # Draw current node
                if state.current_node and state.current_node != start and state.current_node != goal:
                    x, y, z = state.current_node
                    poly = draw_voxel(x, y, z, colors['current'], size=0.9)
                    ax.add_collection3d(poly)
                    dynamic_collections.append(poly)

                # Update statistics
                update_stats_text(current_step)
                current_step += 1

            # Phase 2: Path drawing
            else:
                exploration_done = True

                # Draw all explored nodes first
                if animation_states:
                    last_state = animation_states[-1]
                    for x, y, z in last_state.explored_nodes:
                        if (x, y, z) != start and (x, y, z) != goal:
                            poly = draw_voxel(x, y, z, colors['explored'], size=0.8)
                            ax.add_collection3d(poly)
                            dynamic_collections.append(poly)

                # Draw the final path
                if path:
                    for x, y, z in path:
                        if (x, y, z) != start and (x, y, z) != goal:
                            poly = draw_voxel(x, y, z, colors['path'], size=0.9)
                            ax.add_collection3d(poly)
                            dynamic_collections.append(poly)

                # Update final statistics
                update_stats_text(len(animation_states))

            # Always draw start and goal
            poly_start = draw_voxel(start[0], start[1], start[2], colors['start'], size=1.0)
            ax.add_collection3d(poly_start)
            dynamic_collections.append(poly_start)

            poly_goal = draw_voxel(goal[0], goal[1], goal[2], colors['goal'], size=1.0)
            ax.add_collection3d(poly_goal)
            dynamic_collections.append(poly_goal)

            # Set the axes properties
            ax.set_xlim(-0.5, depth-0.5)
            ax.set_ylim(-0.5, height-0.5)
            ax.set_zlim(-0.5, width-0.5)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.grid(True, alpha=0.3)

            return dynamic_collections + text_objects + [stats_text]

        # Set initial view angle
        ax.view_init(elev=20, azim=45)

        # Calculate total frames
        total_frames = len(animation_states) + (1 if path else 0)

        # Set interval based on learning mode
        if learning_mode:
            interval = max(interval, 500) # Slower for 3D learning mode

        # Create animation
        anim = FuncAnimation(fig, animate, frames=total_frames,
                             interval=interval, blit=False, repeat=False)

        # Show or save
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=1000 / interval)
                print(f"3D animation saved to: {save_path}")
            else:
                try:
                    anim.save(save_path, writer='ffmpeg', fps=1000 / interval)
                    print(f"3D animation saved to: {save_path}")
                except:
                    print("Warning: Could not save animation. Install ffmpeg for video support.")
                    plt.show()
        else:
            print("Use mouse to rotate, zoom, and pan the 3D view!")
            plt.show()
