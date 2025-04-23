import heapq
import gym
import gym_maze
import time
import os
import matplotlib.pyplot as plt
from maze_env import MazeEnvRandom10x10

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_solver(env, render=False):
    tic = time.perf_counter() # Start the timer
    start = tuple(env.maze_view.robot) # Starting position of the robot
    goal = tuple(env.maze_view.goal) # Since it is a heuristic search, we need to know the goal position

    open_list = []
    heapq.heappush(open_list, (0 + manhattan_distance(start, goal), 0, start)) 

    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_list:
        _, current_cost, current = heapq.heappop(open_list) # Get the node with the lowest f_score
        visited.add(current)

        if current == goal:
            break

        x, y = current
        neighbors = []
        if env.maze_view.maze.is_open((x, y), "N"):
            neighbors.append((x, y - 1))
        if env.maze_view.maze.is_open((x, y), "S"):
            neighbors.append((x, y + 1))
        if env.maze_view.maze.is_open((x, y), "E"):
            neighbors.append((x + 1, y))
        if env.maze_view.maze.is_open((x, y), "W"):
            neighbors.append((x - 1, y))

        for neighbor in neighbors: 
            tentative_g = current_cost + 1 # Assuming each step has a cost of 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]: 
                g_score[neighbor] = tentative_g
                f_score = tentative_g + manhattan_distance(neighbor, goal)
                heapq.heappush(open_list, (f_score, tentative_g, neighbor)) 
                came_from[neighbor] = current

    # Reconstruct path
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from.get(node)
        if node is None:
            print("No path found.")
            return
    path.reverse()
    tac = time.perf_counter() # Stop the timer
    time_elapsed = tac - tic

    # Follow the path in the environment
    current = tuple(env.maze_view.robot)
    for next_pos in path:
        dx = next_pos[0] - current[0]
        dy = next_pos[1] - current[1]

        if dx == 1:
            env.maze_view.move_robot("E")
        elif dx == -1:
            env.maze_view.move_robot("W")
        elif dy == 1:
            env.maze_view.move_robot("S")
        elif dy == -1:
            env.maze_view.move_robot("N")

        current = next_pos
        if render:
            env.render()

    print(f"Path length: {len(path)}")
    print(f"States visited: {len(visited)}")
    print(f"Execution time: {time_elapsed:0.4f} seconds")
    return path, visited, time_elapsed

def plot_results(paths, visited_nodes, times):
    # Create results folder if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Create a single figure with subplots for all plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Reward convergence
    axs[0, 0].plot(range(len(paths)), [len(path) for path in paths])
    axs[0, 0].set_xlabel('Scenario')
    axs[0, 0].set_ylabel('Path Length')
    axs[0, 0].set_title("Path Length per Scenario")
    axs[0, 0].grid(True)

    # Steps per episode
    axs[0, 1].plot(range(len(times)), times)
    axs[0, 1].set_xlabel('Scenario')
    axs[0, 1].set_ylabel('Seconds')
    axs[0, 1].set_title('Time to solve per Scenario')
    axs[0, 1].grid(True)

    # Explore rate over time
    axs[1, 0].plot(range(len(visited_nodes)), [len(visited) for visited in visited_nodes])
    axs[1, 0].set_xlabel('Scenario')
    axs[1, 0].set_ylabel('Visited Nodes')
    axs[1, 0].set_title('Visited Nodes per Scenario')
    axs[1, 0].grid(True)


    # Adjust layout and save the combined plot
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "combined_plots_a_star.png"))
    plt.close()

if __name__ == "__main__":
    # env = gym.make("maze-random-10x10-v0")

    times = []
    paths = []
    visiteds = []

    for i in range(100):
        env = gym.make("maze-random-10x10-v0")
        # env = MazeEnvRandom10x10(enable_render=False)  # Create the environment once
        result = a_star_solver(env, render=False)
        if result is not None:
            path, visited, time_elapsed = result
            paths.append(path)
            visiteds.append(visited)
            times.append(time_elapsed)
        else:
            print(f"Scenario {i}: No path found.")

    plot_results(paths, visiteds, times)
