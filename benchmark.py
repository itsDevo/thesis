from q_learning import q_solver
from aStar import a_star_solver
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import gym
import gym_maze
import psutil
from contextlib import contextmanager

def q_plot_results(size, rewards_per_episode, steps_per_episode, shortest_path, explore_rates,q_table,time_for_streak, visited_states_per_episode, time_per_episode, cpu,memory,mode=0):

    # Create results folder if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    avg_cpu = np.mean([np.mean(cpu) for cpu in cpu])
    std_cpu = np.std([np.mean(cpu) for cpu in cpu])
    avg_memory = np.mean([np.mean(memory) for memory in memory])
    std_memory = np.std([np.mean(memory) for memory in memory])


    if mode == 0: # Plotting for a single run
        episodes = range(len(rewards_per_episode))
        # Create a single figure with subplots for all plots
        fig, axs = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle("Q-Learning Performance on 50 Random 5x5 Mazes", fontsize=16)
        # Reward convergence
        axs[0, 0].plot(episodes, rewards_per_episode)
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Total Reward')
        axs[0, 0].set_title('Reward Convergence')
        axs[0, 0].grid(True)

        # Steps per episode
        axs[0, 1].plot(episodes, steps_per_episode)
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Steps to Solve')
        axs[0, 1].set_title('Steps per Episode')
        axs[0, 1].grid(True)

        # Explore rate over time
        axs[1, 0].plot(episodes, explore_rates)
        axs[1, 0].set_xlabel('Episode')
        axs[1, 0].set_ylabel('Explore Rate')
        axs[1, 0].set_title('Exploration vs. Exploitation')
        axs[1, 0].grid(True)

        # Success rate processing
        success_rate = [1 if r > 0 else 0 for r in rewards_per_episode]  # Define success
        window_size = 100
        avg_success_rate = np.convolve(success_rate, np.ones(window_size)/window_size, mode='valid')

        # Plot both raw and smoothed success rate
        axs[1, 1].plot(success_rate, alpha=0.7, label="Raw Success (0/1)", linewidth=1)
        axs[1, 1].plot(avg_success_rate, label=f"Rolling Avg ({window_size})", linewidth=2)
        axs[1, 1].set_xlabel("Episode")
        axs[1, 1].set_ylabel("Success Rate")
        axs[1, 1].set_title("Success Rate Over Episodes")
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        axs[2, 0].plot(episodes, visited_states_per_episode)
        axs[2, 0].set_title("Unique States Visited per Episode")
        axs[2, 0].set_xlabel("Episode")
        axs[2, 0].set_ylabel("States Visited")
        axs[2, 0].grid(True)

        axs[2, 1].axis('off')  # Hide the last subplot
        axs[2, 1].text(0.5, 0.5,
                    f"Time Elapsed: {time_for_streak:.2f} seconds\n"
                    f"Total Episodes: {len(rewards_per_episode)}\n"
                    f"Average usage of CPU: {avg_cpu:.2f} % ±{std_cpu:.2f}\n"
                    f"Average usage of Memory: {avg_memory:.2f} MB ±{std_memory:.2f}\n",
                    fontsize=12, ha='center', va='center')
        
    elif mode == 1: # Plotting for loop runs
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Q-Learning Performance on 50 Random {size} Mazes", fontsize=16)

        avg_steps_per_run = np.array([np.nanmean(steps) for steps in steps_per_episode])
        avg_visited_per_run = np.array([np.nanmean(visited) for visited in visited_states_per_episode])
        avg_time_per_run = np.array([np.nanmean(times) for times in time_per_episode])

        scenarios = range(len(shortest_path))

        steps_std = np.std(avg_steps_per_run)
        visited_std = np.std(avg_visited_per_run)
        time_std = np.std(avg_time_per_run)

        # Plotting
        axs[0, 0].plot(scenarios, shortest_path)
        axs[0, 0].set_xlabel('Scenario')
        axs[0, 0].set_ylabel('Path Length')
        axs[0, 0].set_title('Path Length per Scenario')
        axs[0, 0].grid(True)

        axs[0, 1].plot(scenarios, avg_time_per_run, label='Avg Time per Episode')
        axs[0, 1].set_xlabel('Scenario')
        axs[0, 1].set_ylabel('Time (Seconds)')
        axs[0, 1].set_title('Average Time to Solve per Scenario')
        axs[0, 1].grid(True)

        axs[1, 0].plot(scenarios, avg_visited_per_run, label='Unique States Visited')
        axs[1, 0].set_xlabel('Scenario')
        axs[1, 0].set_ylabel('Unique Visited States')
        axs[1, 0].set_title('Average Unique Visited States per Scenario')
        axs[1, 0].grid(True)

        axs[1, 1].axis('off')  # Hide the last subplot
        axs[1, 1].text(0.5, 0.5,
                       f"Avg Path Length: {np.nanmean(shortest_path):.2f} ± {np.nanstd(shortest_path):.2f}\n"
                       f"Avg Unique States Visited: {np.nanmean(avg_visited_per_run):.2f} ± {visited_std:.2f}\n"
                       f"Avg Time to Solve: {np.nanmean(avg_time_per_run):.4f} sec ± {time_std:.4f}\n"
                       f"Avg Steps: {np.nanmean(avg_steps_per_run):.2f} ± {steps_std:.2f}\n"
                       f"\n"
                        f"Avg CPU Usage: {avg_cpu:.2f} % ±{std_cpu:.2f}\n"
                        f"Avg Memory Usage: {avg_memory:.2f} MB ±{std_memory:.2f}\n",
                       fontsize=12, ha='center', va='center')
        axs[1, 1].set_title("Statistics")

    # Adjust layout and save the combined plot
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"q_learning_{size}.png"))
    plt.close()

def a_star_plot_results(size,paths, visited_nodes, times, cpu, memory):
    # Create results folder if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    avg_cpu = np.mean([np.mean(cpu) for cpu in cpu])
    std_cpu = np.std([np.mean(cpu) for cpu in cpu])
    avg_memory = np.mean([np.mean(memory) for memory in memory])
    std_memory = np.std([np.mean(memory) for memory in memory])

    path_lengths = np.array([len(path) for path in paths])
    visited_counts = np.array([len(visited) for visited in visited_nodes])
    times_array = np.array(times)

    avg_path_length = np.mean(path_lengths)
    std_path_length = np.std(path_lengths)

    avg_visited_nodes = np.mean(visited_counts)
    std_visited_nodes = np.std(visited_counts)

    avg_time = np.mean(times_array)
    std_time = np.std(times_array)

    # Create a single figure with subplots for all plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"A* Performance on 50 Random {size} Mazes", fontsize=16)

    # Path Length
    axs[0, 0].plot(range(len(paths)), path_lengths, label='Path Length')
    axs[0, 0].set_xlabel('Scenario')
    axs[0, 0].set_ylabel('Path Length')
    axs[0, 0].set_title("Path Length per Scenario")
    axs[0, 0].grid(True)

    # Steps per episode
    axs[0, 1].plot(range(len(times)), times_array, label='Time to Solve')
    axs[0, 1].set_xlabel('Scenario')
    axs[0, 1].set_ylabel('Time (Seconds)')
    axs[0, 1].set_title('Time to solve per Scenario')
    axs[0, 1].grid(True)

    # Explore rate over time
    axs[1, 0].plot(range(len(visited_nodes)), visited_counts, label='Visited Nodes')
    axs[1, 0].set_xlabel('Scenario')
    axs[1, 0].set_ylabel('Visited Nodes')
    axs[1, 0].set_title('Unique Visited States per Scenario')
    axs[1, 0].grid(True)

    # KPI Summary
    axs[1,1].axis('off')  # Hide the last subplot
    axs[1,1].text(0.5, 0.5, f"Avg Path Length: {avg_path_length:.2f} ± {std_path_length:.2f}\n"
                            f"Avg Unique Visited States: {avg_visited_nodes:.2f} ± {std_visited_nodes:.2f}\n"
                            f"Avg Time to Solve: {avg_time:.4f} seconds ± {std_time:.4f}\n"
                            f"\n"
                            f"Avg CPU Usage: {avg_cpu:.2f} % ± {std_cpu:.2f}\n"
                            f"Avg Memory Usage: {avg_memory:.2f} MB ± {std_memory:.2f}\n",
                fontsize=12, ha='center', va='center')
    axs[1,1].set_title("Statistics")


    # Adjust layout and save the combined plot
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"a_star_{size}.png"))
    plt.close()

def pad_to_max_length(data, pad_value=np.nan):
    max_len = max(len(seq) for seq in data)
    return np.array([
        np.pad(np.array(seq, dtype=float), (0, max_len - len(seq)), constant_values=pad_value)
        for seq in data
    ], dtype=float)

@contextmanager
def resource_monitoring(label="Task"):
    process = psutil.Process(os.getpid())
    cpu_before = process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 ** 2)

    stats = {}
    try:
        yield stats
    finally:
        cpu_after = process.cpu_percent(interval=None)
        mem_after = process.memory_info().rss / (1024 ** 2)

        stats['cpu_start'] = cpu_before
        stats['cpu_end'] = cpu_after
        stats['mem_start'] = mem_before
        stats['mem_end'] = mem_after

        print(f"\n--- Resource Monitoring for {label} ---")
        print(f"CPU Usage Start: {cpu_before:.2f}%, End: {cpu_after:.2f}%")
        print(f"Memory Usage Start: {mem_before:.2f} MB, End: {mem_after:.2f} MB")
        print(f"----------------------------------------\n")

if __name__ == "__main__":

    all_rewards = []
    all_steps = []
    all_explore = []
    all_visited = []
    all_time_for_streak = []
    all_time_per_episode = []
    all_shortest_path = []

    time_lst = []
    path_lst = []
    visited_lst = []

    a_star_cpu = []
    a_star_mem = []
    q_cpu = []
    q_mem = []

    mazes = ["3x3", "5x5", "10x10"]
    # mazes = ["10x10-plus", "20x20-plus", "30x30-plus"]

    for size in mazes:
        for i in range(50):
            env = gym.make("maze-random-" +size+ "-v0")
        
            with resource_monitoring(f"A* Solver - Scenario") as stats:   
                results_a = a_star_solver(env, render=False)
            
            paths, visited_nodes, times = results_a
            path_lst.append(paths)
            visited_lst.append(visited_nodes)
            time_lst.append(times)
            a_star_cpu.append(stats['cpu_end'] - stats['cpu_start'])
            a_star_mem.append(stats['mem_end'] - stats['mem_start'])
            
            with resource_monitoring(f"Q-Learning Solver - Scenario") as stats:
                results_q =  q_solver(env, verbose=0, render=False)
            
            rewards, steps, explore_rates,visited_states, q_table, time_for_streak, time_per_episode = results_q   
            all_rewards.append(rewards)
            all_steps.append(steps)
            all_shortest_path.append(steps[-1])
            all_explore.append(explore_rates)
            all_visited.append(visited_states)
            all_time_for_streak.append(time_for_streak)
            all_time_per_episode.append(time_per_episode)
            q_cpu.append(stats['cpu_end'] - stats['cpu_start'])
            q_mem.append(stats['mem_end'] - stats['mem_start'])

        q_plot_results(size,all_rewards, all_steps, all_shortest_path, all_explore, q_table, all_time_for_streak, all_visited, all_time_per_episode,q_cpu,q_mem, mode=1)
        a_star_plot_results(size,path_lst, visited_lst, time_lst, a_star_cpu, a_star_mem)
        all_rewards = []
        all_steps = []
        all_explore = []
        all_visited = []
        all_time_for_streak = []
        all_time_per_episode = []
        all_shortest_path = []


        time_lst = []
        path_lst = []
        visited_lst = []

        a_star_cpu = []
        a_star_mem = []
        q_cpu = []
        q_mem = []