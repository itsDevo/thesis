import random
import numpy as np
import gym
import gym_maze
import matplotlib.pyplot as plt
import sys
import os

def run(episodes, is_training=True,verbose=0,render=False):

    env = gym.make("maze-sample-5x5-v0") ### I need to put the environment here


    LEARNING_RATE = 0.9 #alpha value
    DISCOUNT_FACTOR = 0.1 #gamma value 
    EPSILON = 1 # 1 = 100% random actions
    EPSILON_DECAY_RATE = 0.0001 # Since we are gonna have 10000 episodes at the last episode it will become 1

    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    SOLVED_THRESHOLD = np.prod(MAZE_SIZE, dtype=int) # The number of steps to solve the maze to count

    NUM_BUCKETS = MAZE_SIZE #one bucket per grid
    NUM_ACTIONS = env.action_space.n # 4 actions (left, down, up, right)

    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high)) # Bounds of the state space

    NUM_EPISODES = episodes # Number of episodes to train the agent
    MAX_STEPS = np.prod(MAZE_SIZE, dtype=int) * 100 # Maximum number of steps per episode
    STREAK = 100 # Number of episodes to check for convergence

    num_streaks = 0 # Number of streaks of 100 episodes with no improvement in the average reward
    rewards_per_episode = [] # List to store the rewards per episode
    steps_per_episode = [] # List to store the number of steps per episode
    explore_rates = [] # List to store the explore rates per episode

    for episode in range(NUM_EPISODES):
        obv = env.reset()

        state_0 = state_to_bucket(obv, STATE_BOUNDS, NUM_BUCKETS) # Here we get the state of the environment (the position of the agent in the maze)
        done = False
        total_reward = 0
        # truncated = False # There is no truncated in the step function


        for step in range(MAX_STEPS):

            while (not done): # Here we check if the episode is done or not (if the agent has reached the goal or not)
                if is_training and random.random() < EPSILON: # here we make it a random move if the epsilon is high (we make it learn)
                    action = env.action_space.sample() # It takes the action (left, down, up, right)
                else:
                    action = np.argmax(q_table[state_0]) # here we make it to exploit from it's previous learning


                obv,reward,done,info = env.step(action) 

                state = state_to_bucket(obv, STATE_BOUNDS, NUM_BUCKETS) # Here we get the new state of the environment (the position of the agent in the maze)
                total_reward += reward # Here we get the reward of the action taken (the position of the agent in the maze)

                best_q = np.amax(q_table[state]) # Here we get the best q value of the new state (the position of the agent in the maze)
                if is_training: # we apply the q function
                    q_table[state_0 + (action,)] += LEARNING_RATE * (reward + DISCOUNT_FACTOR * (best_q) - q_table[state_0 + (action,)])

                state_0 = state

                    # Print data
                if verbose == 2:
                    print("\nEpisode = %d" % episode)
                    print("step = %d" % step)
                    print("Action: %d" % action)
                    print("State: %s" % str(state))
                    print("Reward: %f" % reward)
                    print("Best Q: %f" % best_q)
                    print("Explore rate: %f" % EPSILON)
                    print("Learning rate: %f" % LEARNING_RATE)
                    print("Streaks: %d" % num_streaks)
                    print("")

                elif verbose == 1: 
                    if done or step >= MAX_STEPS - 1:
                        print("\nEpisode = %d" % episode)
                        print("step = %d" % step)
                        print("Explore rate: %f" % EPSILON)
                        print("Learning rate: %f" % LEARNING_RATE)
                        print("Streaks: %d" % num_streaks)
                        print("Total reward: %f" % total_reward)
                        print("")


            if render:
                env.render()

            if env.is_game_over():
                sys.exit()


            if done:
                print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                      % (episode, step, total_reward, num_streaks))

                if step <= SOLVED_THRESHOLD:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break    

            elif step >= MAX_STEPS - 1:
                print("Episode %d timed out at %d with total reward = %f."
                      % (episode, step, total_reward))

        # if num_streaks >= STREAK:
        #     break

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(step + 1)
        explore_rates.append(EPSILON)

        EPSILON = max(EPSILON - EPSILON_DECAY_RATE, 0) # decrease the epsilon. we want it to exploit as it goes further

        if(EPSILON == 0):
            LEARNING_RATE = 0.0001
    
    env.close()
    plot_results(rewards_per_episode, steps_per_episode, explore_rates)
    


def state_to_bucket(state, STATE_BOUNDS, NUM_BUCKETS):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


def plot_results(rewards_per_episode, steps_per_episode, explore_rates):
    # Create results folder if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    episodes = range(len(rewards_per_episode))
    # Create a single figure with subplots for all plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

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
    avg_success_rate = np.convolve(success_rate, np.ones(window_size)/window_size, mode='same')

    # Plot both raw and smoothed success rate
    axs[1, 1].plot(success_rate, alpha=0.3, label="Raw Success (0/1)", linewidth=1)
    axs[1, 1].plot(avg_success_rate, label=f"Rolling Avg ({window_size})", linewidth=2)
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Success Rate")
    axs[1, 1].set_title("Success Rate Over Episodes")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Adjust layout and save the combined plot
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "combined_plots.png"))
    plt.close()


if __name__ == "__main__":
    run(5000, is_training=True, verbose=0, render=False)