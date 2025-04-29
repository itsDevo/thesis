import random
import numpy as np
import gym
import gym_maze
import sys
import math
import time

def q_solver(env,verbose=0,render=False):

    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))

    if env.spec.id == "maze-sample-3x3-v0" or env.spec.id =="maze-random-3x3-v0":
        LEARNING_RATE = 0.7  # Moderate learning rate for balanced updates
        DISCOUNT_FACTOR = 0.8  # Higher discount factor to value future rewards
        EPSILON = 0.9  # Initial exploration rate, slightly less random actions
        DECAY_FACTOR = (np.prod(MAZE_SIZE, dtype=float) - 5) / 10  # Balanced decay for exploration
        NUM_EPISODES = 1200  # Sufficient episodes for convergence without excessive runtime
        MIN_EXPLORE_RATE = 0.001  # Minimum exploration rate
        MIN_LEARNING_RATE = 0.2  # Minimum learning rate

    elif env.spec.id == "maze-sample-5x5-v0" or env.spec.id =="maze-random-5x5-v0":
        LEARNING_RATE = 0.6  # Moderate learning rate for balanced updates
        DISCOUNT_FACTOR = 0.8  # Higher discount factor to value future rewards
        EPSILON = 0.7  # Initial exploration rate, less random actions
        DECAY_FACTOR = (np.prod(MAZE_SIZE, dtype=float) - 10) / 15  # Balanced decay for exploration
        NUM_EPISODES = 1500  # Sufficient episodes for convergence without excessive runtime
        MIN_EXPLORE_RATE = 0.001  # Minimum exploration rate
        MIN_LEARNING_RATE = 0.2  # Minimum learning rate

    elif env.spec.id == "maze-sample-10x10-v0" or env.spec.id =="maze-random-10x10-v0" or env.spec.id =="maze-random-10x10-plus-v0":
        LEARNING_RATE = 0.3  # Moderate learning rate for balanced updates
        DISCOUNT_FACTOR = 0.9  # High discount factor to value future rewards
        EPSILON = 0.7  # Initial exploration rate, less random actions
        DECAY_FACTOR = (np.prod(MAZE_SIZE, dtype=float) - 1) / 50  # Faster decay for exploration
        NUM_EPISODES = 2000  # Reduced episodes for time efficiency
        MIN_EXPLORE_RATE = 0.001  # Minimum exploration rate
        MIN_LEARNING_RATE = 0.2  # Minimum learning rate

    elif env.spec.id == "maze-random-20x20-plus-v0":
        LEARNING_RATE = 0.2  # Moderate learning rate for balanced updates
        DISCOUNT_FACTOR = 0.9  # High discount factor to value long-term rewards
        EPSILON = 0.65  # Initial exploration rate, moderate randomness
        DECAY_FACTOR = (np.prod(MAZE_SIZE, dtype=float) - 1) / 30  # Balanced decay for exploration
        NUM_EPISODES = 3000  # Sufficient episodes for convergence
        MIN_EXPLORE_RATE = 0.01  # Minimum exploration rate
        MIN_LEARNING_RATE = 0.1  # Lower minimum learning rate for stability

    elif env.spec.id == "maze-random-30x30-plus-v0":
        LEARNING_RATE = 0.15  # Moderate learning rate for balanced updates
        DISCOUNT_FACTOR = 0.92  # High discount factor to value long-term rewards
        EPSILON = 0.6  # Initial exploration rate, moderate randomness
        DECAY_FACTOR = (np.prod(MAZE_SIZE, dtype=float) - 1) / 40  # Balanced decay for exploration
        NUM_EPISODES = 4000  # Sufficient episodes for convergence without excessive runtime
        MIN_EXPLORE_RATE = 0.005  # Minimum exploration rate for efficient exploration
        MIN_LEARNING_RATE = 0.1  # Lower minimum learning rate for stability
        
    elif env.spec.id == "maze-sample-100x100-v0" or env.spec.id =="maze-random-100x100-v0":
        LEARNING_RATE = 0.1  # Lower learning rate for stability
        DISCOUNT_FACTOR = 0.95  # High discount factor to value long-term rewards
        EPSILON = 0.8  # Moderate initial exploration rate
        DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 20  # Balanced decay for exploration
        NUM_EPISODES = 10000  # Reduced episodes for time efficiency
        MIN_EXPLORE_RATE = 0.01  # Slightly higher minimum exploration
        MIN_LEARNING_RATE = 0.1  # Lower minimum learning rate for stability

    SOLVED_THRESHOLD = np.prod(MAZE_SIZE, dtype=int) # The number of steps to solve the maze to count as a streak

    NUM_BUCKETS = MAZE_SIZE #one bucket per grid
    NUM_ACTIONS = env.action_space.n # 4 actions (left, down, up, right)

    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high)) # Bounds of the state space

    # NUM_EPISODES = episodes # Number of episodes to train the agent
    MAX_STEPS = np.prod(MAZE_SIZE, dtype=int) * 10 # Maximum number of steps per episode
    STREAK = 50 # Number of episodes to check for convergence

    num_streaks = 0 # Number of streaks of 100 episodes with no improvement in the average reward
    rewards_per_episode = [] # List to store the rewards per episode
    visited_states_per_episode = [] # List to store the visited states per episode
    steps_per_episode = [] # List to store the number of steps per episode
    explore_rates = [] # List to store the explore rates per episode
    time_per_episode = [] # List to store the time per episode

    tic = time.perf_counter() # Start the timer
    for episode in range(NUM_EPISODES):
        time_episode_start = time.time()
        obv = env.reset()

        state_0 = state_to_bucket(obv, STATE_BOUNDS, NUM_BUCKETS) # Here we get the state of the environment (the position of the agent in the maze)
        total_reward = 0
        visited_states = set()  # Set to store the visited states


        for step in range(MAX_STEPS):

            if random.random() < EPSILON: # here we make it a random move if the epsilon is high (we make it learn)
                action = env.action_space.sample() # It takes the action (left, down, up, right)
            else:
                action = np.argmax(q_table[state_0]) # here we make it to exploit from it's previous learning


            obv,reward,done,info = env.step(action) 

            state = state_to_bucket(obv, STATE_BOUNDS, NUM_BUCKETS) # Here we get the new state of the environment (the position of the agent in the maze)
            total_reward += reward # Here we get the reward of the action taken (the position of the agent in the maze)
            visited_states.add(state)  # Add the new state to the visited states
            best_q = np.amax(q_table[state]) # Here we get the best q value of the new state (the position of the agent in the maze) 
            q_table[state_0 + (action,)] += LEARNING_RATE * (reward + DISCOUNT_FACTOR * (best_q) - q_table[state_0 + (action,)]) # we apply the q function

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
                time_episode_end = time.time()
                estimated_time_per_episode = time_episode_end - time_episode_start
                time_per_episode.append(estimated_time_per_episode)

                print(f"(Q Learning) Estimated time for the episode : {estimated_time_per_episode}")
                print(f"(Q Learning) Episode {episode} finished after {step:.1f} steps with total reward = {total_reward} (streak {num_streaks}).")

                if step <= SOLVED_THRESHOLD:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break    

            elif step >= MAX_STEPS - 1:
                time_episode_end = time.time()
                estimated_time_per_episode = time_episode_end - time_episode_start
                time_per_episode.append(estimated_time_per_episode)

                print(f"(Q Learning) Estimated time for the episode: {estimated_time_per_episode}")
                print(f"(Q Learning) Episode {episode} timed out with {step:.1f} steps and total reward = {total_reward}.")

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(step + 1)
        visited_states_per_episode.append(len(visited_states))
        explore_rates.append(EPSILON)

        if num_streaks >= STREAK:
            break

        if env.spec.id == "maze-sample-3x3-v0" or env.spec.id =="maze-random-3x3-v0":
            EPSILON = max(MIN_EXPLORE_RATE, min(0.9, 1.0 - 0.7 * math.log10((episode+1)/DECAY_FACTOR)))  # Faster decay for small mazes
            LEARNING_RATE = max(MIN_LEARNING_RATE, min(0.7, 1.0 - 0.7 * math.log10((episode+1)/DECAY_FACTOR)))  # Higher learning rate for quick convergence
        
        elif env.spec.id == "maze-sample-5x5-v0" or env.spec.id =="maze-random-5x5-v0":
            EPSILON = max(MIN_EXPLORE_RATE, min(0.7, 1.0 - 0.6 * math.log10((episode+1)/DECAY_FACTOR)))  # Balanced decay
            LEARNING_RATE = max(MIN_LEARNING_RATE, min(0.6, 1.0 - 0.6 * math.log10((episode+1)/DECAY_FACTOR)))  # Balanced learning rate
        
        elif env.spec.id == "maze-sample-10x10-v0" or env.spec.id =="maze-random-10x10-v0":
            EPSILON = max(MIN_EXPLORE_RATE, min(0.7, 1.0 - 0.5 * math.log10((episode+1)/DECAY_FACTOR)))  # Slower decay for larger mazes
            LEARNING_RATE = max(MIN_LEARNING_RATE, min(0.3, 1.0 - 0.5 * math.log10((episode+1)/DECAY_FACTOR)))  # Lower learning rate for stability
        
        elif env.spec.id == "maze-random-20x20-plus-v0":
            EPSILON = max(MIN_EXPLORE_RATE, min(0.65, 1.0 - 0.4 * math.log10((episode+1)/DECAY_FACTOR)))  # Moderate decay for medium mazes
            LEARNING_RATE = max(MIN_LEARNING_RATE, min(0.2, 1.0 - 0.4 * math.log10((episode+1)/DECAY_FACTOR)))  # Balanced learning rate for medium mazes
        
        elif env.spec.id == "maze-random-30x30-plus-v0":
            EPSILON = max(MIN_EXPLORE_RATE, min(0.6, 1.0 - 0.3 * math.log10((episode+1)/DECAY_FACTOR)))  # Slower decay for larger mazes
            LEARNING_RATE = max(MIN_LEARNING_RATE, min(0.15, 1.0 - 0.3 * math.log10((episode+1)/DECAY_FACTOR)))  # Lower learning rate for stability in larger mazes
        
        elif env.spec.id == "maze-sample-100x100-v0" or env.spec.id =="maze-random-100x100-v0":
            EPSILON = max(MIN_EXPLORE_RATE, 1.0 / math.sqrt(episode+1))  # Very slow decay for massive mazes
            LEARNING_RATE = max(MIN_LEARNING_RATE, 1.0 / math.sqrt(episode+1))  # Very slow learning rate for stability


    env.close()
    tac = time.perf_counter() # Stop the timer
    time_to_finish_streak = tac - tic
    print(f"(Q Learning) Total time to finish the streak: {time_to_finish_streak:0.4f} seconds")

    return rewards_per_episode, steps_per_episode, explore_rates, visited_states_per_episode, q_table, time_to_finish_streak, time_per_episode
    


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

if __name__ == "__main__":
    env = gym.make("maze-random-3x3-v0")
    rewards_per_episode, steps_per_episode, explore_rates,visited_states_per_episode, q_table, time_to_finish_streak, time_per_episode =  q_solver(env, verbose=1, render=False)
    