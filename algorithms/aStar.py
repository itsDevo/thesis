import heapq
import gym
import gym_maze

def aStar(maze):
    seen = []
    seen_set = set()
    visited = set()
    done = False
    cost = 0
    xGoal, yGoal = maze.maze_view.goal
    breaker = 20

    # at each, check if done == true
    while not done:
        x, y = maze.maze_view.robot
        visited.add((x, y))

        # get available new neighbours, set their cost to current + 1,
        # calculate their f by adding manhatan distance to end to the cost
        # store coordinates, cost, f in a priority queue sorted by f
        fBasic = abs(xGoal-x) + abs(yGoal-y) + cost + 1  # manhattan distance to current
        if maze.maze_view.maze.is_open((x, y), "N"):
            if (x, y-1) not in visited and not (x, y-1) in seen_set:
                heapq.heappush(seen, (fBasic+1, (x, y-1), cost+1))
                seen_set.add((x, y-1))
        if maze.maze_view.maze.is_open((x, y), "E"):
            if (x+1, y) not in visited and not (x+1, y) in seen_set:
                heapq.heappush(seen, (fBasic-1, (x+1, y), cost+1))
                seen_set.add((x+1, y))
        if maze.maze_view.maze.is_open((x, y), "S"):
            if (x, y+1) not in visited and not (x, y+1) in seen_set:
                heapq.heappush(seen, (fBasic-1, (x, y+1), cost+1))
                seen_set.add((x, y+1))
        if maze.maze_view.maze.is_open((x, y), "W"):
            if (x-1, y) not in visited and not (x-1, y) in seen_set:
                heapq.heappush(seen, (fBasic+1, (x-1, y), cost+1))
                seen_set.add((x-1, y))

        # pop state with next lowest f and set cost to its cost
        _, (x, y), cost = heapq.heappop(seen)
        done = maze.hop(x, y)
        maze.render()

    n_states_visited = len(visited) # + 1 # +1 for the final state you visit ? or do we onlycount moves? in which case, no +1
    n_states_seen = len(visited) + len(seen)
    print("path length: ", n_states_visited)
    print("num states examined: ", n_states_seen)
    return n_states_visited, n_states_seen


if __name__ == "__main__":
    env = gym.make("maze-sample-5x5-v0")
    aStar(env)