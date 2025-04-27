def dfs(maze):
    seen = [] # list to maintain order, used as queue (FIFO)
    seen_cost = [] # list to keep track of distance to each state on the seen queue
    visited = set() # set instead of list as order is irrelevant and lookups will be O(1)
    done = False

    # repeat until robot reaches the goal
    while not done:
        x, y = maze.maze_view.robot
        visited.add((x, y))

        # append every visible entry that hasn't already been visited to seen
        # the coordinate changes (the +1/-1's) are corresponding to change based on direction
        if maze.maze_view.maze.is_open((x, y), "N"):
            if (x, y-1) not in visited:
                seen.append((x, y-1))
        if maze.maze_view.maze.is_open((x, y), "E"):
            if (x+1, y) not in visited:
                seen.append((x+1, y))
        if maze.maze_view.maze.is_open((x, y), "S"):
            if (x, y+1) not in visited:
                seen.append((x, y+1))
        if maze.maze_view.maze.is_open((x, y), "W"):
            if (x-1, y) not in visited:
                seen.append((x-1, y))

        # go to next spot popped from seen (ie. last neighbour seen)
        try:
            x, y = seen.pop()
        except:
            # if all spots seen have been visited the maze is unsolvable
            print("ERROR: Maze unsolveable")
            break

        done = maze.hop(x, y)

        maze.render()

    n_states_visited = len(visited) # + 1 # +1 for the final state you visit ? or do we onlycount moves? in which case, no +1
    n_states_seen = len(visited) + len(seen)
    print("path length: ", n_states_visited)
    print("num states examined: ", n_states_seen)
    return n_states_visited, n_states_seen
