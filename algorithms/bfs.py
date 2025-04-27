def bfs(maze):
    seen = []
    visited = set()
    done = False

    # at each spot, add score check if done == true
    while not done:
        x, y = maze.maze_view.robot
        visited.add((x, y))

        # append every visible entry that hasn't already been visited to seen
        if maze.maze_view.maze.is_open((x, y), "N"):
            if (x, y-1) not in visited and (x, y-1) not in seen:
                seen.append((x, y-1))
        if maze.maze_view.maze.is_open((x, y), "E"):
            if (x+1, y) not in visited and (x+1, y) not in seen:
                seen.append((x+1, y))
        if maze.maze_view.maze.is_open((x, y), "S"):
            if (x, y+1) not in visited and (x, y+1) not in seen:
                seen.append((x, y+1))
        if maze.maze_view.maze.is_open((x, y), "W"):
            if (x-1, y) not in visited and (x-1, y) not in seen:
                seen.append((x-1, y))

        # calc score and go to spot popped from seen
        x, y = seen.pop(0)
        done = maze.hop(x, y)

        maze.render()
    
    n_states_visited = len(visited) # + 1 # +1 for the final state you visit ? or do we onlycount moves? in which case, no +1
    n_states_seen = len(visited) + len(seen)
    print("path length: ", n_states_visited)
    print("num states examined: ", n_states_seen)
    return n_states_visited, n_states_seen
