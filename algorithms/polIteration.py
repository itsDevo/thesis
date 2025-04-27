def polIteration(maze):
    # transition probability = 1 - movement is deterministic, gamma = 0.9, R = s' == goal? 1 : 0
    # max of the R(s, a, s') + gamma*V(s') [use previous steps V(s')] for each of the four directions - n, s, e, w
    # if not open, V = 0
    # for each state I need the v value and the related next state

    sizeX, sizeY = maze.maze_size
    goalX, goalY = maze.maze_view.goal
    # initialise the policy to be always go "E" and V to 0 just to allocate the space
    V = [[0] * sizeX for _ in range(sizeY)]
    V[goalX][goalY] = 1
    dir = [["E"] * sizeX for _ in range(sizeY)]
    gamma = 0.95
    goalDelta = 0.000000000001
    count = 0 # iteration 0 is done by setting V goal to 1 ??? right ?? -> true for val
    polNotConvergerd = True
    n_bellman_calls = 0
    n_state_passes = 0

    def calcV(xSprime, ySprime):
        r = 0.01 if goalX == xSprime and goalY == ySprime else 0 
        return r + gamma * Vprev[xSprime][ySprime]
        
    while(polNotConvergerd):
        valNotConvergerd = True
        count += 1
        # calculate the v for each state just based on "E"
        # then val iterate but only with the dir from pol
        while(valNotConvergerd): # repeate until it converges sufficiently
            Vprev = [row[:] for row in V] # keep the last steps state values
            delta = 0
            for x in range(sizeX):
                for y in range(sizeY):
                    n_state_passes += 1

                    # don't update V for goal state
                    if x == goalX and y == goalY: 
                        continue

                    match dir[x][y]:
                        case "N":
                            if maze.maze_view.maze.is_open((x, y), "N"):
                                val = calcV(x, y-1)
                                n_bellman_calls += 1
                            else:
                                val = 0
                            delta = max(delta, abs(val - V[x][y]))
                            V[x][y] = val
                        case "E":
                            if maze.maze_view.maze.is_open((x, y), "E"):
                                val = calcV(x+1, y)
                                n_bellman_calls += 1
                            else:
                                val = 0
                            delta = max(delta, abs(val - V[x][y]))
                            V[x][y] = val
                        case "S":
                            if maze.maze_view.maze.is_open((x, y), "S"):
                                val = calcV(x, y+1)
                                n_bellman_calls += 1
                            else:
                                val = 0
                            delta = max(delta, abs(val - V[x][y]))
                            V[x][y] = val
                        case "W":
                            if maze.maze_view.maze.is_open((x, y), "W"):
                                val = calcV(x-1, y)
                                n_bellman_calls += 1
                            else:
                                val = 0
                            delta = max(delta, abs(val - V[x][y]))
                            V[x][y] = val
                        case _:
                            print("ERROR: Not a valid direction")

        # calculate the v for each state -> val iterate but only with the dir from pol

                    # ^ calulates V using the bellman equation, for the S' 
            if delta < goalDelta:
                valNotConvergerd = False
    
        # then update pol to point at max bellman based on V
        Vprev = [row[:] for row in V] # keep the last steps state values
        polChanged = False
        for x in range(sizeX):
            for y in range(sizeY):
                n_state_passes += 1

                # don't update V for goal state
                if x == goalX and y == goalY: 
                    continue

                Vmax = 0  # same as initial
                dirVmax = "E"  # same as initial
                
                if maze.maze_view.maze.is_open((x, y), "N"):
                    val = calcV(x, y-1)
                    n_bellman_calls += 1
                    if val > Vmax:
                        Vmax = val
                        dirVmax = "N"
                if maze.maze_view.maze.is_open((x, y), "E"):
                    val = calcV(x+1, y)
                    n_bellman_calls += 1
                    if val > Vmax:
                        Vmax = val
                        dirVmax = "E"
                if maze.maze_view.maze.is_open((x, y), "S"):
                    val = calcV(x, y+1)
                    n_bellman_calls += 1
                    if val > Vmax:
                        Vmax = val
                        dirVmax = "S"
                if maze.maze_view.maze.is_open((x, y), "W"):
                    val = calcV(x-1, y)
                    n_bellman_calls += 1
                    if val > Vmax:
                        Vmax = val
                        dirVmax = "W"
                
                if dir[x][y] != dirVmax:
                    polChanged = True
                dir[x][y] = dirVmax

        # then compare policy to prevPol
        # if changed, repeat
        if not polChanged:
            polNotConvergerd = False

    print(count, " iterations")
    print(n_bellman_calls, " Bellmans  calcs")
    print(n_state_passes, " passes over each state in the mazes")
    
    # traverse given route on maze
    done = False
    path_len = 0
    while not done:
        x, y = maze.maze_view.robot
        done = maze.step(dir[x][y])
        maze.render()
        path_len += 1
    print("path len ", path_len)
    return count, path_len