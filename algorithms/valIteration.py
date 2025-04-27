def valIteration(maze):
    # transition probability = 1 - movement is deterministic, gamma = 0.9, R = s' == goal? 1 : 0
    # max of the R(s, a, s') + gamma*V(s') [use previous steps V(s')] for each of the four directions - n, s, e, w
    # if not open, V = 0
    # for each state I need the v value and the related next state

    sizeX, sizeY = maze.maze_size
    goalX, goalY = maze.maze_view.goal
    # initialise the state to be V=0 with a always "E" just to allocate the space
    V = [[0] * sizeY for _ in range(sizeX)]
    V[goalX][goalY] = 1
    dir = [["E"] * sizeX for _ in range(sizeY)]
    gamma = 0.9
    goalDelta = 0.00000000001 # threshold for conversion
    count = 1
    notConvergerd = True
    n_bellman_calls = 0
    n_state_passes = 0

    def calcV(xSprime, ySprime):
        r = 0.1 if goalX == xSprime and goalY == ySprime else 0 
        return r + gamma * Vprev[xSprime][ySprime]

    while(notConvergerd): # repeat until it converges sufficiently
        count += 1
        Vprev = [row[:] for row in V] # keep the last steps state values
        maxDelta = 0
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
                
                maxDelta = max(abs(Vmax - V[x][y]), maxDelta)
                V[x][y] = Vmax
                dir[x][y] = dirVmax

        if maxDelta < goalDelta:
            notConvergerd = False

    print(count, " iterations")
    print(n_bellman_calls, " Bellmans calcs")
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
