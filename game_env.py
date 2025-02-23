import random #will be used for maze generator

def recursive_backtracking(maze, x=0, y=0):
    directions = [(0,1),(0,-1),(1,0),(-1,0)]
    random.shuffle(directions) #To randomize the direction

    for dx,dy in directions:
        nx, ny = x + dx*2 , y + dy*2  #We multiply it by 2 to avoid 2 following wall blocks.
        if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and maze[nx][ny] == 1:
            maze[nx][ny] = 0 # to carve the path
            maze[x + dx][y + dy] = 0 # to remove all the walls betweeen current and new cell
            recursive_backtracking(maze,nx,ny)
    return maze

# 'S' is the start point, 0 is empty space, 1 is a wall(Block), G is the endpoint.
def maze_initializer(n=4):
    maze = [[1 for _ in range(n)] for _ in range(n)]
    maze[0][0] = 'S'
    maze[n-1][n-1] = 'G'

    recursive_backtracking(maze)

    if maze[-2][-1] == 1 and maze[-1][-2] == 1: # Check if the goal is not blocked
        maze[-2][-1] = 0

    return maze
