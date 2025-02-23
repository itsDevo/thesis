import random #will be used for maze generator

def ran_prim(maze):
    
    #len(maze[0]) -> width, len(maze) -> height
    WIDTH = len(maze[0])
    HEIGHT = len(maze)

    x ,y = random.randint(0 , WIDTH - 1), random.randint(0, HEIGHT -1) 
    maze[x][y] = 0

    walls = []
    directions = [(0,1),(0,-1),(1,0),(-1,0)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0<= nx < WIDTH and 0<= ny < HEIGHT:
            walls.append((nx, ny, x, y))

    while walls:
        wall_index = random.randint(0, len(walls) - 1) #Randomly selects a wall
        wx, wy, px, py = walls.pop(wall_index)

        if maze[wy][wx] == 1: #Checks the cell on the opposite side of the wall is still a wall or not
            ox, oy = wx + (wx - px), wx - (wy - py) #Calculates the opposite side of the wall

            if 0<= ox < WIDTH and 0<= oy < HEIGHT: #to create the path
                maze[wy][wx] = 0
                maze[oy][ox] = 0

                for dx, dy in directions:
                    nx, ny = ox + dx, oy + dy
                    if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and maze[ny][nx] == 1:
                        walls.append((nx, ny, ox, oy))

    return maze 



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

    # recursive_backtracking(maze)
    ran_prim(maze)

    if maze[-2][-1] == 1 and maze[-1][-2] == 1: # Check if the goal is not blocked
        maze[-2][-1] = 0

    return maze


if __name__ == "__main__":

    maze = maze_initializer(6)
    for _ in maze:
        print(_)
