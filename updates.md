# Updates

## 23.02
Maze generator has a very simple shape and it follows the same pattern for different sizes.\
Try to find another maze generator algorithm. 
randomized Prim's algorithm has been added. *DONE*


## 06.03
Check if it is possible to write epsilon_decay_rate as 1/episode so we can have dynamic value and we won't need to change this after we change the episode number.
Now the episode number is a parameter for the function so it can change but the decay rate is fixed so it might be confusing sometimes.

## 26.03
There is an issue with the ready environment
The issue is I can't use it outside of it's folder
I must figure out how does the __init__ files work so I can register the environments in the gym
Otherwise It won't recognize it. 

## 16.04
The code is done!
Now I need to find the implementations for example: To create a streak function
Plus I need to implement other algorithms