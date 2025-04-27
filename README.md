This thesis will focus on comparing A-star algorithm against Q-learning in terms of path finding. Q-learning is a reinforcement learning algorithm which is a subsection of machine learning.
The goal of the thesis is to analyise the viability of using reinforcement learning in path finding problems instead of traditional algorithms and discover the strenghts and weaknesses of each method and possibile future developments.

In this thesis, I will be using a custom environment of GYM.
The environment includes 3 type of mazes (Pre-generated, Randomly Generated and Randomly generated mazes with portals and loops.) and 4 different sizes (3x3, 5x5, 10x10 and 100x100).

The results based on 50 different mazes for both algorithms (Same maze for both algorithms per loop).
In general aStar alogrithm performs well.
However Q-Learning method doesn't perform well on 100x100 since it requires a lot of resource and time.
