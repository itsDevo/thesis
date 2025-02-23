# Goal of the Thesis
The goal of this thesis is to explore and compare the performance of Q-learning, a reinforcement learning algorithm, with traditional pathfinding algorithms (Dijkstra’s Algorithm and A*) in solving maze navigation problems. The mazes will increase in complexity across three stages (4x4, 6x6, and 8x8), featuring walls and dead ends to challenge the algorithms. The comparison will be based on three key metrics: time to solve, number of steps, and success rate. Additionally, Q-learning-specific metrics such as reward convergence and exploration vs. exploitation will be analyzed to understand the learning process. The project will also include a random maze generator to ensure robust testing and reproducibility. The results will be visualized to provide insights into the strengths and weaknesses of each algorithm.


## Step 1: Define the Maze Environment
### Create the Maze Structure:
Define a grid-based maze environment (4x4, 6x6, 8x8).
Add walls and dead ends to increase complexity.
Ensure each maze has a valid path from the start to the goal (bottom-right corner).

### Random Maze Generator:
Implement a random maze generator using an algorithm like Recursive Backtracking or Prim’s Algorithm.
Ensure the generator can create mazes of different sizes and complexities.

## Step 2: Implement the Algorithms
### Q-learning:
Define the state space (agent’s position), action space (up, down, left, right), and reward function.
Implement the Q-learning algorithm with an epsilon-greedy exploration strategy.
Track metrics like success rate, steps to solve, and reward convergence.
Save the Q-table for reuse and visualization.

### Dijkstra’s Algorithm:
Implement Dijkstra’s Algorithm to find the shortest path in the maze.
Track performance metrics (time to solve, number of steps).

### A*:
Implement A* with a heuristic (e.g., Manhattan distance to the goal).
Track performance metrics (time to solve, number of steps).

### Greedy Best-First Search (Optional):
Implement Greedy Best-First Search as a non-optimal comparison algorithm.
Track performance metrics (time to solve, number of steps).

## Step 3: Develop the Code Structure
### File Structure:
  #### maze_generator.py: Random maze generator.
  #### game_environment.py: Maze environment and agent movement logic.
  #### q_learning.py: Q-learning implementation.
  #### traditional_algorithms.py: Dijkstra, A*, and Greedy Best-First Search.
  #### visualization.py: Visualization of results (e.g., success rate, steps to solve).
  #### main.py: Main script to run experiments and generate results.

### Documentation:
Add comments to your code for clarity.
Write a README.md file explaining how to set up and run the project.
Include instructions for reproducing results.

## Step 4: Run Experiments
### Train Q-learning:
Train Q-learning on each maze stage (4x4, 6x6, 8x8).
Save the results (success rate, steps to solve, reward convergence).
Run Traditional Algorithms:
Run Dijkstra, A*, and Greedy Best-First Search on the same mazes.
Record performance metrics (time to solve, number of steps).

## Step 5: Visualize Results
### Q-learning Metrics:
Plot success rate over episodes.
Visualize reward convergence.
Analyze exploration vs. exploitation.
Algorithm Comparison:
Compare steps to solve and time to solve across all algorithms.
Use bar charts or line plots for clear visualization.

## Step 6: Write the Thesis
### Organize Results:
Summarize the results of your experiments.
Include visualizations and analysis.
Discuss Findings:
Compare the performance of Q-learning with traditional algorithms.
Discuss the strengths and weaknesses of each approach.
Prepare Presentation:
Create slides summarizing your thesis for the final presentation.
Highlight key findings and visualizations.
