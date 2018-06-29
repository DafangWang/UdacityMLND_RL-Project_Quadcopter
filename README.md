# UdacityMLND_RL-Project_Quadcopter
Reinforcement learning project for Udacity's Machine Learning Nanodegree.

In this project I run a simulation of a quadcopter with 4 independent rotors to move around a room of 300m x 300m x 300m. The corrdinates define x:(-150,150) y:(-150,150) z:(0,300). Each simulation runs at a maximum of 5s unless a boundary is hit by the quadcopter in which case the simulation terminates.

My task `task_hover10m.py` defines the quadcopter to start at (0,0,10) and to hover at the target (0,0,10). I implemented the DDPG algorithm (that uses an actor and critic method) with prioritized experience replay.
