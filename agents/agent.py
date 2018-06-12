import numpy as np
from task import Task

class PolicySearch_Agent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        # Episode variables
        self.reset_episode()
        
        # State space definitions
        self.low_pos = 3*[0]      # assume there is nothing in negative directions
        self.high_pos = 3*[100]
        self.low_ang = 3*[0]      # assume angle is only positive
        self.high_ang = 3*[10]
        self.low_v = 3*[-10]
        self.high_v = 3*[20]
        self.low_ang_v = 3*[-1]
        self.high_ang_v = 3*[1]
        
        # State combined (xyz, angles, v_xyz, v_angles)
        self.state_low = np.concatenate((self.low_pos,self.low_ang,self.low_v,self.low_ang_v))
        self.state_high = np.concatenate((self.high_pos,self.high_ang,self.high_v,self.high_ang_v))
        self.bins = [25]*12 # Same # of bins for all points (likely will have to change
        
        # Defining the discrete grid
        self.grid = self.create_grid(self.state_low, self.state_high, self.bins)
        
        # Learning parameters
        self.alpha = 0.02  # learning rate
        self.gamma = 0.99 # discount factor
        self.epsilon = self.initial_epsilon = 1.  # initial exploration rate
        self.epsilon_decay_rate = 0.9995 # how quickly should we decrease epsilon
        self.min_epsilon = 0.01
        
        # Create Q-table
        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
       
 

    ##### Create grid & point from the continuous space
    
    def create_grid(self, low, high, bins):
        ''' Define a grid with uniform spaces. Inspired by 
            github.com/udacity/reinforcement-learning
            
        
        Parameters
        ----------
        low : array_like
            Lower bounds for each dimension of the continuous space.
        high : array_like
            Upper bounds for each dimension of the continuous space.
        bins : tuple
            Number of bins along each corresponding dimension.

        Returns
        -------
        grid : list of array_like
            A list of arrays containing split points for each dimension.
        '''
        # Return a discrete grid that is spaced into N bins
        grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
       
        return grid
        
    
    def discretize(self, point, grid):
        '''Discretize a state for the given grid. Inspired by 
           github.com/udacity/reinforcement-learning

        Parameters
        ----------
        sample : array_like
            A single point from a continuous space.
        grid : list of array_like
            A list of arrays containing split points for each dimension.

        Returns
        -------
        discretized_point : array_like
            A sequence of integers with the same number of dimensions as point.
        '''
        # Digitize for each dimension of original continuous space
        discretized_point = [int(np.digitize(p, g)) for p, g in zip(point, grid)]
        return discretized_point
      
    def preprocess_state(self, state):
        '''Map continuous state to discretization.'''
        return tuple(discretize(state, self.state_grid))
        
    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state

    def step(self, reward, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

        # Learn, if at end of episode
        if done:
            self.learn()

    
    def act(self, state):
        # Choose action based on given state and policy
        action = np.dot(state, self.w)  # simple linear policy
        return action

    def learn(self):
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions
        