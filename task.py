import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3
        
        # State should be all positions & velocities for 3 dimensions & 3 angles
        self.state_size = self.action_repeat * 12 
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4 # one for each rotor

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_vel_all = []
        vel_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            #  Get all positions (direstions, angles) for current instant
            instant_pose = self.sim.pose
            # Get all velocities (positions, angular) for current instant
            instant_vel = np.concatenate((self.sim.v, self.sim.angular_v))
            # For action repeat, get all 12pts for state
            pose_vel_all.append(np.concatenate((instant_pose,instant_vel)))
        # Get full state
        next_state = np.concatenate(pose_vel_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        all_pose_vel = np.concatenate((self.sim.pose, self.sim.v, self.sim.angular_v)) 
        state = np.concatenate([all_pose_vel] * self.action_repeat) 
        return state