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
        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4 # one for each rotor

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # Ignore X & Y velocities for now
        reward_xy_vel = 0
        
        # Two states that matter: below target or above target
        # Note being at target exactly is defined as being "below" target
        isBelowTarget = self.sim.pose[2] <= self.target_pos[2]
        if isBelowTarget:
            # positively reward +Z velocity but strongly discourage any -Z velocity
            if self.sim.v[2] <= 0: # v==0 gives zero score
                # Discourage high negative velocities
                reward_z_vel = 1*np.tanh(-(self.sim.v[2])**2)
            else:
                # Strong reward for small velocities, but diminishing returns on higher velocity
                reward_z_vel = 5*np.tanh(0.5 + 10*self.sim.v[2] ) #encourage high velocities
        # Above target (take it down)
        else:
            # positively reward -Z velocity but strongly discourage any +Z velocity
            if self.sim.v[2] > 0:
                # Somewhat discourage positive velocities 
                reward_z_vel = -2*np.tanh((self.sim.v[2])**2)
            else:
                # Encourage negative velocities (reward small quickly; diminishing returns)
                reward_z_vel = 1*np.tanh(0.5 + -self.sim.v[2])
                
        # One velocity reward
        reward_xyz_vel = reward_xy_vel + reward_z_vel
        
        # Reward correct position (z)
        reward_z_pos = 5*np.tanh(1 - 0.003*(abs(self.sim.pose[2] - self.target_pos[2])))
        # Don't stress about position for x & y for now
        reward_xy_pos = 0 
        reward_xyz_pos = reward_xy_pos + reward_z_pos

        # Scale final reward so total is usually less than 10 for each episode
        reward = (reward_xyz_vel + reward_xyz_pos)/100.

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        # Get full state
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state