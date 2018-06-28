import random
from collections import namedtuple, deque

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.total_priority = 0.0

    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""
        # Subtract off 1st priority (if memory is full)
        if len(self.memory) == self.buffer_size:
            self.total_priority -= (self.memory[0]).priority
        # Add to total priority
        self.total_priority += priority
        # Remember this experience
        e = self.experience(state, action, reward, next_state, done, priority)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        sample = []
        for e in self.memory:
            # Pick a random number from 0 to 1
            rand_num = random.random()
            # if number is less than Prob, use experience
            prob =  e.priority / self.total_priority
            if rand_num < prob:
                sample.append(e)
        # Always use previous experience if it is empty        
        return sample if len(sample) > 0 else [self.memory[-1]]

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)