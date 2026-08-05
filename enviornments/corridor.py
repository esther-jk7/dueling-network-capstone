"""
    Drew Hill & Esther Suravarapu

    corridor.py
    
    This class acts as the enviornment for the corridor experiment.
"""
import numpy as np

class CorridorEnv:
    def __init__(self, num_actions, num_states, gamma):
        self.num_actions = num_actions
        self.num_states = num_states
        self.gamma = gamma

        self.actions = {"forward": 0, "back": 1, "left": 2, "right": 3, "no_op": 4}

        self.start_state = 0
        self.end_state = self.num_states - 1

        self.horz_start = self.start_state + 10
        self.horz_end = self.end_state - 10

        self.current_state = self.start_state
        self.terminal = False

    def reset(self):
        """
            Resets the enviornment to the original setup.
            
            returns the current state after reset.
        """
        self.current_state = self.start_state
        self.terminal = False

        return self.current_state

    def transition(self, state, action):
        """
            Determines the next state based on a state and action.

            state: int - The state before transition.
            action: str - The action to take.
            raises ValueError if  invalid action is passed.
            returns the next state from the transition.
        """
        # Returns state if it is the end state.
        if state ==  self.end_state:
            return state

        # Checks if the current state is one the 50 horzontal corridor states.
        if self.horz_start <= state < self.horz_end:
            # Determines transition based on action in horizontal corridor.
            if action == self.actions["right"]:
                return state + 1
            
            elif action == self.actions["left"]:
                return state - 1

            else:
                return state

        # Checks if the current state is one of the 10 non horizontal states.
        else:
            # Determines transition based on action in non horizontal corridor.
            if action == self.actions["forward"]:
                return min(self.end_state, state + 1)

            elif action == self.actions["back"]:
                return max(self.start_state, state - 1)
            
            else:
                return state

    def step(self, action):
        """
            Applies an action and returns the next state, reward and if state 
            is terminal.

            action: str - The action to take.
            returns next state, reward and if state is terminal.
        """
        next_state = self.transition(self.current_state, action)

        if next_state == self.end_state:
            reward = 1
            self.terminal = True

        else:
            reward = 0

        # Updates state to the next state
        self.current_state = next_state

        return next_state, reward, self.terminal

    def to_vector(self, state):
        """
            Converts a vector needed for the network input.

            state: int - The state before transition.
            returns a vector with the state location in enviornment noted.
        """
        vec = np.zeros(self.num_states, dtype = np.float32)
        vec[state] = 1.0

        return vec