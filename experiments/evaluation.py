"""
    Drew Hill & Esther Suravarapu
    
    evaluation.py

    This file us used for policy evaluation for the corridor experiment. The 
    evaluation is done by building an answer key that is used as a comparison
    against the trained networks. This is used to compare the single steam to 
    the duel stream by finding the squared error between the single stream and
    the actual and the duel stream and the actual and comparing those results.

    To complete the evaaluation you must run the methods in the following order:
    1. value_iteration - This finds the best value at each state.
    2. eps_greedy_policy - This finds the policy with random action .1% of the time.
    3. policy_evaluation - Finds the actual q values for the policy to be used as an 
       answer sheet.
    4. compute_se - Scores how good the network performed compared to actual.
"""
import numpy as np
from enviornments.corridor import CorridorEnv

def value_iteration(env, threshold, max_iters):
    """
        Computes the q values for each action at each state through value iteration.

        env: CorridorEnv - The envirment.
        threshold: float - The convergence threshold.
        max_iters: int - Safety to prevent infinate loop.
        returns a q table with q values for each action at each state.
    """
    # Initializes the q table with 0 for each state action pair.
    q = np.zeros((env.num_states, env.num_actions))

    for iter in range(max_iters):
        # Initializes convergence check.
        delta = 0.0

        # Initializes new_q q table.
        new_q = np.zeros((env.num_states, env.num_actions))

        for state in range(env.num_states):

            # Terminal state keeps zero value.
            if state == env.end_state:
                continue

            for action in range(env.num_actions):
                # Gets the next_state from the enviornment.
                next_state = env.transition(state, action)

                # Determines the reward.
                if next_state == env.end_state:
                    reward = 1

                else:
                    reward = 0

                # Computes the Bellman update.
                new_q[state][action] = reward + env.gamma * np.max(q[next_state])

            # Computes change in states value.
            error = np.abs(new_q[state] - q[state])
            delta = max(delta, np.max(error))

        q = new_q

        # Checks for convergence.
        if delta < threshold:
            break

    return q

def eps_greedy_policy(env, q, epsilon):
    """
        Builds epsilon greedy policy from the q table. Q table provides pure 
        greedy policy, this adds a random action for .1% of actions taken.

        env: CorridorEnv - The envirment.
        q: array - Q table using value iteration.
        epsilon - Probabiluty of random action taken.
        returns a probability table adjusted for random actions taken.
    """
    # Initializes table wit space for each state and action.
    pi = np.zeros((env.num_states, env.num_actions))

    for state in range(env.num_states):
        # Gets the greedy action from the q table.
        greedy_action = np.argmax(q[state])

        for action in range(env.num_actions):
            # Gives each action equal random probaility.
            pi[state][action] = epsilon / env.num_actions

            if action == greedy_action:
                pi[state][action] += 1.0 - epsilon

    return pi

def policy_evaluation(env, pi, threshold, max_iters):
    """
        Computes the actuial action values q_pi. This is used as the actual 
        q values based to pure greedy action and probablities if action.

        env: CorridorEnv - The envirment.
        pi: array - The epsilon greedy policy probablility table.
        threshold: float - The convergence threshold.
        max_iters: int - Safety to prevent infinate loop.
        returns a q table with the actual q values.
    """
    # Initializes the q table with 0 for each state action pair.
    q = np.zeros((env.num_states, env.num_actions))

    for iter in range(max_iters):
        # Initializes convergence check.
        delta = 0.0

        # Initializes new_q q table.
        new_q = np.zeros((env.num_states, env.num_actions))

        for state in range(env.num_states):
            # Terminal state keeps zero value
            if state == env.end_state:
                continue

            for action in range(env.num_actions):
                # Gets the next_state from the enviornment.
                next_state = env.transition(state, action)

                # Determines the reward.
                if next_state == env.end_state:
                    reward = 1

                else:
                    reward = 0

                # Computes update.
                new_q[state][action] = reward + env.gamma * np.dot(pi[next_state], q[next_state])

            # Computes change in states value.
            error = np.abs(new_q[state] - q[state])
            delta = max(delta, np.max(error))

        q = new_q

        # Checks for convergence.
        if delta < threshold:
            break

    return q

def compute_se(q_actual, q_network):
    """
        Computes the squared error between the actual values and the networks
        predicted values.

        q_actual: array - The actual values of q_pi.
        q_network: array - The networks predicted q values.
        returns a float for the sum of squared errors.
    """
    return np.sum((q_actual - q_network) ** 2)       