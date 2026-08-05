"""
    Drew Hill & Esther Suravarapu

    expected_sarsa.py

    This file is used to train the networks for the corridor experiment using
    expected SARSA. It selects an action using epsilon greedy, then updates 
    the networks predicted q values, pushing it towars the target.
"""
import torch
import torch.nn as nn
import random

def select_action(network, state, num_actions, epsilon):
    """
        Selects an action using epsilon greedy.

        network: torch.nn - The network.
        state: tensor - The current states encoded.
        num_actions: int - The number of actions.
        epsilon: float - Probability of selecting random action.
        returns an action.
    """
    # Returns a random action with a probability of epsilon.
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)

    # Selects the greedy action from the network.
    with torch.no_grad():
        q_values = network(state)
        return q_values.argmax(dim=1).item()

def expected_sarsa_update(network, optimizer, batch, pi, gamma):
    """
        Completes an update for expected Sarsa. Pushes the networks predicted
        q value fir an action towards target.

        network: torch.nn - The network.
        optimizer: torch - The torch optimizer used to update weights.
        batch: tuple - Holds states, actions, rewards, next_states, terminals.
        pi: tensor - The epsilon greedy policy proabilities.
        gamma: float - Discount factor.
        Returns the loss value.
    """
    states, actions, rewards, next_states, terminals = batch

    # Gets q values from the network.
    q_values = network(states)

    # Gets the q values of the actions taken.
    q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Gets all of the next q values from the network.
        next_values = network(next_states)

        # Gets the indecies and probabilities table for next states.
        next_states_index = next_states.argmax(dim = 1)
        next_pi = pi[next_states_index]

        # Computes weighed averages for the next q values and the target.
        expected_next = (next_pi * next_values).sum(dim = 1)
        target = rewards + gamma* expected_next * (1 - terminals)

    # Computes distatnce from target.
    loss = nn.functional.mse_loss(q_selected, target)

    # Computes new gradient and updates the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()