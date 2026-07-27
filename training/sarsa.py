import torch
import torch.nn as nn


def sarsa_select_action(net, state, num_actions, epsilon=0.001):
    """Epsilon-greedy action selection for SARSA."""
    if torch.rand(1).item() < epsilon:
        return torch.randint(num_actions, (1,)).item()
    with torch.no_grad():
        q_values = net(state)
        return q_values.argmax(dim=1).item()


def sarsa_update(net, optimizer, batch, gamma=0.99):
    """
    SARSA update step - on-policy.
    Uses next_action actually taken, not the greedy best.
    Batch needs (states, actions, rewards, next_states, next_actions, dones).
    """
    states, actions, rewards, next_states, next_actions, dones = batch

    # Q-value for action taken
    q_values = net(states)
    q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # evaluate the action actually taken next (not the best one)
        next_q_values = net(next_states)
        next_q_selected = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = rewards + gamma * next_q_selected * (1 - dones)

    loss = nn.functional.mse_loss(q_selected, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()