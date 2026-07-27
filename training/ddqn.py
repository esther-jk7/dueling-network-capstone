import torch
import torch.nn as nn

def select_action(net, state, num_actions, epsilon=0.001):
    """Epsilon-greedy action selection. Wang et al. (2016) uses epsilon=0.001"""
    if torch.rand(1).item() < epsilon:
        return torch.randint(num_actions, (1,)).item()  # random action
    with torch.no_grad():
        q_values = net(state)
        return q_values.argmax(dim=1).item()  # greedy action


def ddqn_update(online_net, target_net, optimizer, batch, gamma=0.99):
    """
    DDQN update step - van Hasselt et al. (2016).
    online_net selects action, target_net evaluates it.
    """
    states, actions, rewards, next_states, dones = batch

    # current Q-values for actions taken
    q_values = online_net(states)
    q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # online net picks best next action
        next_actions = online_net(next_states).argmax(dim=1)
        # target net evaluates that action
        next_q = target_net(next_states)
        next_q_selected = next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = rewards + gamma * next_q_selected * (1 - dones)

    loss = nn.functional.mse_loss(q_selected, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def sync_target(online_net, target_net):
    """Copy online network weights into target network."""
    target_net.load_state_dict(online_net.state_dict())