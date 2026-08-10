"""
    Esther Suravarapu & Drew Hill

    run_td0.py

    Runs the corridor experiment as described in Wang et al. 2016.
    TD(0) policy evaluation — agent follows a fixed epsilon-greedy
    policy and learns Q-values for that policy.
"""
import sys, os
sys.path.append(os.path.abspath('..'))

import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt

from enviornments.corridor import CorridorEnv
from networks.single_stream import SingleStreamNetwork
from networks.dueling import DuelingNetwork
import experiments.evaluation as e


def run_td0(network_type, num_actions, seed, num_episodes=2000, record_every=10):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    gamma = 0.99
    epsilon = 0.001
    env = CorridorEnv(num_actions, 70, gamma)

    # build ground truth
    q = e.value_iteration(env, 1e-10, 100000)
    pi = e.eps_greedy_policy(env, q, epsilon)
    q_pi = e.policy_evaluation(env, pi, 1e-12, 200000)

    all_states = torch.eye(env.num_states)

    if network_type == 'single':
        network = SingleStreamNetwork(env.num_states, 50, env.num_actions)
    else:
        network = DuelingNetwork(env.num_states, env.num_actions, 50, 25)

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    curve = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            state_vec = torch.tensor(env.to_vector(state), dtype=torch.float32).unsqueeze(0)

            # follow epsilon-greedy policy
            if random.random() < epsilon:
                action = random.randint(0, env.num_actions - 1)
            else:
                action = np.argmax(pi[state])

            next_state, reward, done = env.step(action)
            next_state_vec = torch.tensor(env.to_vector(next_state), dtype=torch.float32).unsqueeze(0)

            # TD(0) update
            q_current = network(state_vec)
            q_sa = q_current[0, action]

            with torch.no_grad():
                if done:
                    target = torch.tensor(reward, dtype=torch.float32)
                else:
                    q_next = network(next_state_vec)
                    # expected value under policy for next state
                    pi_next = torch.tensor(pi[next_state], dtype=torch.float32)
                    expected_next = (pi_next * q_next[0]).sum()
                    target = reward + gamma * expected_next

            loss = nn.functional.mse_loss(q_sa, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()

            state = next_state

        # record SE
        if ep % record_every == 0:
            q_network = network(all_states).detach().numpy()
            se = e.compute_se(q_pi, q_network)
            curve.append((ep, se))

    return curve


def smooth(y, window=10):
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window)/window, mode='valid')


if __name__ == '__main__':
    seeds = [0, 1, 2, 3, 4]
    num_episodes = 5000
    record_every = 10

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, num_actions in enumerate([5, 10, 20]):
        print(f'Running TD(0) — {num_actions} actions...')

        single_curves = [run_td0('single', num_actions, s, num_episodes, record_every) for s in seeds]
        duel_curves = [run_td0('duel', num_actions, s, num_episodes, record_every) for s in seeds]

        iters = np.array([x[0] for x in single_curves[0]])
        single_mean = np.mean([[x[1] for x in c] for c in single_curves], axis=0)
        duel_mean = np.mean([[x[1] for x in c] for c in duel_curves], axis=0)

        print(f'  Final SE -> single: {single_mean[-1]:.2f}, dueling: {duel_mean[-1]:.2f}')

        w = 10
        s_iters = iters[w-1:]
        ax = axes[i]
        ax.plot(s_iters, smooth(single_mean, w), '--', color='crimson', label='Single Stream', linewidth=2)
        ax.plot(s_iters, smooth(duel_mean, w), '-', color='seagreen', label='Dueling', linewidth=2)
        ax.set_title(f'{num_actions} Actions', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episodes', fontsize=12)
        ax.set_ylabel('Squared Error', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.suptitle('Corridor Experiment: TD(0) Policy Evaluation\nSingle Stream vs Dueling (Wang et al. 2016 reproduction)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('td0_results.png', dpi=150, bbox_inches='tight')
    print('Saved td0_results.png')

    plt.show()
