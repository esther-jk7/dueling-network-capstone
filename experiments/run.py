"""
    Drew Hill & Esther Suravarapu

    run.py

    Runs the full corridor experiment comparing single-stream and dueling
    networks across 5, 10, and 20 actions using Expected SARSA and DDQN.
    Generates learning curve plots averaged over multiple seeds.
"""
import sys
import os
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
from training.ddqn import ddqn_update, sync_target


# ── Training functions ──────────────────────────────────────

def train_expected_sarsa(network, env, pi, q_pi, num_iters, gamma, record_every):
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    pi_tensor = torch.tensor(pi, dtype=torch.float32)
    all_states = torch.eye(env.num_states)
    curve = []

    for i in range(num_iters):
        next_state_indices = []
        actions = []
        rewards = []
        terminal = []
        state_indices = []

        for state in range(env.num_states):
            if state == env.end_state:
                continue
            action = random.randint(0, env.num_actions - 1)
            next_state = env.transition(state, action)
            actions.append(action)
            next_state_indices.append(next_state)
            state_indices.append(state)
            if next_state == env.end_state:
                rewards.append(1.0)
                terminal.append(1.0)
            else:
                rewards.append(0.0)
                terminal.append(0.0)

        states = all_states[state_indices]
        actions_t = torch.tensor(actions)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states = all_states[next_state_indices]
        terminal_t = torch.tensor(terminal, dtype=torch.float32)

        q_values = network(states)
        q_selected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = network(next_states)
            next_pi = pi_tensor[next_states.argmax(dim=1)]
            expected_next = (next_pi * next_q_values).sum(dim=1)
            target = rewards_t + gamma * expected_next * (1 - terminal_t)
            target = target.clamp(0, 1)

        loss = nn.functional.mse_loss(q_selected, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()

        if i % record_every == 0:
            q_network = network(all_states).detach().numpy()
            se = e.compute_se(q_pi, q_network)
            curve.append((i, se))

    return curve


def train_ddqn(online, target_net, env, q_pi, num_iters, gamma, record_every):
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
    all_states = torch.eye(env.num_states)
    curve = []

    for i in range(num_iters):
        next_state_indices = []
        actions = []
        rewards = []
        terminal = []
        state_indices = []

        for state in range(env.num_states):
            if state == env.end_state:
                continue
            action = random.randint(0, env.num_actions - 1)
            next_state = env.transition(state, action)
            actions.append(action)
            next_state_indices.append(next_state)
            state_indices.append(state)
            if next_state == env.end_state:
                rewards.append(1.0)
                terminal.append(1.0)
            else:
                rewards.append(0.0)
                terminal.append(0.0)

        states = all_states[state_indices]
        actions_t = torch.tensor(actions)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states = all_states[next_state_indices]
        terminal_t = torch.tensor(terminal, dtype=torch.float32)

        batch = (states, actions_t, rewards_t, next_states, terminal_t)
        ddqn_update(online, target_net, optimizer, batch, gamma=gamma)

        if i % 100 == 0:
            sync_target(online, target_net)

        if i % record_every == 0:
            q_network = online(all_states).detach().numpy()
            se = e.compute_se(q_pi, q_network)
            curve.append((i, se))

    return curve


# ── Experiment runners ──────────────────────────────────────

def run_single_experiment(algorithm, network_type, num_actions, seed,
                          num_states=70, gamma=0.99, epsilon=0.001,
                          num_iters=10000, record_every=50):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = CorridorEnv(num_actions, num_states, gamma)
    q = e.value_iteration(env, 1e-10, 100000)
    pi = e.eps_greedy_policy(env, q, epsilon)
    q_pi = e.policy_evaluation(env, pi, 1e-12, 200000)

    if network_type == 'single':
        network = SingleStreamNetwork(env.num_states, 50, env.num_actions)
    else:
        network = DuelingNetwork(env.num_states, env.num_actions, 50, 25)

    if algorithm == 'expected_sarsa':
        return train_expected_sarsa(network, env, pi, q_pi, num_iters, gamma, record_every)
    else:
        if network_type == 'single':
            target_net = SingleStreamNetwork(env.num_states, 50, env.num_actions)
        else:
            target_net = DuelingNetwork(env.num_states, env.num_actions, 50, 25)
        sync_target(network, target_net)
        return train_ddqn(network, target_net, env, q_pi, num_iters, gamma, record_every)


def run_averaged(algorithm, network_type, num_actions, seeds, **kwargs):
    all_curves = [run_single_experiment(algorithm, network_type, num_actions, s, **kwargs) for s in seeds]
    iters = [x[0] for x in all_curves[0]]
    mean_se = np.mean([[x[1] for x in c] for c in all_curves], axis=0)
    return np.array(iters), mean_se


# ── Plotting ──────────────────────────────────────────────

def smooth(y, window=20):
    return np.convolve(y, np.ones(window)/window, mode='valid')


def plot_algorithm(algorithm, results, filename):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, num_actions in enumerate([5, 10, 20]):
        iters, single, duel = results[num_actions]
        w = 20
        s_iters = iters[w-1:]

        ax = axes[i]
        ax.plot(s_iters, smooth(single, w), '--', color='crimson', label='Single Stream', linewidth=2)
        ax.plot(s_iters, smooth(duel, w), '-', color='seagreen', label='Dueling', linewidth=2)
        ax.set_title(f'{num_actions} Actions', fontsize=14, fontweight='bold')
        ax.set_xlabel('Iterations', fontsize=12)
        ax.set_ylabel('Squared Error', fontsize=12)
        ax.set_xscale('log')
        ax.legend(fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.3)

    name = 'Expected SARSA' if algorithm == 'expected_sarsa' else 'DDQN'
    plt.suptitle(f'Corridor Experiment: {name}\nSingle Stream vs Dueling (Averaged over 5 seeds, smoothed)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f'Saved {filename}')


def plot_combined(sarsa_results, ddqn_results, filename):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for i, num_actions in enumerate([5, 10, 20]):
        w = 20

        # Expected SARSA row
        iters, single, duel = sarsa_results[num_actions]
        s_iters = iters[w-1:]
        ax = axes[0][i]
        ax.plot(s_iters, smooth(single, w), '--', color='crimson', label='Single Stream', linewidth=2)
        ax.plot(s_iters, smooth(duel, w), '-', color='seagreen', label='Dueling', linewidth=2)
        ax.set_title(f'{num_actions} Actions', fontsize=14, fontweight='bold')
        ax.set_xlabel('Iterations', fontsize=11)
        ax.set_ylabel('Squared Error', fontsize=11)
        ax.set_xscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        if i == 0:
            ax.annotate('Expected SARSA', xy=(0, 0.5), xytext=(-50, 0),
                        xycoords='axes fraction', textcoords='offset points',
                        fontsize=13, fontweight='bold', rotation=90, va='center')

        # DDQN row
        iters, single, duel = ddqn_results[num_actions]
        s_iters = iters[w-1:]
        ax = axes[1][i]
        ax.plot(s_iters, smooth(single, w), '--', color='crimson', label='Single Stream', linewidth=2)
        ax.plot(s_iters, smooth(duel, w), '-', color='seagreen', label='Dueling', linewidth=2)
        ax.set_title(f'{num_actions} Actions', fontsize=14, fontweight='bold')
        ax.set_xlabel('Iterations', fontsize=11)
        ax.set_ylabel('Squared Error', fontsize=11)
        ax.set_xscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        if i == 0:
            ax.annotate('DDQN', xy=(0, 0.5), xytext=(-50, 0),
                        xycoords='axes fraction', textcoords='offset points',
                        fontsize=13, fontweight='bold', rotation=90, va='center')

    plt.suptitle('Corridor Experiment: Expected SARSA vs DDQN\nSingle Stream vs Dueling Network (Averaged over 5 seeds, smoothed)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f'Saved {filename}')


# ── Main ──────────────────────────────────────────────────

if __name__ == '__main__':
    seeds = [0, 1, 2, 3, 4]
    action_counts = [5, 10, 20]

    # Run Expected SARSA experiments
    sarsa_results = {}
    for num_actions in action_counts:
        print(f'Expected SARSA — {num_actions} actions...')
        iters, single = run_averaged('expected_sarsa', 'single', num_actions, seeds)
        _, duel = run_averaged('expected_sarsa', 'duel', num_actions, seeds)
        sarsa_results[num_actions] = (iters, single, duel)
        print(f'  Final SE -> single: {single[-1]:.2f}, dueling: {duel[-1]:.2f}')

    # Run DDQN experiments
    ddqn_results = {}
    for num_actions in action_counts:
        print(f'DDQN — {num_actions} actions...')
        iters, single = run_averaged('ddqn', 'single', num_actions, seeds)
        _, duel = run_averaged('ddqn', 'duel', num_actions, seeds)
        ddqn_results[num_actions] = (iters, single, duel)
        print(f'  Final SE -> single: {single[-1]:.2f}, dueling: {duel[-1]:.2f}')

    # Generate plots
    plot_algorithm('expected_sarsa', sarsa_results, 'expected_sarsa_results.png')
    plot_algorithm('ddqn', ddqn_results, 'ddqn_results.png')
    plot_combined(sarsa_results, ddqn_results, 'full_comparison.png')

    # Print summary table
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    print(f'{"":>10} {"Expected SARSA":>25} {"DDQN":>25}')
    print(f'{"Actions":>10} {"Single":>12} {"Dueling":>12} {"Single":>12} {"Dueling":>12}')
    print('-'*60)
    for na in action_counts:
        ss = sarsa_results[na][1][-1]
        sd = sarsa_results[na][2][-1]
        ds = ddqn_results[na][1][-1]
        dd = ddqn_results[na][2][-1]
        print(f'{na:>10} {ss:>12.2f} {sd:>12.2f} {ds:>12.2f} {dd:>12.2f}')
    print('='*60)
