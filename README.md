# Reproduction of Dueling Network Architecture for Deep RL

**Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016)**  
*Proceedings of the 33rd International Conference on Machine Learning (ICML), Vol. 48, pp. 1995–2003*

Drew Hill & Esther Suravarapu  
CS5180 — Reinforcement Learning & Sequential Decision Making  
Northeastern University | Summer 2026

---

## Overview

This repository contains a reproduction of the corridor experiment from the Dueling Network Architecture paper. The dueling architecture splits a Q-network into two streams — a value stream V(s) and an advantage stream A(s,a) — and recombines them using mean subtraction (Equation 9). We compare this against a standard single-stream baseline across 5, 10, and 20 action variants of a corridor environment.

**Key result:** Dueling outperforms single-stream at all three action counts, with the advantage growing as actions increase (23% → 27% → 51% lower squared error). This confirms the paper's core claim.

## Repository Structure

```
dueling-network-capstone/
├── networks/
│   ├── single_stream.py          # Baseline single-stream Q-network
│   └── dueling.py                # Dueling network (V + A streams, Equation 9)
├── training/
│   ├── ddqn.py                   # DDQN training with target clamping
│   └── expected_sarsa.py         # Expected SARSA training (Drew)
├── enviornments/
│   └── corridor.py               # 70-state corridor environment (Drew)
├── experiments/
│   ├── evaluation.py             # Value iteration, policy evaluation, SE metric (Drew)
│   ├── run_td0.ipynb             # Primary experiment: TD(0) policy evaluation
│   ├── run.ipynb                 # Additional experiments: Expected SARSA + DDQN
│   └── corridor_visual.py        # Corridor visualization
├── td0_results.png               # Primary result learning curves
├── corridor_full_results.png     # SARSA vs DDQN comparison curves
└── README.md
```

## Setup

### Requirements
- Python 3.10+
- PyTorch
- NumPy
- Matplotlib

### Installation
```bash
git clone https://github.com/esther-jk7/dueling-network-capstone.git
cd dueling-network-capstone
python3 -m venv venv
source venv/bin/activate
pip install torch numpy matplotlib
```

## Running Experiments

### Primary Experiment — TD(0) Policy Evaluation
This reproduces the corridor experiment from the paper using episode-based TD(0).

Open `experiments/run_td0.ipynb` in Jupyter Notebook, Google Colab, or VS Code and run all cells. This trains both single-stream and dueling networks across 5, 10, and 20 actions, averaged over 5 random seeds. Takes approximately 30–40 minutes. Outputs learning curves and final squared error values.

### Additional Experiments — Expected SARSA and DDQN
Open `experiments/run.ipynb` and run all cells. This compares Expected SARSA and DDQN as alternative training algorithms and generates comparison plots.

## Results

### TD(0) Policy Evaluation (Primary)

| Actions | Single Stream SE | Dueling SE | Improvement |
|---------|-----------------|------------|-------------|
| 5       | 4.51            | 3.48       | 23%         |
| 10      | 11.99           | 8.77       | 27%         |
| 20      | 33.17           | 16.25      | 51%         |

Dueling outperforms single-stream at all action counts. The advantage grows with the number of actions, confirming the paper's central claim.

### Training Algorithm Comparison

| Algorithm       | Type       | Stability | Notes |
|----------------|------------|-----------|-------|
| TD(0)          | Prediction | Stable    | Paper's method. Best results. |
| Expected SARSA | Prediction | Stable    | Consistent with TD(0) findings. |
| DDQN           | Control    | Unstable  | Requires target clamping. Max operator causes overestimation. |

## Key Implementation Details

- **State encoding:** 70-dimensional one-hot vectors
- **Corridor:** 70 states (horizontal + vertical sections), reward +1 at goal
- **Epsilon-greedy:** ε = 0.001 (fixed policy, not updated during training)
- **Optimizer:** Adam, lr = 0.001
- **Gradient clipping:** 1.0
- **Seeds:** 5, averaged

### Critical Reproduction Insight
The paper uses **episode-based TD(0) policy evaluation** — the agent walks through the corridor step by step following a fixed policy, with one TD(0) update per transition. Batch sampling across all states simultaneously does not reproduce the paper's results. This was the most important implementation detail for successful reproduction.

## Reproducibility Verdict

**Partially Reproducible.** The core claim (dueling advantage grows with action count) is fully confirmed. However, the paper lacks critical implementation details for the corridor experiment — data collection method, action semantics, and state encoding are not fully specified. Other researchers have reported similar replication difficulties.

## Contributions

| Member | Responsibilities |
|--------|-----------------|
| Esther Suravarapu | Network architectures (single-stream, dueling), training algorithms (DDQN, TD(0)), bug identification and fixes, experiment runner (run_td0) |
| Drew Hill | Corridor environment, evaluation pipeline (value iteration, policy evaluation), Expected SARSA training, notebook conversion |

## References

- Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. *ICML*, Vol. 48, pp. 1995–2003.
- Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529–533.
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. *AAAI*, pp. 2094–2100.