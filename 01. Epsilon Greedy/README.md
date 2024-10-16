# Epsilon Greedy

**Overview:**

The Epsilon-Greedy algorithm is one of the simplest and most commonly used strategies for solving the multi-armed bandit problem in Reinforcement Learning (RL). The problem involves choosing between multiple options (bandits) with unknown probabilities of reward to maximize the total reward over a series of trials. Epsilon-Greedy balances the trade-off between exploration (trying new options to discover potentially better rewards) and exploitation (choosing the best-known option).

**Multi-Armed Bandit Problem:**

- Imagine a slot machine (bandit) with multiple arms, each arm providing a different reward probability. The goal is to find which arm yields the highest reward over time.
- The challenge is that the probabilities of winning are unknown at the start. You need to balance exploring different arms (to find the best one) and exploiting the current best option based on your observations.

**Exploration vs. Exploitation:**

Exploration refers to trying out new bandits to discover whether they offer better rewards than the current best option.
Exploitation involves selecting the bandit with the highest estimated reward based on previous trials.
The Epsilon-Greedy strategy introduces a parameter called epsilon (ϵ), which controls how often you explore versus exploit.

**Epsilon-Greedy Algorithm:**

Epsilon Parameter (ϵ): A small probability (ϵ) is defined (e.g., 0.1). This is the exploration rate, determining the likelihood of trying a random action instead of choosing the current best.

Bandit Selection:

With probability (1 - ϵ), the algorithm exploits by choosing the bandit with the highest estimated probability of reward.
With probability ϵ, the algorithm explores by choosing a random bandit, ensuring that over time, all options have a chance to be selected.
Reward Estimation:

After each trial, the reward obtained is used to update the estimated probability of success for the selected bandit. The estimated value is updated incrementally to reflect new data while retaining the influence of prior observations.
