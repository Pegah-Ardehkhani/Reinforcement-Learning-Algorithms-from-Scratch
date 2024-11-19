# SARSA: State-Action-Reward-State-Action

SARSA (State-Action-Reward-State-Action) is an **on-policy reinforcement learning algorithm** used to learn the optimal policy in a Markov Decision Process (MDP). The algorithm updates the Q-values (action-value function) by interacting with the environment and following the current policy.

#### Key Features of SARSA:
1. **On-Policy Learning**: 
   - The Q-value updates are based on the action chosen by the current policy. This makes SARSA an on-policy algorithm, as it learns the value of the policy being followed during the exploration.

2. **Update Rule**:
   The Q-value update in SARSA is governed by the equation:
   \[
   Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma Q(S', A') - Q(S, A) \right]
   \]
   Where:
   - \( S, A \): Current state and action.
   - \( R \): Reward received after taking action \( A \).
   - \( S', A' \): Next state and action chosen by the policy.
   - \( \alpha \): Learning rate.
   - \( \gamma \): Discount factor.

3. **Exploration-Exploitation Balance**:
   - SARSA typically uses an **epsilon-greedy policy** for action selection, which balances exploration (trying new actions) and exploitation (choosing the best-known action).

4. **Behavior**:
   - As an on-policy algorithm, SARSA accounts for the policy being followed, making it sensitive to the exploration strategy.

#### Example Use Case:
SARSA is ideal for environments where it is crucial to learn a safe or cautious policy, as it directly evaluates the policy being followed rather than the optimal policy (like in Q-Learning).

#### Comparison with Q-Learning:
- SARSA updates are based on the action taken by the current policy (\( A' \)), while Q-Learning updates are based on the best action (\( \max Q(S', a) \)) regardless of the current policy. This makes SARSA more policy-aware but sometimes slower to converge.
