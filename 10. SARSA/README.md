# SARSA: State-Action-Reward-State-Action

**SARSA** (State-Action-Reward-State-Action) is an **on-policy reinforcement learning algorithm** that enables an agent to learn the optimal policy in a Markov Decision Process (MDP). Being on-policy, SARSA evaluates and improves the policy that the agent is currently following during learning, rather than a separate, optimal policy. This means the algorithm updates its Q-values (action-value function) based on the actions taken by the current policy, including exploratory actions.

In simpler terms, SARSA not only learns from the environment but also from the actual behavior of the agent, ensuring that the learned values reflect the policy being followed. This approach is particularly useful in environments where adherence to a cautious or safe policy is crucial.

---

### Key Features
1. **On-Policy Learning**: 
   - SARSA learns the value of the current policy being followed, unlike off-policy methods like Q-Learning that evaluate the optimal policy independently of the behavior policy.

2. **Update Rule**:
   The Q-value update in SARSA is governed by the equation:
   $Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma Q(S', A') - Q(S, A) \right]$
   Where:
   - $( S, A )$: Current state and action.
   - $( R )$: Reward received after taking action $( A )$.
   - $( S', A' )$: Next state and action chosen by the policy.
   - $( \alpha )$: Learning rate.
   - $( \gamma )$: Discount factor.

3. **Exploration vs. Exploitation**:
   - SARSA typically uses an **epsilon-greedy policy** for action selection, allowing the agent to balance exploring new actions and exploiting the best-known actions.

---

### Advantages and Disadvantages

#### Advantages:
- Learns directly from the current policy, making it suitable for environments where cautious learning is important.
- Works well in stochastic environments or when safety is a priority.

#### Disadvantages:
- Convergence is slower compared to Q-Learning, especially in deterministic environments.
- Performance is sensitive to the exploration strategy (e.g., epsilon-greedy).

---

### How SARSA Works (Step-by-Step)

SARSA iteratively improves the Q-values and the policy by interacting with the environment. Below are the steps:

1. **Initialize**:
   - Set all Q-values $( Q(s, a) )$ to 0 (or small random values).
   - Define the exploration policy (e.g., epsilon-greedy).

2. **For Each Episode**:
   - Start in an initial state $( S )$.
   - Choose an action $( A )$ using the current policy (e.g., epsilon-greedy).
   - Repeat until the episode ends:
     1. Take action $( A )$, observe reward $( R )$ and next state $( S' )$.
     2. Choose next action $( A' )$ in $( S' )$ using the current policy.
     3. Update $Q(S, A)$:
        
        $Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma Q(S', A') - Q(S, A) \right]$
        
     5. Transition to $( S' )$ and $( A' )$.

3. **Update Policy**:
   - Derive the optimal policy from the learned Q-values.

---

### When to Use SARSA

SARSA is particularly useful when:
- The environment has **stochastic transitions** or **rewards**.
- Adhering to a **safe or cautious policy** is critical (e.g., avoiding dangerous states).
- You want to learn the value of the current policy rather than the optimal policy directly.

---

### Comparison with Q-Learning

| Feature               | SARSA                       | Q-Learning                |
|-----------------------|----------------------------|--------------------------|
| Type                 | On-Policy                  | Off-Policy               |
| Update Rule          | $Q(S, A)$ depends on the action $( A' )$ chosen by the current policy. | $( Q(S, A) )$ uses $( \max Q(S', a) )$ (best action in $( S' )$ ). |
| Exploration Strategy  | Directly affects learning. | Independent of the learned policy. |
| Convergence Speed    | Slower                     | Faster                   |
| Behavior             | More cautious, follows the current policy. | Can explore aggressively to find optimal policies. |

---

### Numerical Example of SARSA

#### Example Setup:
A **2x2 grid world**:
- States: $( S_1, S_2, S_3, S_4 )$ (numbered as grid positions).
- Actions: Up $(U)$, Down $(D)$, Left $(L)$, Right $(R)$.
- Start state: $( S_1 )$ (top-left corner).
- Goal state: $( S_4 )$ (bottom-right corner) with a reward of +10.
- Other states have a step cost of -1.
- The agent follows an epsilon-greedy policy.

#### Walkthrough:

| State | Actions | Next State | Reward |
|-------|---------|------------|--------|
| $( S_1 )$ | $R$ | $( S_2 )$ | -1 |
| $( S_2 )$ | $D$ | $( S_4 )$ | +10 |

**Initial Q-Table**:
Assume $( Q(S, A) = 0 )$ for all state-action pairs.

#### Episode Steps:
1. Start in $( S_1 )$, select $( A = R )$ (epsilon-greedy).
2. Move to $( S_2 )$, receive $( R = -1 )$, choose $( A' = D )$.
3. Move to $( S_4 )$, receive $( R = +10 )$ (goal state).
4. Update Q-values:
   - At $( (S_2, D) )$:
     $Q(S_2, D) = Q(S_2, D) + \alpha \left[ R + \gamma Q(S_4, \cdot) - Q(S_2, D) \right]$
     Since $( Q(S_4, \cdot) = 0 )$ (terminal state):
     $Q(S_2, D) = 0 + 0.1 \left[ 10 + 0 \times 0 - 0 \right] = 1.0$
   - At $( (S_1, R) )$:
     $Q(S_1, R) = 0 + 0.1 \left[ -1 + 0.9 \times Q(S_2, D) - Q(S_1, R) \right]$
     Using $( Q(S_2, D) = 1.0 )$:
     $Q(S_1, R) = 0 + 0.1 \left[ -1 + 0.9 \times 1 - 0 \right] = -0.01$

**Updated Q-Table After One Episode**:
| State | Action | Q-Value |
|-------|--------|---------|
| $( S_1 )$ | $R$ | -0.01   |
| $( S_2 )$ | $D$ | 1.0     |

### What This Example Teaches:
- SARSA updates $Q(s, a)$ based on the **current policy's actions**.
- Iterative updates refine the Q-values and improve the policy over time.
