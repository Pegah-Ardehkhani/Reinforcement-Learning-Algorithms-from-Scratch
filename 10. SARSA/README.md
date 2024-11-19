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
| Update Rule          | $Q(S, A)$ depends on the action $( A' )$ chosen by the current policy. | $Q(S, A)$ uses $( \max Q(S', a) )$ (best action in $S'$ ). |
| Exploration Strategy  | Directly affects learning. | Independent of the learned policy. |
| Convergence Speed    | Slower                     | Faster                   |
| Behavior             | More cautious, follows the current policy. | Can explore aggressively to find optimal policies. |

---

### Numerical Example of SARSA

#### Example Setup:
A **2x2 grid world**:
- **States**: $( S_1, S_2, S_3, S_4 )$.
- **Actions**: Up $(U)$, Down $(D)$, Left $(L)$, Right $(R)$.
- **Rewards**:
  - $( S_4 )$: Goal state, reward = $( +10 )$.
  - Other states have a step cost = $( -1 )$.
- **Start state**: $( S_1 )$ (top-left corner).
- **Policy**: The agent follows an epsilon-greedy policy.

**Grid Layout**:

| State  | State  |
|--------|--------|
| $( S_1 )$ | $( S_2 )$ |
| $( S_3 )$ | $( S_4 )$ |

#### Initial Q-Table:
- $( Q(s, a) = 0 )$ for all state-action pairs.

#### Step-by-Step Walkthrough:

1. **Episode Start**:
   - Start at $( S_1 )$.
   - Use epsilon-greedy to choose $( A = R )$.

2. **First Transition**:
   - Take action $( R )$ from $( S_1 )$.
   - Move to $( S_2 )$, receive $( R = -1 )$.
   - Choose $( A' = D )$ in $( S_2 )$ using epsilon-greedy.

   **Update $( Q(S_1, R) )$:**
   
   $Q(S_1, R) \leftarrow Q(S_1, R) + \alpha \left[ R + \gamma Q(S_2, D) - Q(S_1, R) \right]$
   
   Substituting values:
   - $( Q(S_1, R) = 0 )$, $( R = -1 )$, $( Q(S_2, D) = 0 )$, $( \alpha = 0.1 )$, $( \gamma = 0.9 )$:
   
   $Q(S_1, R) \leftarrow 0 + 0.1 \left[ -1 + 0.9 \times 0 - 0 \right] = -0.1$

4. **Second Transition**:
   - Take action $( D )$ from $( S_2 )$.
   - Move to $( S_4 )$, receive $( R = +10 )$.
   - $( S_4 )$ is terminal, so $( Q(S_4, \cdot) = 0 )$.

   **Update $( Q(S_2, D) )$:**
   
   $Q(S_2, D) \leftarrow Q(S_2, D) + \alpha \left[ R + \gamma Q(S_4, \cdot) - Q(S_2, D) \right]$
   
   Substituting values:
   - $( Q(S_2, D) = 0 )$, $( R = +10 )$, $( Q(S_4, \cdot) = 0 )$, $( \alpha = 0.1 )$, $( \gamma = 0.9 )$:
   
   $Q(S_2, D) \leftarrow 0 + 0.1 \left[ 10 + 0 - 0 \right] = 1.0$

#### Updated Q-Table After One Episode:
| State    | Action | Q-Value |
|----------|--------|---------|
| $( S_1 )$ | $( R )$ | $-0.1$    |
| $( S_2 )$ | $( D )$ | $1.0$     |
| All other states/actions | - | $0$       |

### What This Example Teaches:
- SARSA updates $Q(s, a)$ based on the **current policy's actions**.
- Iterative updates refine the Q-values and improve the policy over time.








