import numpy as np

# Define possible actions in the environment
ACTION_SPACE = ('U', 'D', 'L', 'R')  # U: Up, D: Down, L: Left, R: Right

class Grid:  # Environment representing a grid world
  def __init__(self, rows, cols, start):
    """
    Initialize the grid environment with the given dimensions and starting position.
    :param rows: Number of rows in the grid
    :param cols: Number of columns in the grid
    :param start: Starting position (i, j) for the agent
    """
    self.rows = rows
    self.cols = cols
    self.i = start[0]
    self.j = start[1]

  def set(self, rewards, actions):
    """
    Set the rewards and possible actions for each cell in the grid.
    :param rewards: Dictionary where keys are (i, j) positions and values are rewards
    :param actions: Dictionary where keys are (i, j) positions and values are lists of allowed actions
    """
    self.rewards = rewards
    self.actions = actions

  def set_state(self, s):
    """
    Set the agent's position in the grid to a specific state.
    :param s: Tuple (i, j) representing the agent's new position
    """
    self.i = s[0]
    self.j = s[1]

  def current_state(self):
    """
    Returns the current state (position) of the agent.
    :return: Tuple (i, j) representing the agent's current position
    """
    return (self.i, self.j)

  def is_terminal(self, s):
    """
    Check if a state is terminal, meaning there are no actions available from this state.
    :param s: Tuple (i, j) representing a state
    :return: True if the state is terminal, False otherwise
    """
    return s not in self.actions

  def reset(self):
    """
    Reset the agent to the starting position.
    :return: Tuple (i, j) representing the agent's start position
    """
    self.i = 2
    self.j = 0
    return (self.i, self.j)

  def get_next_state(self, s, a):
    """
    Get the next state given a current state and an action.
    :param s: Current state (i, j) as a tuple
    :param a: Action to be taken ('U', 'D', 'L', 'R')
    :return: Tuple (new_i, new_j) representing the next state after taking action
    """
    i, j = s[0], s[1]

    # Update position based on the action, if the action is allowed from the current state
    if a in self.actions[(i, j)]:
      if a == 'U':
        i -= 1
      elif a == 'D':
        i += 1
      elif a == 'R':
        j += 1
      elif a == 'L':
        j -= 1
    return i, j

  def move(self, action):
    """
    Move the agent in the specified direction, if the action is allowed from the current position.
    :param action: Action to take ('U', 'D', 'L', 'R')
    :return: Reward received after taking the action
    """
    # Check if the action is valid in the current state
    if action in self.actions[(self.i, self.j)]:
      if action == 'U':
        self.i -= 1
      elif action == 'D':
        self.i += 1
      elif action == 'R':
        self.j += 1
      elif action == 'L':
        self.j -= 1
    # Return the reward for the new position, defaulting to 0 if no reward is defined
    return self.rewards.get((self.i, self.j), 0)

  def undo_move(self, action):
    """
    Undo the last move, moving the agent in the opposite direction of the action.
    :param action: Action to undo ('U', 'D', 'L', 'R')
    """
    # Move in the opposite direction of the specified action
    if action == 'U':
      self.i += 1
    elif action == 'D':
      self.i -= 1
    elif action == 'R':
      self.j -= 1
    elif action == 'L':
      self.j += 1
    # Ensure the state after undoing is valid
    assert(self.current_state() in self.all_states())

  def game_over(self):
    """
    Check if the game is over, which is true if the agent is in a terminal state.
    :return: True if in a terminal state, False otherwise
    """
    return (self.i, self.j) not in self.actions

  def all_states(self):
    """
    Get all possible states in the grid, defined as any position with actions or rewards.
    :return: Set of tuples representing all possible states
    """
    return set(self.actions.keys()) | set(self.rewards.keys())


def standard_grid():
  """
  Define a standard 3x4 grid with rewards and actions.
  Layout:
    .  .  .  1
    .  x  . -1
    s  .  .  .
  Legend:
    - s: Starting position
    - x: Blocked position (no actions allowed)
    - Numbers: Rewards at certain states

  :return: An instance of the Grid class with rewards and actions set
  """
  g = Grid(3, 4, (2, 0))  # 3x4 grid with start position at (2, 0)

  # Define rewards for reaching specific states
  rewards = {(0, 3): 1, (1, 3): -1}

  # Define possible actions from each state
  actions = {
    (0, 0): ('D', 'R'),  # Can go Down or Right from (0, 0)
    (0, 1): ('L', 'R'),  # Can go Left or Right from (0, 1)
    (0, 2): ('L', 'D', 'R'),  # Can go Left, Down, or Right from (0, 2)
    (1, 0): ('U', 'D'),  # Can go Up or Down from (1, 0)
    (1, 2): ('U', 'D', 'R'),  # Can go Up, Down, or Right from (1, 2)
    (2, 0): ('U', 'R'),  # Can go Up or Right from (2, 0)
    (2, 1): ('L', 'R'),  # Can go Left or Right from (2, 1)
    (2, 2): ('L', 'R', 'U'),  # Can go Left, Right, or Up from (2, 2)
    (2, 3): ('L', 'U'),  # Can go Left or Up from (2, 3)
  }

  # Set rewards and actions in the grid
  g.set(rewards, actions)
  return g

convergence_threshold = 1e-3  # Threshold for convergence in policy evaluation

# Define possible actions in the grid world
ACTION_SPACE = ('U', 'D', 'L', 'R')  # Up, Down, Left, Right

def print_values(V, g):
    """
    Print the value function in a grid layout.
    :param V: A dictionary mapping each state to its computed value.
    :param g: The grid environment object.
    """
    for i in range(g.rows):
        print("---------------------------")
        for j in range(g.cols):
            v = V.get((i, j), 0)  # Get the value for each state, defaulting to 0 if not in V
            if v >= 0:
                print(" %.2f|" % v, end="")  # Align positive values with two decimal places
            else:
                print("%.2f|" % v, end="")  # Align negative values with two decimal places
        print("")
    print("\n")  # Extra newline for better readability

def print_policy(P, g):
    """
    Print the policy in a grid layout.
    :param P: A dictionary mapping each state to an action ('U', 'D', 'L', 'R').
    :param g: The grid environment object.
    """
    for i in range(g.rows):
        print("---------------------------")
        for j in range(g.cols):
            a = P.get((i, j), ' ')  # Get the action for each state, or blank if not in policy
            print("  %s  |" % a, end="")
        print("")
    print("\n")  # Extra newline for better readability

def get_transition_probs_and_rewards(grid):
    """
    Extract the transition probabilities and rewards for the grid.
    :param grid: The grid environment object.
    :return: A dictionary of transition probabilities and rewards for each state-action pair.
    """
    transition_probs = {}  # Dictionary for transition probabilities
    rewards = {}           # Dictionary for rewards

    # Iterate over each cell in the grid
    for i in range(grid.rows):
        for j in range(grid.cols):
            s = (i, j)
            # Only define transitions for non-terminal states
            if not grid.is_terminal(s):
                for a in ACTION_SPACE:
                    s2 = grid.get_next_state(s, a)  # Next state based on action
                    transition_probs[(s, a, s2)] = 1  # Deterministic transition
                    if s2 in grid.rewards:
                        rewards[(s, a, s2)] = grid.rewards[s2]  # Assign reward if defined

    return transition_probs, rewards

def evaluate_deterministic_policy(grid, policy, transition_probs, rewards, gamma, initV=None):
    """
    Evaluate a deterministic policy using iterative policy evaluation.
    :param grid: The grid environment object.
    :param policy: Dictionary mapping each state to the selected action under the policy.
    :param transition_probs: Dictionary of transition probabilities for each state-action pair.
    :param rewards: Dictionary of rewards for each state-action-next_state triplet.
    :param gamma: Discount factor.
    :param initV: Optional initial value function, used to start evaluation with a prior estimate.
    :return: Final value function V after policy evaluation.
    """
    # Initialize the value function V for all states, using initV if provided
    V = initV if initV is not None else {s: 0 for s in grid.all_states()}

    # Iterative policy evaluation loop
    while True:
        biggest_change = 0  # Track the largest change in V for convergence check
        for s in grid.all_states():
            if not grid.is_terminal(s):
                old_v = V[s]
                new_v = 0  # Calculate new value for state s
                for a in ACTION_SPACE:
                    for s2 in grid.all_states():
                        action_prob = 1 if policy.get(s) == a else 0
                        r = rewards.get((s, a, s2), 0)
                        new_v += action_prob * transition_probs.get((s, a, s2), 0) * (r + gamma * V[s2])
                V[s] = new_v  # Update value function
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))  # Track largest update

        # Stop if the value function has converged
        if biggest_change < convergence_threshold:
            break

    return V  # Return the converged value function

def deterministic_policy_iteration(gamma, initial_policy="random", show_iteration=False):
    """
    Perform policy iteration for the grid environment.
    :param gamma: Discount factor.
    :param initial_policy: Initial policy, either "random" or a predefined dictionary.
    :param show_iteration: If True, show the policy and value function at each iteration.
    :return: Final value function and policy after convergence.
    """
    # Initialize the grid environment
    grid = standard_grid()

    # Get transition probabilities and rewards
    transition_probs, rewards = get_transition_probs_and_rewards(grid)

    # Initialize the policy
    if initial_policy == "random":
        policy = {s: np.random.choice(ACTION_SPACE) for s in grid.actions.keys()}
    else:
        policy = initial_policy

    # Print rewards in the grid layout
    print("Rewards:")
    print_values(grid.rewards, grid)

    # Print initial policy layout
    print("Initial Policy:")
    print_policy(policy, grid)

    # Policy iteration loop
    V = None  # Initialize value function V
    iteration = 0
    while True:
        # Optionally print iteration details
        if show_iteration:
            print(f"Iteration {iteration}: Policy Improvement")

        # Policy evaluation step: evaluate the current policy
        V = evaluate_deterministic_policy(grid, policy, transition_probs, rewards, gamma, initV=V)

        # Optionally print the value function after evaluation
        if show_iteration:
            print("Value Function:")
            print_values(V, grid)

        # Policy improvement step
        is_policy_converged = True  # Track if policy has converged
        for s in grid.actions.keys():
            old_a = policy[s]  # Current action in the policy
            new_a = None
            best_value = float('-inf')  # Initialize best value for action selection

            # Loop through all actions to find the best action under current V
            for a in ACTION_SPACE:
                v = 0
                for s2 in grid.all_states():
                    r = rewards.get((s, a, s2), 0)
                    v += transition_probs.get((s, a, s2), 0) * (r + gamma * V[s2])

                if v > best_value:
                    best_value = v
                    new_a = a

            # Update the policy with the best action found
            policy[s] = new_a
            if new_a != old_a:
                is_policy_converged = False  # If any action changes, policy is not yet converged

        # Optionally print the policy after improvement step
        if show_iteration:
            print("Policy:")
            print_policy(policy, grid)

        # Exit if policy has converged
        if is_policy_converged:
            break
        iteration += 1

    # Print final value function and policy after convergence
    print("Final Value Function:")
    print_values(V, grid)
    print("Final Policy:")
    print_policy(policy, grid)

    return V, policy  # Return final value function and policy

if __name__ == '__main__':
    gamma = 0.9  # Define the discount factor
    # Run policy iteration with a random initial policy and display each iteration
    deterministic_policy_iteration(gamma, initial_policy="random", show_iteration=True)