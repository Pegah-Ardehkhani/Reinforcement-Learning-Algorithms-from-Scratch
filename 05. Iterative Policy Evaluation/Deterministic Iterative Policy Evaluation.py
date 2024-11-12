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

convergence_threshold = 1e-3  # Threshold for convergence

def print_values(V, g, label=True):
    if label:
        print("Value Function:")
    for i in range(g.rows):
        print("---------------------------")
        for j in range(g.cols):
            v = V.get((i, j), 0)
            if v >= 0:
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="")  # -ve sign takes up an extra space
        print("")
    print("\n")  # Extra newline for better readability

def print_policy(P, g):
    print("Policy:")
    for i in range(g.rows):
        print("---------------------------")
        for j in range(g.cols):
            a = P.get((i, j), ' ')
            print("  %s  |" % a, end="")
        print("")
    print("\n")  # Extra newline for better readability

def deterministic_iterative_policy_evaluation(policy, gamma):
    """
    Perform iterative policy evaluation for a given policy and discount factor gamma.
    :param policy: Dictionary mapping each state (i, j) to an action ('U', 'D', 'L', 'R')
    :param gamma: Discount factor
    """
    # Initialize the grid environment
    grid = standard_grid()

    # Define transition probabilities and rewards
    transition_probs = {}
    rewards = {}
    for i in range(grid.rows):
        for j in range(grid.cols):
            s = (i, j)
            if not grid.is_terminal(s):
                for a in ACTION_SPACE:
                    s2 = grid.get_next_state(s, a)
                    transition_probs[(s, a, s2)] = 1  # Deterministic transitions
                    if s2 in grid.rewards:
                        rewards[(s, a, s2)] = grid.rewards[s2]

    # Print the initial policy
    print_policy(policy, grid)

    # Initialize V(s) = 0 for all states
    V = {s: 0 for s in grid.all_states()}

    # Iterative Policy Evaluation
    it = 0
    while True:
        biggest_change = 0
        for s in grid.all_states():
            if not grid.is_terminal(s):
                old_v = V[s]
                new_v = 0  # Accumulate value based on policy

                for a in ACTION_SPACE:
                    for s2 in grid.all_states():
                        # Probability of taking the action from the policy
                        action_prob = 1 if policy.get(s) == a else 0

                        # Reward for (s, a, s')
                        r = rewards.get((s, a, s2), 0)

                        # Calculate the expected value
                        new_v += action_prob * transition_probs.get((s, a, s2), 0) * (r + gamma * V[s2])

                # Update the value function
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        # Print iteration results
        print(f"Iteration: {it}, Biggest Change: {biggest_change:.6f}")
        print_values(V, grid)

        # Check for convergence
        it += 1
        if biggest_change < convergence_threshold:
            break

    # Final print for the converged value function
    print("Final Value Function:")
    print_values(V, grid, label=False)  # Call without label


if __name__ == '__main__':
    # Define a sample policy for testing
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'U',
        (2, 1): 'R',
        (2, 2): 'U',
        (2, 3): 'L',
    }
    gamma = 0.9  # Discount factor

    # Run iterative policy evaluation
    deterministic_iterative_policy_evaluation(policy, gamma)