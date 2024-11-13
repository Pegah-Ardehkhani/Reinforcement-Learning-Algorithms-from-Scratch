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

class WindyGrid:
    def __init__(self, rows, cols, start):
        """
        Initialize the grid with given dimensions and starting position.

        Parameters:
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        start (tuple): Starting position (row, col).
        """
        self.rows = rows
        self.cols = cols
        self.i = start[0]  # Row position of the agent
        self.j = start[1]  # Column position of the agent

    def set(self, rewards, actions, probs):
        """
        Set the rewards, actions, and transition probabilities for each state.

        Parameters:
        rewards (dict): Dictionary mapping each state (i, j) to its reward.
        actions (dict): Dictionary mapping each state (i, j) to available actions.
        probs (dict): Dictionary defining the transition probabilities p(s' | s, a).
        """
        self.rewards = rewards
        self.actions = actions
        self.probs = probs

    def set_state(self, s):
        """
        Set the agent's current state.

        Parameters:
        s (tuple): The state to set (i, j).
        """
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        """
        Return the agent's current state.

        Returns:
        tuple: The current state (i, j).
        """
        return (self.i, self.j)

    def is_terminal(self, s):
        """
        Check if a given state is terminal (no actions available).

        Parameters:
        s (tuple): The state to check.

        Returns:
        bool: True if the state is terminal, False otherwise.
        """
        return s not in self.actions

    def move(self, action):
        """
        Attempt to move the agent in the specified direction, using transition probabilities.

        Parameters:
        action (str): The action to take ('U', 'D', 'L', or 'R').

        Returns:
        int: The reward associated with the resulting state (if any).
        """
        # Get the current state
        s = (self.i, self.j)

        # Get the transition probabilities for the action
        next_state_probs = self.probs[(s, action)]
        next_states = list(next_state_probs.keys())    # Possible next states
        next_probs = list(next_state_probs.values())   # Corresponding probabilities

        # Choose the next state based on the defined probabilities
        next_state_idx = np.random.choice(len(next_states), p=next_probs)
        s2 = next_states[next_state_idx]

        # Update the agent's current position
        self.i, self.j = s2

        # Return the reward for the resulting state (if any)
        return self.rewards.get(s2, 0)

    def game_over(self):
        """
        Check if the game is over (if the agent is in a terminal state).

        Returns:
        bool: True if the game is over, False otherwise.
        """
        return (self.i, self.j) not in self.actions

    def all_states(self):
        """
        Get a set of all possible states, including states with rewards and states with available actions.

        Returns:
        set: A set containing all states in the grid.
        """
        return set(self.actions.keys()) | set(self.rewards.keys())


def windy_grid():
    """
    Create a specific instance of the WindyGrid with predefined rewards, actions, and transition probabilities.

    Returns:
    WindyGrid: An instance of the WindyGrid class.
    """
    # Initialize the grid with 3 rows, 4 columns, and a starting position at (2, 0)
    g = WindyGrid(3, 4, (2, 0))

    # Define rewards for certain states
    rewards = {(0, 3): 1, (1, 3): -1}  # Positive reward at (0, 3), negative reward at (1, 3)

    # Define actions available at each non-terminal state
    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'D', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('L', 'R', 'U'),
        (2, 3): ('L', 'U'),
    }

    # Define transition probabilities p(s' | s, a)
    # Each entry specifies possible outcomes when performing an action in a given state
    probs = {
        ((2, 0), 'U'): {(1, 0): 1.0},
        ((2, 0), 'D'): {(2, 0): 1.0},
        ((2, 0), 'L'): {(2, 0): 1.0},
        ((2, 0), 'R'): {(2, 1): 1.0},
        ((1, 0), 'U'): {(0, 0): 1.0},
        ((1, 0), 'D'): {(2, 0): 1.0},
        ((1, 0), 'L'): {(1, 0): 1.0},
        ((1, 0), 'R'): {(1, 0): 1.0},
        ((0, 0), 'U'): {(0, 0): 1.0},
        ((0, 0), 'D'): {(1, 0): 1.0},
        ((0, 0), 'L'): {(0, 0): 1.0},
        ((0, 0), 'R'): {(0, 1): 1.0},
        ((0, 1), 'U'): {(0, 1): 1.0},
        ((0, 1), 'D'): {(0, 1): 1.0},
        ((0, 1), 'L'): {(0, 0): 1.0},
        ((0, 1), 'R'): {(0, 2): 1.0},
        ((0, 2), 'U'): {(0, 2): 1.0},
        ((0, 2), 'D'): {(1, 2): 1.0},
        ((0, 2), 'L'): {(0, 1): 1.0},
        ((0, 2), 'R'): {(0, 3): 1.0},
        ((2, 1), 'U'): {(2, 1): 1.0},
        ((2, 1), 'D'): {(2, 1): 1.0},
        ((2, 1), 'L'): {(2, 0): 1.0},
        ((2, 1), 'R'): {(2, 2): 1.0},
        ((2, 2), 'U'): {(1, 2): 1.0},
        ((2, 2), 'D'): {(2, 2): 1.0},
        ((2, 2), 'L'): {(2, 1): 1.0},
        ((2, 2), 'R'): {(2, 3): 1.0},
        ((2, 3), 'U'): {(1, 3): 1.0},
        ((2, 3), 'D'): {(2, 3): 1.0},
        ((2, 3), 'L'): {(2, 2): 1.0},
        ((2, 3), 'R'): {(2, 3): 1.0},
        ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},  # Probabilistic transition
        ((1, 2), 'D'): {(2, 2): 1.0},
        ((1, 2), 'L'): {(1, 2): 1.0},
        ((1, 2), 'R'): {(1, 3): 1.0},
    }

    # Set the rewards, actions, and transition probabilities in the grid
    g.set(rewards, actions, probs)

    # Return the configured grid instance
    return g

# Convergence threshold for value iteration
convergence_threshold = 1e-3

# Define possible actions in the grid world
ACTION_SPACE = ('U', 'D', 'L', 'R')  # U: Up, D: Down, L: Left, R: Right

def print_values(V, g):
    """
    Print the value function in a grid layout.
    :param V: Dictionary mapping each state to its value.
    :param g: Grid object, used for dimensions and layout.
    """
    for i in range(g.rows):
        print("---------------------------")
        for j in range(g.cols):
            v = V.get((i, j), 0)  # Get the value for each state, default to 0 if not in V
            if v >= 0:
                print(" %.2f|" % v, end="")  # Format for positive values
            else:
                print("%.2f|" % v, end="")  # Format for negative values
        print("")
    print("\n")

def print_policy(P, g):
    """
    Print the policy in a grid layout.
    :param P: Dictionary mapping each state to the optimal action under the policy.
    :param g: Grid object, used for dimensions and layout.
    """
    for i in range(g.rows):
        print("---------------------------")
        for j in range(g.cols):
            a = P.get((i, j), ' ')  # Get the action for each state, default to blank if not in policy
            print("  %s  |" % a, end="")
        print("")
    print("\n")

def get_transition_probs_and_rewards(grid):
    """
    Extract the transition probabilities and rewards for both deterministic and probabilistic grids.
    :param grid: Grid object, could be standard_grid or windy_grid.
    :return: transition_probs and rewards dictionaries.
    """
    transition_probs = {}  # Dictionary to store transition probabilities
    rewards = {}           # Dictionary to store rewards for each state-action-next_state

    # Determine if the grid is probabilistic by checking for the 'probs' attribute
    is_probabilistic = hasattr(grid, 'probs')

    for s in grid.all_states():
        if not grid.is_terminal(s):
            for a in ACTION_SPACE:
                # For probabilistic environments (WindyGrid), use predefined probabilities
                if is_probabilistic:
                    if (s, a) in grid.probs:
                        transition_probs[(s, a)] = grid.probs[(s, a)]
                else:
                    # For deterministic environments, assume a single transition with probability 1
                    s2 = grid.get_next_state(s, a)
                    transition_probs[(s, a)] = {s2: 1.0}

                # Record rewards if defined in the grid's reward structure
                for next_state in transition_probs[(s, a)]:
                    if next_state in grid.rewards:
                        rewards[(s, a, next_state)] = grid.rewards[next_state]

    return transition_probs, rewards

def value_iteration(gamma, grid_fn, show_iteration=False):
    """
    Perform value iteration to find the optimal value function and policy for deterministic and probabilistic grids.
    :param gamma: Discount factor for future rewards.
    :param grid_fn: Function to initialize the grid (either standard_grid or windy_grid).
    :param show_iteration: If True, display the value function after each iteration.
    :return: Final value function and optimal policy after convergence.
    """
    # Initialize the grid using the provided function (standard_grid or windy_grid)
    grid = grid_fn()
    transition_probs, rewards = get_transition_probs_and_rewards(grid)

    # Initialize value function V for all states
    V = {s: 0 for s in grid.all_states()}

    # Print rewards in the grid layout
    print("Rewards:")
    print_values(grid.rewards, grid)

    # Value Iteration loop
    iteration = 0
    while True:
        biggest_change = 0  # Track the maximum change in V for convergence check
        for s in grid.all_states():
            if not grid.is_terminal(s):
                old_v = V[s]  # Store old value for convergence check
                new_v = float('-inf')  # Initialize with negative infinity for max operation

                # Bellman optimality update for each action
                for a in ACTION_SPACE:
                    v = 0
                    if (s, a) in transition_probs:
                        # Calculate expected value of taking action a in state s
                        for s2, prob in transition_probs[(s, a)].items():
                            r = rewards.get((s, a, s2), 0)
                            v += prob * (r + gamma * V[s2])
                        new_v = max(new_v, v)  # Take the max over actions

                V[s] = new_v  # Update value for the state
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        # Optionally print the value function after each iteration
        if show_iteration:
            print(f"Iteration {iteration}: Value Function")
            print_values(V, grid)

        # Stop if the value function has converged
        if biggest_change < convergence_threshold:
            break
        iteration += 1

    # Derive the optimal policy from the converged value function
    policy = {}
    for s in grid.actions.keys():
        best_action = None
        best_value = float('-inf')

        # Find the best action for each state based on V
        for a in ACTION_SPACE:
            v = 0
            if (s, a) in transition_probs:
                # Calculate expected value for taking action a in state s
                for s2, prob in transition_probs[(s, a)].items():
                    r = rewards.get((s, a, s2), 0)
                    v += prob * (r + gamma * V[s2])
                if v > best_value:
                    best_value = v
                    best_action = a  # Update best action

        policy[s] = best_action  # Set the best action for the state

    # Print final value function and optimal policy
    print("Final Value Function:")
    print_values(V, grid)
    print("Optimal Policy:")
    print_policy(policy, grid)

    return V, policy

# Run value iteration
if __name__ == '__main__':
    gamma = 0.9  # Define the discount factor
    # Run value iteration for both standard and windy grids
    print("Running Value Iteration on Windy Grid:")
    value_iteration(gamma, windy_grid, show_iteration=True)
    print("Running Value Iteration on Standard Grid:")
    value_iteration(gamma, standard_grid, show_iteration=True)