import numpy as np

# Define the possible actions
ACTION_SPACE = ('U', 'D', 'L', 'R')  # Up, Down, Left, Right

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

def evaluate_deterministic_policy(grid, policy, gamma, initV=None):
    """
    Evaluate a deterministic policy using iterative policy evaluation.
    :param grid: The grid environment object.
    :param policy: Dictionary mapping each state to the selected action under the policy.
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
                a = policy.get(s, None)  # Get the action under the current policy
                if a is not None:
                    # Compute the expected value for the current policy's action
                    for s2, prob in grid.probs[(s, a)].items():
                        r = grid.rewards.get(s2, 0)
                        new_v += prob * (r + gamma * V[s2])
                V[s] = new_v  # Update value function
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))  # Track largest update

        # Stop if the value function has converged
        if biggest_change < convergence_threshold:
            break

    return V  # Return the converged value function

def probabilistic_policy_iteration(gamma, initial_policy="random", show_iteration=False):
    """
    Perform probabilistic policy iteration for the WindyGrid environment.
    :param gamma: Discount factor.
    :param initial_policy: Initial policy, either "random" or a predefined dictionary.
    :param show_iteration: If True, show the policy and value function at each iteration.
    :return: Final value function and policy after convergence.
    """
    # Initialize the WindyGrid environment
    grid = windy_grid()

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
        V = evaluate_deterministic_policy(grid, policy, gamma, initV=V)

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
                if (s, a) in grid.probs:
                    for s2, prob in grid.probs[(s, a)].items():
                        r = grid.rewards.get(s2, 0)
                        v += prob * (r + gamma * V[s2])

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
    probabilistic_policy_iteration(gamma, initial_policy="random", show_iteration=True)