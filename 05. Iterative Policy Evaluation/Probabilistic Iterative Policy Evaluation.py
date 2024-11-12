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

import numpy as np

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
    """
    Print the policy for the grid. For each state, display the action(s) with probabilities.
    If there's a single action with a 1.0 probability, display just that action.
    """
    print("Policy:")
    for i in range(g.rows):
        print("---------------------------")
        for j in range(g.cols):
            actions = P.get((i, j), {})
            if actions:
                # If there's only one action with probability 1.0, display that action alone
                if len(actions) == 1 and list(actions.values())[0] == 1.0:
                    action = list(actions.keys())[0]
                    print("  %s  |" % action, end="")
                else:
                    # Display all actions with their probabilities
                    action_probs = ", ".join([f"{a}:{p:.1f}" for a, p in actions.items()])
                    print(" %s |" % action_probs, end="")
            else:
                # If no actions are defined, print an empty space or terminal state indicator
                print("     |", end="")
        print("")
    print("\n")  # Extra newline for better readability


def Probabilistic_iterative_policy_evaluation(policy, gamma):
    """
    Perform iterative policy evaluation for a given probabilistic policy and discount factor gamma.
    :param policy: Dictionary mapping each state (i, j) to a dict of actions with probabilities
    :param gamma: Discount factor
    """
    # Initialize the grid environment
    grid = windy_grid()

    # Define transition probabilities and rewards from the grid
    transition_probs = {}
    rewards = {}
    for (s, a), transitions in grid.probs.items():
        for s2, prob in transitions.items():
            transition_probs[(s, a, s2)] = prob
            rewards[(s, a, s2)] = grid.rewards.get(s2, 0)

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

                # Iterate over all actions in the action space
                for a in ACTION_SPACE:
                    # Probability of taking action `a` under the policy for state `s`
                    action_prob = policy.get(s, {}).get(a, 0)

                    # Sum up for each possible next state `s2`
                    for s2 in grid.all_states():
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
    # Define a probabilistic policy for testing
    policy = {
        (2, 0): {'U': 0.5, 'R': 0.5},
        (1, 0): {'U': 1.0},
        (0, 0): {'R': 1.0},
        (0, 1): {'R': 1.0},
        (0, 2): {'R': 1.0},
        (1, 2): {'U': 1.0},
        (2, 1): {'R': 1.0},
        (2, 2): {'U': 1.0},
        (2, 3): {'L': 1.0},
    }
    gamma = 0.9  # Discount factor

    # Run iterative policy evaluation
    Probabilistic_iterative_policy_evaluation(policy, gamma)