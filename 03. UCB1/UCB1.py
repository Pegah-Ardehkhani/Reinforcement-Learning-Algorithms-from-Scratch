import matplotlib.pyplot as plt
import numpy as np


class Bandit:
    def __init__(self, p):
        """
        Initialize a bandit with a given win probability 'p'.
        Attributes:
        - p: The true probability of winning (unknown to the agent).
        - p_estimate: The agent's current estimate of this probability.
        - N: Number of trials
        """
        self.p = p
        self.p_estimate = 0.0  # Start with an initial estimate of 0.
        self.N = 0  # Num of samples collected so far. Start with an initial value of 0.

    def pull(self):
        """
        Simulates pulling the bandit's arm.
        Returns 1 (win) with probability 'p', and 0 (loss) otherwise.
        """
        return np.random.random() < self.p  # Random number in [0,1], true if less than p

    def update(self, x):
        """
        Updates the agent's estimate of the bandit's win rate.
        Formula used:
        p_estimate = ((total_trials-1) * current_p_estimate + current_reward) / total_trials
        """
        self.N += 1.0  # Increment the number of samples collected
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N


def ucb1(bandit_probabilities, num_trials):
    """
    Runs the UCB1 algorithm to learn the best bandit over a number of trials.
    Parameters:
    - bandit_probabilities: The true win probabilities of the bandits.
    - num_trials: The total number of trials (or pulls).
    """
    bandits = [Bandit(p) for p in bandit_probabilities]  # Initialize bandits with true probabilities
    rewards = np.empty(num_trials)  # Array to store rewards for each trial
    optimal_j = np.argmax(bandit_probabilities)  # The index of the optimal (highest true probability) bandit
    num_optimal = 1  # Track how often the best true bandit is chosen (Set to 1 because of one step of initialization)
    total_plays = 0  # Initialize the total number of plays across all bandits to 0

    # Initialization step: Play each bandit (arm) once to gather initial rewards
    for j in range(len(bandits)):
      x = bandits[j].pull()  # Pull the j-th bandit (arm) to get a reward 'x'
      total_plays += 1  # Increment the total number of plays by 1
      bandits[j].update(x)  # Update the bandit's statistics (mean reward, number of pulls) with the new reward 'x'

    for i in range(num_trials):
        j = np.argmax([(b.p_estimate + np.sqrt(2*np.log(total_plays) / b.N)) for b in bandits])
        total_plays += 1  # Increment if we play an bandit
        # Pull the selected bandit's arm
        x = bandits[j].pull()
        rewards[i] = x  # Store the reward for this trial (1 for win, 0 for loss)

        # Update the estimate of the bandit based on the result
        bandits[j].update(x)

        if j == optimal_j:
            num_optimal += 1  # Increment if we chose the true optimal bandit


    # Compute statistics for the experiment
    mean_estimates = [b.p_estimate for b in bandits]  # The final estimated probabilities for each bandit
    total_reward = rewards.sum()  # The total reward earned over all trials
    overall_win_rate = total_reward / num_trials  # The overall win rate (reward per trial)


    # Print out experiment results
    print("Optimal bandit index:", np.argmax(mean_estimates))
    print("True optimal bandit index:", optimal_j)
    print("Mean estimates for each bandit:", mean_estimates)
    print("Total reward earned:", total_reward)
    print("Overall win rate:", overall_win_rate)
    print("Number of times selected true optimal bandit:", num_optimal)
    print("Number of times selected each bandit:", [int(b.N) for b in bandits])

    # Plot the results for visualization
    plot_results(rewards, bandit_probabilities, num_trials)


def plot_results(rewards, bandit_probabilities, num_trials):
    """
    Plots the win rates over time and the optimal win rate for comparison.
    Parameters:
    - rewards: The rewards obtained during the experiment.
    - bandit_probabilities: The true win probabilities of the bandits.
    - num_trials: The total number of trials (or pulls).
    """
    cumulative_rewards = np.cumsum(rewards)  # Cumulative sum of rewards at each trial
    win_rates = cumulative_rewards / (np.arange(num_trials) + 1)  # Win rate at each point in time

    plt.figure(figsize=(12, 6))
    plt.plot(win_rates, label='Win Rate')  # Plot the computed win rate
    plt.plot(np.ones(num_trials) * np.max(bandit_probabilities),
             linestyle='--', color='red', label='Optimal Win Rate')  # Plot the best possible win rate
    plt.xlabel('Trials')
    plt.ylabel('Win Rate')
    plt.title('UCB1 Bandit Algorithm Performance')
    plt.legend(loc=4)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]  # The true probabilities of each bandit
    NUM_TRIALS = 10000  # Number of trials to simulate
    ucb1(BANDIT_PROBABILITIES, NUM_TRIALS)