import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim

from ppo_bot import MLP
from ppo_bot import ActorCritic
from ppo_bot import train
from ppo_bot import evaluate
from car_racing import CarRacing

# speed, 7 angles (45, 15, 5, 0), angle of wheel, 6 ahead road segment angles (3, 5, 7, 10, 15, 20)
INPUT_DIM = 15
HIDDEN_DIM_1 = 512  # next: 256
HIDDEN_DIM_2 = 256
OUTPUT_DIM = 5  # no action, left, right, accel, brake

LEARNING_RATE = 0.01
MAX_EPISODES = 5_000_000_000
DISCOUNT_FACTOR = 0.99
PPO_STEPS = 5
PPO_CLIP = 0.2

REWARD_THRESHOLD = 1100
# REWARD_THRESHOLD = 908
# REWARD_THRESHOLD = 200
TEST_EVERY = 100
N_TRIALS = 25
PRINT_EVERY = 10

plt.ion()
track_seed = 123
# track_seed = random.randint(1, 100000)
# track_seed = 12147  # first corner sharp high speed

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


def run_rl_bot():
    start = time.time()
    # rendering mode speed: human: 18.1s, rgb_array: 9.4, state_pixels: 7.8, none: 6.8
    train_env = CarRacing(render_mode="none", continuous=False)
    test_env = CarRacing(render_mode="human", continuous=False)
    print(f'track_seed: {track_seed}')
    train_env.reset(seed=track_seed)
    test_env.reset(seed=track_seed)

    actor = MLP(INPUT_DIM, HIDDEN_DIM_1, HIDDEN_DIM_2, OUTPUT_DIM)
    critic = MLP(INPUT_DIM, HIDDEN_DIM_1, HIDDEN_DIM_2, 1)

    policy = ActorCritic(actor, critic)

    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    train_rewards = []
    test_rewards = []
    test_reward = 0

    for episode in range(1, MAX_EPISODES + 1):

        policy_loss, value_loss, train_reward = train(train_env, track_seed,
                                                      policy, optimizer, DISCOUNT_FACTOR,
                                                      PPO_STEPS, PPO_CLIP)

        if episode % TEST_EVERY == 0:
            test_reward = evaluate(test_env, track_seed, policy)

        train_rewards.append(train_reward)
        test_rewards.append(test_reward)

        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])

        if episode % PRINT_EVERY == 0:
            print(
                f'| Episode: {episode:3} | Mean Train Rewards: '
                f'{mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
            plot_durations(test_rewards=test_rewards, train_rewards=train_rewards)

        if mean_test_rewards >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')

    plot_durations(test_rewards=test_rewards, train_rewards=train_rewards, show_result=True)
    plt.ioff()
    plt.show()
    train_env.close()
    test_env.close()
    print(f'total time: {time.time() - start}')


def plot_durations(test_rewards, train_rewards, show_result=False):
    fig = plt.figure(1, figsize=(9, 6))
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title(f'Training... dim: {HIDDEN_DIM_1}x{HIDDEN_DIM_2}')

    plt.xlabel(f'Episode ({len(train_rewards)})', fontsize=20)
    plt.ylabel(f'Reward (max test: {max(test_rewards):5.1f})', fontsize=20)
    plt.plot(train_rewards, label='Train Reward')
    plt.plot(test_rewards, label='Test Reward')
    # plt.hlines(REWARD_THRESHOLD, 0, len(test_rewards), color='r')
    plt.legend(loc='upper left')

    fig.canvas.start_event_loop(0.001)  # this updates the plot and doesn't steal window focus
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


if __name__ == "__main__":
    run_rl_bot()
