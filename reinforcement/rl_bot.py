import time

import numpy as np

from reinforcement.car_racing import CarRacing


def run_rl_bot():
    start = time.time()
    game = CarRacing(render_mode="human")  # human: 18.1s, rgb_array: 9.4, state_pixels: 7.8, none: 6.8
    seed = 123
    # seed = random.randint(1, 100000)
    # seed = 12147  # first corner sharp high speed
    print(f'seed: {seed}')
    game.reset(seed=seed)

    action = np.array([0, +1.0, 0])
    state, step_reward, terminated, truncated, info = game.step(action)
    steps = 1
    total_reward = 0.0

    while not terminated:
        action = get_next_action(info['angle_distances'], info['car'])

        state, step_reward, terminated, truncated, info = game.step(action)
        total_reward += step_reward
        if steps % 200 == 0 or terminated or truncated:
            print("\naction " + str([f"{x:+0.2f}" for x in action]))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        if terminated or truncated:
            break
    game.close()
    print(f'total time: {time.time() - start}')


# distances angles are +-105, 90, 45, 15, 5, 0
def get_next_action(distances, car):
    braking_point_corner_distance = 40
    max_turning_speed = 70  # 60, 906.79; 65, 907.69; 70, 908.29
    sharp_turn_speed = 55  # 55, 908.29
    narrow_distance_buffer = 25
    medium_distance_buffer = 5
    sharp_distance_buffer = 5

    # if on grass: coast
    if len(distances) == 0:
        return np.array([0, 0, 0])

    speed = get_speed(car)
    # brake for upcoming corner
    if speed > max_turning_speed and distances[0] <= braking_point_corner_distance:
        return np.array([0, 0, 0.5])

    narrow_dist = max(distances[-5], distances[0], distances[5])
    medium_dist = max(distances[-15], distances[15]) - medium_distance_buffer
    sharp_dist = max(distances[-45], distances[45]) - sharp_distance_buffer

    # sharp turn
    if sharp_dist > narrow_dist and sharp_dist > medium_dist:
        braking = 0
        if speed > sharp_turn_speed:
            braking = 0.3
        if distances[-45] > distances[45]:
            return np.array([-1, 0, braking])
        return np.array([1, 0, braking])

    # medium turn
    if medium_dist > narrow_dist:
        if distances[-15] > distances[15]:
            return np.array([-0.15, 1, 0])  # .5, 905.89; .6, 905.69; .7, 906.49; .8, 906.39; .9, 905.19; 1, 906.79
        return np.array([0.15, 1, 0])

    # go mostly straight with accel
    acceleration = 1
    if distances[-5] - distances[0] > narrow_distance_buffer:
        return np.array([-0.1, acceleration, 0])
    if distances[5] - distances[0] > narrow_distance_buffer:
        return np.array([0.1, acceleration, 0])
    return np.array([0, acceleration, 0])


def get_speed(car):
    return np.sqrt(
        np.square(car.hull.linearVelocity[0])
        + np.square(car.hull.linearVelocity[1])
    )


if __name__ == "__main__":
    run_rl_bot()
