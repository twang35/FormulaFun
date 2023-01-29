import numpy as np

from reflex.car_racing import CarRacing


def run_reflex_bot():
    game = CarRacing(render_mode="human")
    game.reset(seed=123)
    # game.reset()

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


# distances angles are +-105, 90, 45, 15, 5, 0
def get_next_action(distances, car):
    braking_point_corner_distance = 40
    max_turning_speed = 60
    narrow_distance_buffer = 10
    medium_distance_buffer = 5
    sharp_distance_buffer = 5

    # if on grass: coast
    if len(distances) == 0:
        return np.array([0, 0, 0])

    speed = get_speed(car)
    print(speed)
    # brake for upcoming corner
    if speed > max_turning_speed and distances[0] <= braking_point_corner_distance:
        return np.array([0, 0, 0.7])

    narrow_dist = max(distances[-5], distances[0], distances[5])
    medium_dist = max(distances[-15], distances[15]) - medium_distance_buffer
    sharp_dist = max(distances[-45], distances[45]) - sharp_distance_buffer

    # sharp turn
    if sharp_dist > narrow_dist and sharp_dist > medium_dist:
        if distances[-45] > distances[45]:
            return np.array([-1, 0, 0])
        return np.array([1, 0, 0])

    # medium turn
    if medium_dist > narrow_dist:
        if distances[-15] > distances[15]:
            return np.array([-0.1, 0.3, 0])
        return np.array([0.1, 0.3, 0])

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
    run_reflex_bot()
