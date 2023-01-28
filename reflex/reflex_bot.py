import numpy as np

from reflex.car_racing import CarRacing

game = CarRacing(render_mode="human")
game.reset(seed=123)

action = np.array([0, +1.0, 0])
game.step(action)

for i in range(300):
    action = np.array([0, +1.0, 0])
    game.step(action)
    game.render()
