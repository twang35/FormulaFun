import numpy as np

from car_racing import CarRacing

game = CarRacing(render_mode="human")
game.reset()

for i in range(300):
    action = np.array([0, +1.0, 0])
    game.step(action)
    game.render()
