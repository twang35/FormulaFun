# Formula Fun
"Formula 1, more like Formula Fun!" - Amanda

The race track and game engine come from [OpenAI CarRacing-v0](https://www.gymlibrary.dev/environments/box2d/car_racing/).

### Human vs Bots
Is a human coded racer better or is an AI controlled racer better?

## Human
![human](https://user-images.githubusercontent.com/5852883/219848284-34f80967-8277-48fd-a587-a70dfd9d7fa3.gif)

#### Max reward: 886.9
- Basically trying to stay on the track and take the corners without spinning out.

## Reflex Bot

![reflex](https://user-images.githubusercontent.com/5852883/219848982-5f91c4a8-542e-4a1c-9310-3575efef815e.gif)

#### Max reward: 908.3
- Manually coded reflex bot
- Tries to take corners with highest possible speed
- Mostly stays centered on the track

## PPO (Proximal Policy Optimization) Model

![ppo3](https://user-images.githubusercontent.com/5852883/219849531-7434e499-6ce8-4466-8b60-958dfac32f90.gif)

#### Max reward: 831
- Worse than manual human control
- Learning often got stuck in local optimums

## TD3 (Twin Delayed Deep Deterministic Policy Gradients) Model

![td3](https://user-images.githubusercontent.com/5852883/219849780-73fcbdaf-89e2-4372-8bf4-507c26b9c118.gif)

#### Max reward: 921.0
- Fairly quick learning
- Able to hit corner apex during turns
- Carries speed through shallow turns
