import gymnasium
import time

env = gymnasium.make("Pendulum-v1", render_mode="human")
env.reset()
while True:
    env.step([0.0])
    time.sleep(0.016)

