import torch
import gym
import numpy as np


def run(model, render=True):
    env = gym.make("CartPole-v0")
    env.reset()
    action = env.action_space.sample()
    for _ in range(1000):
        if render:
            env.render()
        observation, reward, done, info = env.step(action)
        model_out = (
            model(torch.from_numpy(observation).type(torch.float32)).detach().numpy()
        )
        action = np.argmax(model_out)

    env.close()


def main():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 10), torch.nn.ReLU(), torch.nn.Linear(10, 2)
    )
    run(model)


if __name__ == "__main__":
    main()
