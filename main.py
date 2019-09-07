import torch
import gym
import numpy as np


def run(model, render=True):
    env = gym.make("CartPole-v0")
    env.reset()
    action = env.action_space.sample()

    states, actions, rewards = [], [], []
    observation, _, done, info = env.step(action)

    for _ in range(1000):
        if done:
            break

        if render:
            env.render()

        states.append(observation)

        model_out = model(torch.from_numpy(observation).type(torch.float32))
        prob = model_out.detach().numpy().astype(float)
        prob /= prob.sum()
        action = np.random.choice(2, p=prob)

        observation, reward, done, info = env.step(action)

        actions.append(action)
        rewards.append(reward)

    env.close()

    return states, actions, rewards


def main():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 2),
        torch.nn.Sigmoid(),
    )
    run(model)


if __name__ == "__main__":
    main()
