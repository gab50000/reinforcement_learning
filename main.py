import torch
import gym
import numpy as np


def run(model, render=True):
    env = gym.make("CartPole-v0")
    env.reset()
    action = env.action_space.sample()

    states, action_probs, actions, rewards = [], [], [], []
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

        action_probs.append(model_out)
        actions.append(action)
        rewards.append(reward)

    env.close()

    return states, action_probs, actions, rewards


def normalize_rewards(rewards, discount_factor=0.9):
    discounted = np.zeros(len(rewards))
    discounted[-1] = rewards[-1]

    for i in reversed(range(discounted.shape[0] - 1)):
        discounted[i] = rewards[i] + discount_factor * discounted[i + 1]
    return discounted


def test_normalize_rewards():
    rewards = [1, 1, 1, 10, 1, -10]
    target = [4.7512, 4.168, 3.52, 2.8, -8, -10]
    result = normalize_rewards(rewards)
    np.testing.assert_allclose(target, result)


def evaluate(states, action_probs, actions, rewards, model):
    for s, ap, ac, r in zip(states, action_probs, actions, rewards):
        discounted_rewards = normalize_rewards(rewards)
        # check if the action in response to a given state led to a positive or
        # a negative reward
        # if yes, enforce this action
        prob_selection = torch.stack(action_probs)[range(len(action_probs)), actions]
        prob_selection.backward()

        with torch.no_grad():
            for param in model.parameters():
                param += ...


def main():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 2),
        torch.nn.Sigmoid(),
    )
    states, action_probs, actions, rewards = run(model)
    evaluate(states, action_probs, actions, rewards, model)


if __name__ == "__main__":
    main()
