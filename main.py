import fire
import gym
import numpy as np
import torch


def run(model, render=True):
    env = gym.make("CartPole-v0")
    env.reset()
    action = env.action_space.sample()

    states, action_probs, actions, rewards = [], [], [], []
    observation, _, done, info = env.step(action)

    for n in range(1000):
        if done:
            rewards[-1] = -10
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


def run_multiple(model, n):
    states, action_probs, actions, rewards = [], [], [], []
    total_steps = 0
    for _ in range(n):
        s, ap, a, rew = run(model, render=False)
        total_steps += len(s)
        states.append(s)
        action_probs.append(ap)
        actions.append(a)
        rewards.append(rew)
    print(f"On average {total_steps/n} steps")
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


def evaluate(
    states, action_probs, actions, discounted_rewards, model, learning_rate=0.1
):
    # check if the action in response to a given state led to a positive or
    # a negative reward
    # if yes, enforce this action
    prob_selection = torch.stack(action_probs)[range(len(action_probs)), actions]
    prob_reward = (
        prob_selection * torch.from_numpy(discounted_rewards).type(torch.float32)
    ).sum()

    prob_reward.backward()
    with torch.no_grad():
        for param in model.parameters():
            param += learning_rate * param.grad
            param.grad.zero_()


def flatten(list_of_lists):
    return [x for list_ in list_of_lists for x in list_]


def main(n_epochs=100, n_average=10, learning_rate=0.1, show=False):
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 2),
        torch.nn.Softmax(dim=0),
    )
    for _ in range(n_epochs):
        states, action_probs, actions, rewards = run_multiple(model, n_average)
        states = flatten(states)
        action_probs = flatten(action_probs)
        actions = flatten(actions)
        discounted_rewards = np.concatenate([normalize_rewards(rew) for rew in rewards])
        discounted_rewards = (
            discounted_rewards - discounted_rewards.mean()
        ) / discounted_rewards.std()
        evaluate(
            states,
            action_probs,
            actions,
            discounted_rewards,
            model,
            learning_rate=learning_rate,
        )

        if show:
            run(model, render=True)


if __name__ == "__main__":
    fire.Fire(main)
