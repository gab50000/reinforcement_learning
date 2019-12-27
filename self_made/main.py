import fire
import gym
import numpy as np
import torch


def run(
    model, max_steps=1000, render=True, sample=True, temperature=1, game="CartPole-v0"
):
    env = gym.make(game)
    env.reset()
    action = env.action_space.sample()

    states, action_probs, actions, rewards = [], [], [], []
    observation, _, done, info = env.step(action)

    for _ in range(max_steps):
        if done:
            if rewards[-1] >= 0:
                rewards[-1] = -10
            break

        if render:
            env.render()

        states.append(observation)

        model_out = model(torch.from_numpy(observation).type(torch.float32))
        prob = model_out.detach().numpy().astype(float)
        prob = np.exp(prob / temperature)
        prob /= prob.sum()
        if sample:
            action = np.random.choice(prob.size, p=prob)
        else:
            action = np.argmax(prob)

        observation, reward, done, info = env.step(action)

        action_probs.append(model_out)
        actions.append(action)
        rewards.append(reward)
    env.close()

    return states, action_probs, actions, rewards


def run_multiple(model, n, game, temperature=1):
    states, action_probs, actions, rewards = [], [], [], []
    total_steps = 0
    for _ in range(n):
        s, ap, a, rew = run(model, render=False, temperature=temperature, game=game)
        total_steps += len(s)
        states.append(s)
        action_probs.append(ap)
        actions.append(a)
        rewards.append(rew)
    print(f"On average {total_steps/n} steps")
    print(f"On average {np.average([sum(r) for r in rewards]):.02f} rewards")
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


def train(
    n_epochs=100,
    n_average=10,
    learning_rate=0.1,
    show=False,
    model_filename="",
    temperature=1,
    game="CartPole-v0",
    action_size=2,
    observation_size=4,
    discount_factor=0.9,
):
    model = torch.nn.Sequential(
        torch.nn.Linear(observation_size, 30),
        torch.nn.ReLU(),
        torch.nn.Linear(30, 30),
        torch.nn.ReLU(),
        torch.nn.Linear(30, 30),
        torch.nn.ReLU(),
        torch.nn.Linear(30, action_size),
        torch.nn.Softmax(dim=0),
    )
    for _ in range(n_epochs):
        states, action_probs, actions, rewards = run_multiple(
            model, n_average, temperature=temperature, game=game
        )
        states = flatten(states)
        action_probs = flatten(action_probs)
        actions = flatten(actions)
        discounted_rewards = np.concatenate(
            [normalize_rewards(rew, discount_factor=discount_factor) for rew in rewards]
        )
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
            run(model, render=True, sample=False, game=game)

        if model_filename:
            torch.save(model, model_filename)


def run_model(filename, game):
    model = torch.load(filename)
    run(model, max_steps=1_000_000, sample=False, game=game)


if __name__ == "__main__":
    fire.Fire()
