from collections import namedtuple

import fire
import torch
import numpy as np
import gym


TrainingResult = namedtuple(
    "TrainingResult", "observations, actions, probabilities, rewards, lengths"
)


class LogPolicy(torch.nn.Module):
    def __init__(self, n_obs, n_hidden, n_actions):
        super().__init__()

        self.policy = torch.nn.Sequential(
            torch.nn.Linear(n_obs, n_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(n_hidden, n_actions),
        )

    def forward(self, x):
        return torch.log_softmax(self.policy(x), dim=-1)


def discounted_rewards(rewards, gamma=0.9):
    dr = np.zeros(len(rewards))
    dr[-1] = rewards[-1]
    for i in reversed(range(len(rewards) - 1)):
        dr[i] = rewards[i] + gamma * dr[i + 1]
    return dr


def calc_loss(action_probs, discounted_rewards):
    log_probs = torch.log(action_probs)
    return -torch.mean(discounted_rewards * action_probs)


def run_one_episode(env, policy, *, render=False):
    obs = env.reset()

    observations, actions, action_probs, rewards = [], [], [], []

    while True:
        if render:
            env.render()

        action_prob = policy(torch.from_numpy(obs).type(torch.float32))
        action = torch.multinomial(torch.exp(action_prob), 1).detach().item()

        obs, rew, done, _ = env.step(action)

        observations.append(obs)
        actions.append(action)
        action_probs.append(action_prob)
        rewards.append(rew)

        if done:
            break

    rewards = torch.from_numpy(discounted_rewards(rewards))

    return observations, actions, action_probs, rewards


def train_one_batch(env, policy, *, batch_size=5000, render=False):
    """
    Runs and restarts the environment until the desired batch size has been reached.
    """
    # put all observations, actions, action probabilities and rewards
    # into these lists
    batch_observations = []
    batch_actions = []
    batch_action_probs = []
    batch_rewards = []
    episode_lengths = []

    # undiscounted rewards
    rewards = []

    obs = env.reset()

    while True:
        if render:
            env.render()

        action_prob = policy(torch.from_numpy(obs).type(torch.float32))
        action = torch.multinomial(torch.exp(action_prob), 1).detach().item()

        obs, rew, done, _ = env.step(action)

        batch_observations.append(obs)
        batch_actions.append(action)
        batch_action_probs.append(action_prob)
        rewards.append(rew)

        if done:
            print("Number of observations:", len(batch_observations), end="\r")
            episode_len = len(rewards)
            batch_rewards.append(torch.from_numpy(discounted_rewards(rewards)))
            episode_lengths.append(episode_len)

            if len(batch_observations) >= batch_size:
                break

            obs = env.reset()
            rewards = []

    batch_action_probs = torch.stack(batch_action_probs)
    batch_action_probs = batch_action_probs[range(sum(episode_lengths)), batch_actions]
    batch_rewards = torch.cat(batch_rewards)

    return TrainingResult(
        batch_observations,
        batch_actions,
        batch_action_probs,
        batch_rewards,
        episode_lengths,
    )


def train(env, policy, *, n_epochs=50, batch_size=5000, render=False):
    optim = torch.optim.Adam(policy.parameters(), lr=0.01)

    for ep in range(n_epochs):
        training_result = train_one_batch(
            env, policy, batch_size=batch_size, render=False
        )
        obs, actions, probs, rewards, ep_lengths = training_result
        optim.zero_grad()
        loss = calc_loss(probs, rewards)
        print(f"\nLoss: {loss.item():.2f}")
        print("Average episode length:", sum(ep_lengths) / len(ep_lengths))
        loss.backward()
        optim.step()

        if render:
            run_one_episode(env, policy, render=render)


def main(env_name, *, n_hidden=30, n_epochs=50, batch_size=5000, render=False):
    env = gym.make(env_name)
    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    pol = LogPolicy(n_obs, n_hidden, n_actions)

    train(env, pol, n_epochs=n_epochs, batch_size=batch_size, render=render)

    env.close()


if __name__ == "__main__":
    fire.Fire(main)
