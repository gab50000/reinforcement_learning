from collections import namedtuple

import fire
import torch
import numpy as np
import gym


TrainingResult = namedtuple(
    "TrainingResult", "observations, actions, probabilities, rewards, length"
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

    rewards = discounted_rewards(rewards).tolist()

    return TrainingResult(observations, actions, action_probs, rewards, len(rewards))


def run_one_batch(env, policy, *, batch_size=5000, render=False):
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

    while True:
        training_result = run_one_episode(env, policy)
        obs, actions, action_probs, rew, len_ = training_result

        batch_observations += obs
        batch_actions += actions
        batch_action_probs += action_probs
        batch_rewards += rew
        episode_lengths.append(len_)

        obs_len = len(batch_observations)
        print(f"{obs_len} samples", end="\r")
        if obs_len >= batch_size:
            break

    batch_action_probs = torch.stack(batch_action_probs)
    batch_action_probs = batch_action_probs[range(sum(episode_lengths)), batch_actions]
    batch_rewards = torch.from_numpy(np.array(batch_rewards))

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
        training_result = run_one_batch(
            env, policy, batch_size=batch_size, render=False
        )
        obs, actions, probs, rewards, ep_lengths = training_result
        optim.zero_grad()
        loss = calc_loss(probs, rewards)
        print(f"\nLoss: {loss.item():.2f}")
        print("Average episode length:", np.mean(ep_lengths))
        print("Median episode length:", np.median(ep_lengths))
        print("Longest episode:", max(ep_lengths))
        print("Shortest episode:", min(ep_lengths))
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
