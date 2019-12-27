import fire
import torch
import numpy as np
import gym


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


def discounted_rewards(rewards, gamma):
    dr = np.zeros(len(rewards))
    dr[-1] = rewards[-1]
    for i in reversed(range(len(rewards) - 1)):
        dr[i] = rewards[i] + gamma * dr[i + 1]
    return dr


def train_one_epoch(env, policy, *, n_epochs=50, batch_size=5000, render=False):
    batch_observations = []
    actions = []
    batch_rewards = []
    episode_lengths = []

    obs = env.reset()

    for ep in range(n_epochs):
        if render:
            env.render()

        action_probs = policy(torch.from_numpy(obs).type(torch.float32))
        action = torch.multinomial(torch.exp(action_probs), 1).detach().item()

        obs, rew, done, _ = env.step(action)

        batch_observations.append(obs)
        actions.append(action)
        batch_rewards.append(rew)

        if done:
            break


def main(env_name, n_hidden=30):
    env = gym.make(env_name)
    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    pol = LogPolicy(n_obs, n_hidden, n_actions)

    train_one_epoch(env, pol, render=True)


if __name__ == "__main__":
    fire.Fire(main)
