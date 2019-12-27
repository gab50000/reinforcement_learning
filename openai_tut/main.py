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


def discounted_rewards(rewards, gamma=0.9):
    dr = np.zeros(len(rewards))
    dr[-1] = rewards[-1]
    for i in reversed(range(len(rewards) - 1)):
        dr[i] = rewards[i] + gamma * dr[i + 1]
    return dr


def train_one_epoch(env, policy, *, n_epochs=50, batch_size=5000, render=False):
    batch_observations = []
    batch_actions = []
    batch_action_probs = []
    batch_rewards = []
    episode_lengths = []

    for ep in range(n_epochs):

        observations, actions, action_probs, rewards = [], [], [], []

        obs = env.reset()

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
                episode_len = len(rewards)
                batch_observations += observations
                batch_actions += actions
                action_probs = torch.stack(action_probs)
                batch_action_probs.append(action_probs[range(episode_len), actions])
                batch_rewards.append(discounted_rewards(rewards))
                break
        breakpoint()


def main(env_name, n_hidden=30):
    env = gym.make(env_name)
    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    pol = LogPolicy(n_obs, n_hidden, n_actions)

    train_one_epoch(env, pol, render=True)


if __name__ == "__main__":
    fire.Fire(main)
