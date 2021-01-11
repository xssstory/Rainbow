import torch
import numpy as np
import matplotlib.pyplot as plt
import sys


def smooth(arr, window=50):
    arr = np.asarray(arr)
    out = np.zeros_like(arr)
    for i in range(len(out)):
        out[i] = np.mean(arr[max(i - window, 0): i + 1])
    return out


if __name__ == "__main__":
    metric_path = sys.argv[1]
    metrics = torch.load(metric_path)
    episode_length = metrics['episode_length'][1:]
    episode_reward = metrics['episode_reward'][1:]
    smoothed_length = smooth(episode_length)
    smoothed_reward = smooth(episode_reward)
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(np.cumsum(episode_length), smoothed_length)
    ax[0].set_xlabel("Total timesteps")
    ax[0].set_ylabel("Mean episode length")
    ax[0].grid()
    ax[1].plot(np.cumsum(episode_length), smoothed_reward)
    ax[1].set_xlabel("Total timesteps")
    ax[1].set_ylabel("Mean episode reward")
    ax[1].grid()
    plt.show()
