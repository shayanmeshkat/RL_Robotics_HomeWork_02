import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import fnmatch

# df = pd.read_parquet('reward_list.parquet')
df_base = pd.read_parquet('./Results/base_training_data.parquet')
# df_base = pd.read_parquet('base_training_data.parquet')


df = pd.read_parquet('./Results/base_testing_data.parquet')
# df_base = pd.read_parquet('QAS_Sink_State_15k.parquet')


window_size = 20
iters_num = df.columns.__len__()
iters_num = 5


agent_iter_list = []
agent_base_iter_list = []


for iter_ind in range(iters_num):
    agent_iter_mid = df[f"Iter_{iter_ind+1}"]
    agent_base_iter_mid = df_base[f"Iter_{iter_ind+1}"]

    agent_iter_list.append(agent_iter_mid)
    agent_base_iter_list.append(agent_base_iter_mid)

episode_mean_val = [[] for _ in range(iters_num)]
episode_base_mean_val = [[] for _ in range(iters_num)]

for iter_ind in range(iters_num):
    for i in range(len(agent_iter_list[0])):
        mean_temp = np.mean(agent_iter_list[iter_ind][i:i+window_size])
        mean_temp_base = np.mean(agent_base_iter_list[iter_ind][i:i+window_size])
        episode_mean_val[iter_ind].append(mean_temp)
        episode_base_mean_val[iter_ind].append(mean_temp_base)
episode_mean_val = np.array(episode_mean_val)
episode_base_mean_val = np.array(episode_base_mean_val)

agents_mean_val = np.mean(episode_mean_val, axis=0)

agents_mean_val_base = np.mean(episode_base_mean_val, axis=0)

percentile_50 = np.percentile(episode_mean_val, 50, axis=0)
percentile_25 = np.percentile(episode_mean_val, 25, axis=0)
percentile_75 = np.percentile(episode_mean_val, 75, axis=0)

percentile_25_base = np.percentile(episode_base_mean_val, 25, axis=0)
percentile_50_base = np.percentile(episode_base_mean_val, 50, axis=0)
percentile_75_base = np.percentile(episode_base_mean_val, 75, axis=0)


l_width = 2
E = np.arange(episode_mean_val.shape[1])
E_base = np.arange(episode_base_mean_val.shape[1])
plt.figure(figsize=(10, 5))
plt.plot(E, percentile_50, linewidth= l_width, color='blue', label='Testing Reward')

plt.plot(E_base, percentile_50_base, linewidth= l_width, color='red', label='Training Reward')

plt.fill_between(E, percentile_25, percentile_75, alpha=0.5)
plt.fill_between(E_base, percentile_25_base, percentile_75_base, alpha=0.5)
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
plt.minorticks_on()
plt.xlabel('Episode')
plt.ylabel('Discounted Cumulative Reward')
# plt.ylim(-2, 1)
# plt.xlim(0, 4000)
plt.title('Training and Testing Reward per Episode')
plt.legend()
plt.savefig('agent_reward.pdf', format="pdf",
            bbox_inches='tight', dpi=300)
plt.show()