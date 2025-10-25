import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def save_reward(reward_list, file_name):

    reward_list = np.array(reward_list)

    # Save the reward list as a parquet file
    m, n = reward_list.shape
    column_names = [f'Iter_{i+1}' for i in range(m)]
    data = {f'Iter_{i+1}': reward_list[i] for i in range(m)}
    df = pd.DataFrame(data, columns=column_names)
    df.to_parquet(file_name + '.parquet')
    return df


# reward_list = np.array(reward_list)
# test_reward_list = np.array(test_reward_list)

# # Save the reward list as a parquet file
# m, n = test_reward_list.shape
# column_names = [f'Iter_{i+1}' for i in range(m)]
# data = {f'Iter_{i+1}': test_reward_list[i] for i in range(m)}
# df = pd.DataFrame(data, columns=column_names)
# df.to_parquet('reward_base_list.parquet')