import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
import environment as env_module
import save_results
import pandas as pd
import os
import math


def encode_position(x, y, X=4, Y=3):
    """
    Convert (x, y) agent position in a grid (X by Y)
    into a 1D one-hot vector of length X*Y.
    """
    # Create zero array
    one_hot = np.zeros(X * Y, dtype=np.float32)

    # Compute 1D index
    index = y * X + x   # row-major order

    # Set agent location to 1
    one_hot[index] = 1

    return one_hot



# Neural Network Architecture
class DQN(nn.Module):
    def __init__(self, state_size, hidden_layer, action_size):
        super().__init__()

        # Define the layers
        self.fc1 = nn.Linear(state_size, hidden_layer)
        self.out = nn.Linear(hidden_layer, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x
    



# Memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        sample_batch = random.sample(self.memory, sample_size)
        return sample_batch
    
    def __len__(self):
        return len(self.memory)
    
def get_epsilon(episode, eps_start=1.0, eps_end=0.0001, decay=0.005):
    return eps_end + (eps_start - eps_end) * math.exp(-decay * episode)

# Training loop
class Grid_World_v0():

    learning_rate = 0.001
    gamma = 0.9
    network_sync_freq = 10
    replay_memory_size = 2000
    mini_batch = 64
    hidden_layer_size = None  # Will default to num_states if None
    epsilon_end = 0.0001  # Minimum epsilon value (exploration floor)
    epsilon_decay = 0.005  # Decay rate for epsilon-greedy exploration

    loss_fun = nn.MSELoss()
    optimizer = None

    action_set = ['UP', 'DOWN', 'RIGHT', 'LEFT']



    def train(self, episodes=100, steps_per_episode=200, render=False, is_slippery=True):

        

        env_grid = env_module.env()

        # num_states = env.observation_space.n
        # num_actions = env.action_space.n

        num_states = env_grid.grid_x * env_grid.grid_y
        num_actions = env_grid.actions_num

        # Use hidden_layer_size if set, otherwise default to num_states
        hidden_size = self.hidden_layer_size if self.hidden_layer_size is not None else num_states

        epsilon = 0.1
        memory = ReplayMemory(self.replay_memory_size)

        policy_dqn = DQN(state_size=num_states, hidden_layer=hidden_size, action_size=num_actions)

        target_dqn = DQN(state_size=num_states, hidden_layer=hidden_size, action_size=num_actions)

        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('before training policy')

        self.print_dqn(policy_dqn)

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

        rewards_per_episode = np.zeros(episodes)

        epsilon_history = []
        test_reward_per_run = []

        step_count = 0
        # env_grid = env_module.env()
        # state_x, state_y = env_grid.agent
        # state = encode_position(state_x, state_y, X=env_grid.grid_x, Y=env_grid.grid_y)




        for i in range(episodes):

            # state = env.reset()[0]
            
            env_grid = env_module.env()
            state_x, state_y = env_grid.agent
            state = encode_position(state_x, state_y, X=env_grid.grid_x, Y=env_grid.grid_y)

            terminal_state = False
            train_rewards = 0
            for j in range(steps_per_episode):
                # print('state in train loop:', state)

                if random.random() < epsilon:
                    # action = env.action_space.sample()
                    action = np.random.randint(num_actions)
                else:
                    with torch.no_grad():
                        # state_tensor = torch.FloatTensor(self.state_to_onehot(state)).unsqueeze(0)
                        # q_values = policy_dqn(state_tensor)
                        # action = torch.argmax(q_values).item()

                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()
                
                # Use the class attributes for epsilon calculation
                epsilon = get_epsilon(i, eps_end=self.epsilon_end, decay=self.epsilon_decay)

                # next_state, reward, terminal_state, step_exhausted, inf0 = env.step(action)
                state_x, state_y = env_grid.agent

                new_state_x, new_state_y = env_grid.execute_action(state_x, state_y, action)

                next_state = encode_position(new_state_x, new_state_y, X=env_grid.grid_x, Y=env_grid.grid_y)

                reward, terminal_state = env_grid.get_reward()

                train_rewards += reward
                # print(f"Episode: {i+1}, Step: {j+1}, Action: {self.action_set[action]}, Reward: {reward}, Epsilon: {epsilon:.3f}")


                memory.append((state, action, next_state, reward, terminal_state))  

                # Move to the next state
                state = next_state

                step_count += 1


                

                if terminal_state:
                    break


            rewards_per_episode[i] = train_rewards

            # print('rewards_per_episode in train after episode is:', rewards_per_episode)

            # if len(memory) > self.mini_batch and np.sum(rewards_per_episode) > 0:

            if len(memory) > self.mini_batch and any(r > 0 for r in rewards_per_episode):

                    
                # print('sampling from memory for optimization')
                mini_batch = memory.sample(self.mini_batch)
                self.optimize(mini_batch
                            , policy_dqn
                            , target_dqn
                            )

                # epsilon = max(epsilon - 1/episodes, 0)
                # epsilon_history.append(epsilon)

                if step_count > self.network_sync_freq:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

            torch.save(policy_dqn.state_dict(), "grid_world_v0_dqn.pt")
            

            if (i+1)%10==0:
            
                test_reward_per_episode = self.test(episodes=10, steps_per_episode=steps_per_episode, is_slippery=is_slippery)

                # test_reward_per_episode = test_reward_per_episode[-1]

                test_reward_per_run.append(test_reward_per_episode)
                
                
        test_reward_per_run = [arr.tolist() for arr in test_reward_per_run]
        print('test_reward_per_run:', test_reward_per_run)
        test_reward_per_run = [x for sublist in test_reward_per_run for x in sublist]
        print('test_reward_per_run after flatten:', test_reward_per_run)

        


        # env.close()

        # Save policy
        # torch.save(policy_dqn.state_dict(), "grid_world_v0_dql.pt")

        return rewards_per_episode, test_reward_per_run

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, next_state, reward, terminal_state in mini_batch:

            # print(f'state: {state}, action: {action}, next_state: {next_state}, reward: {reward}, terminal_state: {terminal_state}')

            if terminal_state:
                target = torch.FloatTensor([reward])

            else:
                with torch.no_grad():
                    # print('state in optimize loop:', state)
                    # print('next_state in optimize loop:', next_state)
                    # print('dqn input:', self.state_to_dqn_input(state, num_states))
                    # exit()
                    target = torch.FloatTensor(reward+ self.gamma * target_dqn(self.state_to_dqn_input(next_state, num_states)).max())

            # print('after target calculation')
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            target_q = target_dqn(self.state_to_dqn_input(state, num_states))
            target_q[action] = target
            target_q_list.append(target_q)

        loss = self.loss_fun(torch.stack(current_q_list), torch.stack(target_q_list))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def state_to_dqn_input(self, state, num_states:int) -> torch.Tensor:
        # print('num_states in state_to_dqn_input:', num_states)
        # print('state in state_to_dqn_input:', state)
        input_tensor = torch.zeros(num_states)
        # print('input_tensor before:', input_tensor)
        
        # If state is already a one-hot encoded array, find the index
        # if isinstance(state, np.ndarray):
        state_index = np.argmax(state)
        # else:
        #     state_index = state
            
        input_tensor[state_index] = 1

        # print('input_tensor after:', input_tensor)
        # exit()
        return input_tensor
    

    def test(self, episodes=100, steps_per_episode=200, is_slippery=False):
        
        # Load learned policy first (before loop)
        env_grid = env_module.env()
        num_states = env_grid.grid_x * env_grid.grid_y
        num_actions = env_grid.actions_num

        # Use the same hidden_layer_size as in training
        hidden_size = self.hidden_layer_size if self.hidden_layer_size is not None else num_states

        policy_dqn = DQN(state_size=num_states, hidden_layer=hidden_size, action_size=num_actions) 
        policy_dqn.load_state_dict(torch.load("grid_world_v0_dqn.pt"))
        policy_dqn.eval()    # switch model to evaluation mode

        print('Policy (trained):')
        self.print_dqn(policy_dqn)


        # Track test rewards
        test_rewards = np.zeros(episodes)

        for i in range(episodes):
            # Reinitialize environment for each episode
            env_grid = env_module.env()
            state_x, state_y = env_grid.agent
            state = encode_position(state_x, state_y, X=env_grid.grid_x, Y=env_grid.grid_y)
            
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions            

            step_iteration = 0
            test_reward = 0
            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            for j in range(steps_per_episode):
                # Select best action   
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                next_state_x, next_state_y = env_grid.execute_action(state_x, state_y, action)
                
                next_state = encode_position(next_state_x, next_state_y, X=env_grid.grid_x, Y=env_grid.grid_y)
                
                reward, terminated = env_grid.get_reward()

                test_reward += reward



                # Update state and agent position for next iteration
                state = next_state
                state_x, state_y = next_state_x, next_state_y


                if terminated:
                    break

            # Record if episode was successful
            test_rewards[i] = test_reward
            
            

        # Return test rewards
        return test_rewards

    # Print DQN: state, best action, q values
    def print_dqn(self, dqn):
        # Get number of input nodes
        num_states = dqn.fc1.in_features

        # Loop each state and print policy to console
        for s in range(num_states):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action to L D R U
            best_action = self.action_set[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            print(f'{s:02},{best_action},[{q_values}]', end=' ')         
            if (s+1)%4==0:
                print() # Print a newline every 4 states

def plotting(rewards, test_rewards):
    
        # Save the data as a parquet file
    

    # Plot the training rewards
    rolling_mean_window = 10
    rolling_mean = pd.Series(rewards).rolling(window=rolling_mean_window).mean()
    # rolling_mean_test = pd.Series(test_rewards).rolling(window=rolling_mean_window).mean()
    rew = np.array(rolling_mean.fillna(0))
    # rew_test = np.array(rolling_mean_test.fillna(0))


    plt.figure(figsize=(10, 6))
    plt.plot(range(len(rolling_mean)), rew,
            label=f'Rolling Mean (window={rolling_mean_window})',
            color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Reward per Episode')
    plt.legend()
    plt.grid('minor')

if __name__ == '__main__':

    os.makedirs('./Results', exist_ok=True)

    training_data_file_name = './Results/base_training_data'
    testing_data_file_name = './Results/base_testing_data'

    file_path = "grid_world_v0_dqn.pt"

    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} deleted successfully.")
    else:
        print(f"{file_path} does not exist.")

    # Number of iterations to run
    steps = 80
    num_episodes = 2000
    num_iterations = 1  # Change this to run more or fewer iterations

    reward_list = [[] for _ in range(num_iterations)]
    test_reward_list = [[] for _ in range(num_iterations)]

    is_slippery = False
    
    # Initialize lists to store results from all iterations
    all_training_rewards = []
    all_testing_rewards = []
    
    for iteration in range(num_iterations):
        torch.manual_seed(iteration)
        np.random.seed(iteration)
        random.seed(iteration)
        
        print(f"\n{'='*50}")
        print(f"Starting Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*50}\n")
        
        grid_world_v0 = Grid_World_v0()
        
        # Train the model
        reward_per_episode, test_reward_per_episode = grid_world_v0.train(episodes=num_episodes, steps_per_episode=steps, is_slippery=is_slippery)
        print('reward_per_episode:', reward_per_episode)

        reward_list[iteration] = reward_per_episode
        print('reward_list:', reward_list)

        test_reward_list[iteration] = test_reward_per_episode

        # Test the model
        # test_reward_per_episode = grid_world_v0.test(episodes=num_episodes, steps_per_episode=steps, is_slippery=is_slippery)

        # test_reward_list[iteration] = test_reward_per_episode
        # print('test_reward_list:', test_reward_list)



    reward_list = [arr.tolist() for arr in reward_list]
    # test_reward_list = [arr.tolist() for arr in test_reward_list]
    print('Final reward_list:', reward_list)
    print('Final test_reward_list:', test_reward_list)

    save_results.save_reward(reward_list, training_data_file_name)
    save_results.save_reward(test_reward_list, testing_data_file_name)

    # Plot the results (using the last iteration)
    plotting(reward_list[-1], test_reward_list[-1])
    plt.savefig('./Results/training_testing_rewards.png')
    plt.show()
