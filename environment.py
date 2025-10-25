import random, math, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.widgets import Button

from enum import Enum
import numpy as np


class ActionsNew(Enum):
    # stop = 0
    up    = 0
    down  = 1
    right = 2
    left  = 3
    # stop  = 4

class env():
    def __init__(self):
        self.grid_x = 4
        self.grid_y = 3
        self.agent = (0,0)
        self.actions_num = 4
        self.map()  # Initialize the environment map and attributes
    
    def execute_action(self, state_x, state_y, action):
        
        x, y = state_x, state_y
        
        # Implement slip: 80% intended, 10% perpendicular left, 10% perpendicular right
        rand = np.random.rand()
        
        # Define perpendicular actions for each direction
        # Format: action -> (left_perpendicular, right_perpendicular)
        perpendicular_map = {
            ActionsNew.up.value: (ActionsNew.left.value, ActionsNew.right.value),
            ActionsNew.down.value: (ActionsNew.right.value, ActionsNew.left.value),
            ActionsNew.left.value: (ActionsNew.down.value, ActionsNew.up.value),
            ActionsNew.right.value: (ActionsNew.up.value, ActionsNew.down.value)
        }
        
        if rand < 0.8:
            # Execute intended action (80%)
            actual_action = action
        elif rand < 0.9:
            # Slip perpendicular to the left (10%)
            actual_action = perpendicular_map[action][0]
        else:
            # Slip perpendicular to the right (10%)
            actual_action = perpendicular_map[action][1]
        # actual_action = action

        action_ = ActionsNew(actual_action)
        # print('action in execute action is:', action_)
        if (x,y,action_) not in self.forbidden_transitions:

            if action_ == ActionsNew.down:  # down
                y = y-1
            elif action_ == ActionsNew.left:  # left
                x = x-1
            elif action_ == ActionsNew.right:  # right
                x = x+1
            elif action_ == ActionsNew.up:  # up
                y = y+1
            # if action_ == ActionsNew.stop:  # stop
            #     x = x
            #     y = y
        self.agent = (x,y)
        # print('agent in execute action after update is:', self.agent)
        # exit()
        return x, y
    
    def get_reward(self):
        """
        Returns the string with the propositions that are True in this state
        """
        reward = -0.04  # Default reward for each step taken
        # if self.agent in self.objects[0]:
        #     reward = 1
        # elif self.agent in self.objects[1]:
        #     reward = -1
        terminal = False
        if self.agent in self.goal_locations:
            reward = 1
            terminal = True
        elif self.agent in self.penalty_locations:
            reward = -1
            terminal = True

        return reward, terminal

    def map(self):
        self.objects = {}
        self.objects[(4,3)] = "a"
        self.objects[(4,2)] = "b"

        self.goal_locations = {(3,2)}  # Goal location
        self.penalty_locations = {(3,1)}  # Penalty location
        
        # Adding the agent
        # self.agent = (2,0)
        self.actions = [ActionsNew.up.value,
                        ActionsNew.down.value,ActionsNew.right.value,
                        ActionsNew.left.value]

        # Adding walls
        self.forbidden_transitions = set()
        for x in [0]:
            for y in range(self.grid_y): # No right turn to buildings
                self.forbidden_transitions.add((x,y,ActionsNew.left))
        for x in [self.grid_x-1]:
            for y in range(self.grid_y): # No right turn to buildings
                self.forbidden_transitions.add((x,y,ActionsNew.right))

        for y in [0]:
            for x in range(self.grid_x): # No right turn to buildings
                self.forbidden_transitions.add((x,y,ActionsNew.down))
        for y in [self.grid_y-1]:
            for x in range(self.grid_x): # No right turn to buildings
                self.forbidden_transitions.add((x,y,ActionsNew.up))

        # Adding the Block at (x=2, y=2)
        for y in [1]:
            for x in [0]: # No right turn to buildings
                self.forbidden_transitions.add((x,y,ActionsNew.right))
            for x in [2]: # No left turn to buildings
                self.forbidden_transitions.add((x,y,ActionsNew.left))
        
        for x in [1]:
            for y in [0]: # No going up to buildings
                self.forbidden_transitions.add((x,y,ActionsNew.up))
            for y in [2]: # No going down to buildings
                self.forbidden_transitions.add((x,y,ActionsNew.down))
        

# Define the epsilon-greedy policy
    def choose_action(self, Q_slice, state_x, 
                      state_y, epsilon):
        
        if np.random.rand() < epsilon:
            return np.random.randint(self.actions_num)
        else:
            if np.max(Q_slice) == 0:
                return np.random.randint(self.actions_num)
            else:
                return np.argmax(Q_slice)



def play():
    # commands
    str_to_action = {"w":ActionsNew.up.value,"d":ActionsNew.right.value,
                     "s":ActionsNew.down.value,"a":ActionsNew.left.value}
    
    # Setup the interactive visualization
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.canvas.manager.set_window_title('Grid World Environment')
    
    # Action buttons
    button_up = Button(plt.axes([0.7, 0.05, 0.1, 0.075]), 'Up (w)')
    button_down = Button(plt.axes([0.7, 0.15, 0.1, 0.075]), 'Down (s)')
    button_left = Button(plt.axes([0.6, 0.1, 0.1, 0.075]), 'Left (a)')
    button_right = Button(plt.axes([0.8, 0.1, 0.1, 0.075]), 'Right (d)')
    button_reset = Button(plt.axes([0.7, 0.25, 0.1, 0.075]), 'Reset')
    
    # Initialize action input variable and game state variables
    action_input = [None]
    game_state = {'total_reward': 0, 'step_count': 0}  # Use dict to allow modification in nested function
    
    # Button callback functions
    def on_up(event): action_input[0] = "w"; plt.draw()
    def on_down(event): action_input[0] = "s"; plt.draw()
    def on_left(event): action_input[0] = "a"; plt.draw()
    def on_right(event): action_input[0] = "d"; plt.draw()
    def on_reset(event): 
        game.agent = (0, 0)  # Reset agent to starting position
        game_state['total_reward'] = 0  # Reset total reward
        game_state['step_count'] = 0   # Reset step count
        visualize_environment(game, total_reward=game_state['total_reward'])
        print(f"\nGame reset! Agent back to {game.agent}, total reward reset to 0")
    
    # Connect callbacks
    button_up.on_clicked(on_up)
    button_down.on_clicked(on_down)
    button_left.on_clicked(on_left)
    button_right.on_clicked(on_right)
    button_reset.on_clicked(on_reset)
    
    def visualize_environment(game, reward=None, action_taken=None, total_reward=None):
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        
        # Get grid dimensions
        grid_width, grid_height = game.grid_x, game.grid_y
        
        # Draw grid
        for i in range(grid_width+1):
            ax1.plot([i, i], [0, grid_height], 'k-', alpha=0.3)
        for j in range(grid_height+1):
            ax1.plot([0, grid_width], [j, j], 'k-', alpha=0.3)
        
        # Draw walls (blocked cells)
        wall_color = 'black'
        # Add the wall at (1,1) based on forbidden transitions
        rect = patches.Rectangle((1, 1), 1, 1, linewidth=2, edgecolor='k', 
                                facecolor=wall_color, alpha=0.8)
        ax1.add_patch(rect)
        ax1.text(1.5, 1.5, 'Wall', ha='center', va='center', 
                color='white', fontsize=10, fontweight='bold')
        
        # Draw goal locations
        for pos in game.goal_locations:
            x, y = pos
            rect = patches.Rectangle((x, y), 1, 1, linewidth=2, edgecolor='green', 
                                    facecolor='lightgreen', alpha=0.7)
            ax1.add_patch(rect)
            ax1.text(x + 0.5, y + 0.5, 'GOAL', ha='center', va='center', 
                    color='darkgreen', fontsize=10, fontweight='bold')
        
        # Draw penalty locations
        for pos in game.penalty_locations:
            x, y = pos
            rect = patches.Rectangle((x, y), 1, 1, linewidth=2, edgecolor='red', 
                                    facecolor='lightcoral', alpha=0.7)
            ax1.add_patch(rect)
            ax1.text(x + 0.5, y + 0.5, 'PIT', ha='center', va='center', 
                    color='darkred', fontsize=10, fontweight='bold')
        
        # Draw agent
        agent_x, agent_y = game.agent
        agent_circle = plt.Circle((agent_x + 0.5, agent_y + 0.5), 0.3, 
                                 color='blue', linewidth=2, edgecolor='navy')
        ax1.add_patch(agent_circle)
        ax1.text(agent_x + 0.5, agent_y + 0.5, 'A', ha='center', va='center', 
                color='white', fontsize=12, fontweight='bold')
        
        # Set axis properties
        ax1.set_xlim(0, grid_width)
        ax1.set_ylim(0, grid_height)
        ax1.set_aspect('equal')
        ax1.set_title('Grid World Environment')
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')
        
        # Add coordinate labels
        ax1.set_xticks(range(grid_width + 1))
        ax1.set_yticks(range(grid_height + 1))
        
        # Display game information
        ax2.axis('off')
        info_text = "Game Information:\n\n"
        info_text += f"Agent Position: {game.agent}\n"
        
        if reward is not None:
            info_text += f"Last Reward: {reward:.3f}\n"
            
        if total_reward is not None:
            info_text += f"Total Reward: {total_reward:.3f}\n"
            
        if action_taken is not None:
            action_names = {0: 'UP', 1: 'DOWN', 2: 'RIGHT', 3: 'LEFT'}
            info_text += f"Action Taken: {action_names.get(action_taken, 'UNKNOWN')}\n"
        
        info_text += "\nEnvironment:\n"
        info_text += "• Green: Goal (+1 reward)\n"
        info_text += "• Red: Pit (-1 reward)\n"
        info_text += "• Black: Wall (blocked)\n"
        info_text += "• Default step: -0.04 reward\n\n"
        
        info_text += "Slip Mechanics:\n"
        info_text += "• 80% intended action\n"
        info_text += "• 10% slip left\n"
        info_text += "• 10% slip right\n\n"
        
        info_text += "Controls:\n"
        info_text += "  w: Move up\n"
        info_text += "  s: Move down\n"
        info_text += "  a: Move left\n"
        info_text += "  d: Move right\n"
        info_text += "  Reset: Start over\n\n"
        
        info_text += "Click buttons or use keyboard!"
        
        ax2.text(0.05, 0.95, info_text, fontsize=11, va='top', transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
    
    print("Starting Grid World Interactive Game")
    
    # Initialize game
    game = env()
    game.map()  # Setup the environment
    
    visualize_environment(game, total_reward=game_state['total_reward'])
    
    while True:
        # Display current state
        current_reward, _ = game.get_reward()
        visualize_environment(game, current_reward, total_reward=game_state['total_reward'])
        
        # Check for terminal conditions
        if game.agent in game.goal_locations:
            ax1.text(2, 1.5, "GOAL REACHED!", fontsize=20, ha='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.8),
                    color='white', fontweight='bold')
            plt.draw()
            print(f"\nGoal reached! Total reward: {game_state['total_reward']:.3f} in {game_state['step_count']} steps")
            plt.pause(3)
            break
        elif game.agent in game.penalty_locations:
            ax1.text(2, 1.5, "FELL IN PIT!", fontsize=20, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8),
                    color='white', fontweight='bold')
            plt.draw()
            print(f"\nFell in pit! Total reward: {game_state['total_reward']:.3f} in {game_state['step_count']} steps")
            plt.pause(3)
            break
        
        # Get action input
        action_input[0] = None
        print(f"\nStep {game_state['step_count'] + 1} - Position {game.agent} - Total Reward: {game_state['total_reward']:.3f}")
        print("Action (w/a/s/d)? ", end="", flush=True)
        
        while action_input[0] is None:
            plt.pause(0.1)
            # Also accept keyboard input
            try:
                import select
                import sys
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    a = input().strip().lower()
                    if a in str_to_action:
                        action_input[0] = a
                        break
            except:
                pass  # Handle systems without select module
        
        a = action_input[0]
        
        # Execute action
        if a in str_to_action:
            intended_action = str_to_action[a]
            old_pos = game.agent
            new_x, new_y = game.execute_action(game.agent[0], game.agent[1], intended_action)
            reward, _ = game.get_reward()
            game_state['total_reward'] += reward
            game_state['step_count'] += 1
            
            # Show what happened
            if (new_x, new_y) != old_pos:
                print(f"Moved from {old_pos} to {(new_x, new_y)}, reward: {reward:.3f}")
            else:
                print(f"Stayed at {old_pos} (blocked), reward: {reward:.3f}")
                
            visualize_environment(game, reward, intended_action, game_state['total_reward'])
        else:
            print("Invalid action. Use w/a/s/d.")
    
    plt.ioff()  # Turn off interactive mode when done
    input("Press Enter to close...")
    plt.close()

# This code allows to play the game interactively

if __name__ == '__main__':
    play()
    play()
