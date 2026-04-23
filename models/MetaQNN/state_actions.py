import os
import json
import pickle
from collections import deque

from models.MetaQNN.config.rl_config import *
from models.MetaQNN.config.train_config import *


def get_possible_actions(state):
    """Gets all actions for a given state"""

    possible_actions = []

    # If start state, initialize state so the following lines don't crash lol
    if state is None:
        state = { 'layer_type' : None, 'layer_depth' : 0, 'representation_size': IMAGE_SIZE }

    layer_type = state['layer_type']
    layer_depth = state['layer_depth']

    if layer_depth >= MAX_DEPTH:
        # If at max depth, must go to terminal state
        possible_actions.append({ 'layer_type' : TERMINATION })

    elif layer_type is None:
        # If initial state, go to convolution or pooling
        possible_actions.extend(get_convolution_actions(layer_depth=layer_depth, representation_size=state['representation_size']))
        possible_actions.extend(get_pooling_actions(layer_depth=layer_depth, representation_size=state['representation_size']))

    elif layer_type == CONVOLUTION:
        # Convolution layers can go to any layer
        possible_actions.extend(get_convolution_actions(layer_depth=layer_depth, representation_size=state['representation_size']))
        possible_actions.extend(get_pooling_actions(layer_depth=layer_depth, representation_size=state['representation_size']))
        possible_actions.extend(get_fully_connected_actions(layer_depth=layer_depth, representation_size=state['representation_size'], num_consecutive=0))
        possible_actions.append({ 'layer_type' : TERMINATION })

    elif layer_type == POOLING:
        # Pooling layers can go to convolution, FC, or terminal
        possible_actions.extend(get_convolution_actions(layer_depth=layer_depth, representation_size=state['representation_size']))
        possible_actions.extend(get_fully_connected_actions(layer_depth=layer_depth, representation_size=state['representation_size'], num_consecutive=0))
        possible_actions.append({ 'layer_type' : TERMINATION })

    elif layer_type == FULLY_CONNECTED:
        # FC layers can go to FC or terminal
        possible_actions.extend(get_fully_connected_actions(
            layer_depth=layer_depth, 
            num_consecutive=state['num_consecutive'], 
            representation_size=state['representation_size'], 
            curr_num_neurons=state['num_neurons'])
        )
        possible_actions.append({ 'layer_type' : TERMINATION })

    else:
        print("should not happen prolly unless we decide to implement GAP")

    return possible_actions


def get_action_values(Q, state, actions):
    """Gets the action values for a list of actions of a given state"""
    # Get action values for each state-action pair (or default if not yet explored)
    state_string = to_string(state)
    action_values = [
        # Get action value by converting state to string
        Q.get(state_string, {}).get(to_string(action), INITIAL_Q_VALUE)
        for action in actions
    ]

    return action_values


def get_convolution_actions(layer_depth, representation_size):
    convolution_actions = []
    
    # Build all convolution types
    for num_channels in AVAIL_NUM_CHANNELS:
        for kernel_size in AVAIL_KERNEL_SIZES:
            # Only allow kernel_size < curr_representation_size
            if kernel_size >= representation_size:
                continue

            convolution_actions.append({ 
                'layer_type' : CONVOLUTION, 
                'out_channels' : num_channels, 
                'kernel_size' : kernel_size,
                'layer_depth' : layer_depth + 1,
                'representation_size': representation_size
            })

    return convolution_actions


def get_pooling_actions(layer_depth, representation_size):
    pooling_actions = []

    # Build all pooling types:
    for (kernel_size, stride) in AVAIL_KERNEL_SIZE_STRIDES:
        # Only allow kernel_size < curr_representation_size
        if kernel_size >= representation_size:
            continue

        pooling_actions.append({ 
            'layer_type' : POOLING, 
            'kernel_size' : kernel_size, 
            'stride' : stride,
            'layer_depth' : layer_depth + 1,
            'representation_size': (representation_size - kernel_size) // stride + 1
        })

    return pooling_actions


def get_fully_connected_actions(num_consecutive, layer_depth, representation_size, curr_num_neurons=None):
    # If already at max consecutive FC, no fully connected actions available
    if num_consecutive >= MAX_CONSECUTIVE_FC:
        return []
    
    fully_connected_actions = []

    # Build all FC types:
    for num_neurons in AVAIL_NUM_NEURONS:
        # Only allow neurons <= number of current neurons
        if curr_num_neurons and num_neurons > curr_num_neurons:
            continue
        
        # Only allow transitions to FC if representation_size <= 8
        if representation_size > 8:
            continue

        fully_connected_actions.append({ 
            'layer_type' : FULLY_CONNECTED, 
            'num_neurons' : num_neurons, 
            'num_consecutive' : num_consecutive + 1,
            'layer_depth' : layer_depth + 1,
            'representation_size' : 1
        })

    return fully_connected_actions


def save_Q(Q, file_path):
    """
    Q looks like:
    dict {
        "{ 'layer_type' : 0, ... }" : 
            {
                "{ 'layer_type' : 0, ... }" : 0.5,
                "{ 'layer_type' : 0, ... }" : 0.6,
                "{ 'layer_type' : 0, ... }" : 0.3
            }
        ,
        "{ 'layer_type' : 0, ... }" : 
            {
                "{ 'layer_type' : 0, ... }" : 0.5,
                "{ 'layer_type' : 0, ... }" : 0.6,
                "{ 'layer_type' : 0, ... }" : 0.3
            }
        ...
    }
    """
    with open(file_path, 'w') as file:
        json.dump(Q, file, indent=4)


def load_Q(file_path):
    if file_path is None or not os.path.exists(file_path):
        # If Q file path does not exist, return an empty one
        return {}

    with open(file_path) as file:
        Q = json.load(file)

    return Q


def save_buffer(replay_buffer, file_path):
    """Buffer is a deque with elements consisting of 
    (state_sequence, action_sequence, accuracy)
    ([None, "state1", "state2", ...], [action1, action2, ...], 70%)
    """
    with open(file_path, 'wb') as file:
        pickle.dump(replay_buffer, file)



def load_buffer(file_path):
    if file_path is None or not os.path.exists(file_path):
        # If buffer file path does not exist, return an empty one
        return deque()

    with open(file_path, 'rb') as file:
        replay_buffer = pickle.load(file)

        return replay_buffer


def to_string(state):
    return json.dumps(state)


def parse_state(state_string):
    return json.loads(state_string)


if __name__ == "__main__":
    loaded_Q = load_Q('metaqnn/saves/Q_values.json')
    loaded_buffer = load_buffer('metaqnn/saves/replay_buffer.pkl')

    # print(loaded_Q)

    state = {'layer_type': 0, 'out_channels': 64, 'kernel_size': 5, 'layer_depth': 3, 'representation_size': 7}
    possible_actions = get_possible_actions(state)
    for action in possible_actions:
        print(action)
