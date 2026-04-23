import random
import torch.nn as nn

from models.MetaQNN.config.rl_config import *
from models.MetaQNN.config.train_config import *

from models.MetaQNN.state_actions import get_possible_actions, get_action_values, to_string
from models.MetaQNN.state_actions import load_Q, save_Q, load_buffer, save_buffer

from models.MetaQNN.train import train, initialize_datasets, create_model, get_optimizer, get_scheduler
from models.MetaQNN.logging import save_model_metrics


Q_file_path = 'models/MetaQNN/logs/Q_values.json'
buffer_file_path = 'models/MetaQNN/logs/replay_buffer.pkl'
log_json_path = 'models/MetaQNN/logs/logs.json'


def q_learning(num_episodes, start_episode=0):
    # Initialize Q and replay buffer
    Q = load_Q(Q_file_path)
    replay_buffer = load_buffer(buffer_file_path)

    # Initialize datasets
    train_loader, val_loader, _ = initialize_datasets()

    print("Initialized Q, replay buffer, and datasets")

    for episode in range(start_episode, num_episodes):
        # Calculate epsilon based on how many models we've trained
        epsilon = get_epsilon(len(replay_buffer))

        S, U = sample_new_network(Q, epsilon=epsilon)
        model = create_model(U)
        optimizer = get_optimizer(model)
        scheduler = get_scheduler(optimizer)

        print(f"Sampled network #{episode+1} | Epsilon {epsilon}")
        for layer in U:
            print(layer)

        accuracy = train(
            model, num_epochs=NUM_EPOCHS, train_loader=train_loader, val_loader=val_loader,
            loss_func=nn.CrossEntropyLoss(), optimizer=optimizer, scheduler=scheduler
        )

        replay_buffer.append((S, U, accuracy))
        save_model_metrics(U, epsilon, accuracy, log_json_path)

        for _ in range(min(len(replay_buffer) * 10, REPLAY_NUMBER)):
            # Sample from replay buffer
            S_sample, U_sample, accuracy_sample = random.choice(replay_buffer)

            # Update Q values
            Q = update_Q_values(Q, S_sample, U_sample, accuracy_sample)

        # Save new Q values and buffer
        save_Q(Q, Q_file_path)
        save_buffer(replay_buffer, buffer_file_path)


def sample_new_network(Q, epsilon):
    # Initialize state and action sequences
    state_sequence = [None]
    action_sequence = []

    while True:
        rand = random.random()

        if rand > epsilon:
            # Take the greedy action
            possible_actions = get_possible_actions(state_sequence[-1])
            action_values = get_action_values(Q, state_sequence[-1], possible_actions)
            
            max_val = max(action_values)
            best_actions = [a for a, v in zip(possible_actions, action_values) if v == max_val]
            next_layer = random.choice(best_actions)

        else:
            # Take a random action
            possible_actions = get_possible_actions(state_sequence[-1])
            rand_action = random.randint(0, len(possible_actions)-1)

            next_layer = possible_actions[rand_action]

        action_sequence.append(next_layer)

        if next_layer['layer_type'] == TERMINATION:
            break

        state_sequence.append(next_layer)

    return state_sequence, action_sequence


def update_Q_values(Q, S, U, accuracy):
    target = accuracy

    for i in range(len(S) - 1, -1, -1):
        state = to_string(S[i])
        action = to_string(U[i])

        if state not in Q:
            Q[state] = {}

        # Update Q value
        Q[state][action] = (1 - ALPHA) * Q[state].get(action, INITIAL_Q_VALUE) + ALPHA * target

        # Update target to be best action value
        possible_actions = get_possible_actions(S[i])

        # Get all explored actions for the state
        explored_actions = Q[state].keys()

        # Check if all possible actions are fully explored
        fully_explored = all(to_string(a) in explored_actions for a in possible_actions)

        # Calculate max action value observed so far
        max_observed = max(Q[state].values()) if Q[state] else INITIAL_Q_VALUE
        
        # If we haven't explored all actions, best action should be at least the INITIAL_Q_VALUE
        if not fully_explored:
            target = max(INITIAL_Q_VALUE, max_observed)
        else:
            target = max_observed

    return Q


def get_epsilon(models_trained):
    # Calculate epsilon based on how many models we've trained
    i = 0
    while i < len(EPSILON_SCHEDULER):
        if models_trained < EPSILON_SCHEDULER[i]:
            break

        i += 1

    epsilon = 1.0 - 0.1 * i

    return epsilon


if __name__ == "__main__":
    buffer = load_buffer(buffer_file_path)
    q_learning(num_episodes=300, start_episode=len(buffer))
