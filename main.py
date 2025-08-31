import numpy as np

def main():
    np.set_printoptions(linewidth=200)
    environment_rows = 13
    environment_columns = 17

    q_values = np.zeros((environment_rows, environment_columns, 4))
    actions = ['up', 'right', 'down', 'left']
    rewards = np.full((environment_rows, environment_columns), -100.)

    roads = {}
    for i in range(environment_rows):
        if i == 0 or i == 3 or i == 7 or i == 12:
            roads[i] = [i for i in range(environment_columns)]
        elif i == 1 or i == 2:
            roads[i] = [0, 8]
        else:
            roads[i] = [0, 4, 8, 12, 16]

    for row_index in range(environment_rows):
        for column_index in roads[row_index]:
            rewards[row_index, column_index] = -1.

    rewards[0, 3] = 100
    rewards[7, 11] = 100
    print(rewards)

    def is_terminal_state(current_row_index, current_column_index):
        if rewards[current_row_index, current_column_index] == -1.:
            return False
        else:
            return True

    def get_starting_location():
        current_row_index = np.random.randint(environment_rows)
        current_column_index = np.random.randint(environment_columns)
        while is_terminal_state(current_row_index, current_column_index):
            current_row_index = np.random.randint(environment_rows)
            current_column_index = np.random.randint(environment_columns)
        return current_row_index, current_column_index

    def get_next_action(current_row_index, current_column_index, epsilon):
        if np.random.random() < epsilon:
            return np.argmax(q_values[current_row_index, current_column_index])
        else:
            return np.random.randint(4)

    def get_next_location(current_row_index, current_column_index, action_index):
        new_row_index = current_row_index
        new_column_index = current_column_index
        if actions[action_index] == 'up' and current_row_index > 0:
            new_row_index -= 1
        elif actions[action_index] == 'right' and current_column_index < environment_columns - 1:
            new_column_index += 1
        elif actions[action_index] == 'down' and current_row_index < environment_rows - 1:
            new_row_index += 1
        elif actions[action_index] == 'left' and current_column_index > 0:
            new_column_index -= 1
        return new_row_index, new_column_index

    def get_shortest_path(start_row_index, start_column_index):
        if is_terminal_state(start_row_index, start_column_index):
            return []
        else:
            current_row_index, current_column_index = start_row_index, start_column_index
            shortest_path = []
            shortest_path.append([current_row_index, current_column_index])
            while not is_terminal_state(current_row_index, current_column_index):
                action_index = get_next_action(current_row_index, current_column_index, 1.)
                current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
                shortest_path.append([current_row_index, current_column_index])
            return shortest_path
        
    epsilon = 0.9
    discount_factor = 0.9
    learning_rate = 0.9

    for episode in range(1000):
        row_index, column_index = get_starting_location()
        while not is_terminal_state(row_index, column_index):
            action_index = get_next_action(row_index, column_index, epsilon)
            old_row_index, old_column_index = row_index, column_index
            row_index, column_index = get_next_location(row_index, column_index, action_index)
            reward = rewards[row_index, column_index]
            old_q_value = q_values[old_row_index, old_column_index, action_index]
            temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            q_values[old_row_index, old_column_index, action_index] = new_q_value

    print('Training complete!')
    print(q_values)
    print(get_shortest_path(3, 16))
    
    path = get_shortest_path(3, 16)
    path.reverse()
    print(path)

if __name__ == "__main__":
    main()
