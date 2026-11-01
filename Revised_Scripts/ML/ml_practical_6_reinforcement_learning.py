# ML Practical 6: Reinforcement Learning (Q-Learning Maze)
import numpy as np
import random

# 0 = Free, 1 = Wall, Start = (0, 0), Goal = (3, 3)
maze = [
    [0, 0, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 0, 0],
    [0, 1, 1, 0]
]
start, goal = (0, 0), (3, 3)
actions = ['U', 'D', 'L', 'R']
rows, cols = len(maze), len(maze[0])
Q = np.zeros((rows, cols, len(actions)))

alpha, gamma, epsilon = 0.7, 0.9, 0.9
episodes = 500

def step(state, action):
    r, c = state
    if action == 'U': r -= 1
    if action == 'D': r += 1
    if action == 'L': c -= 1
    if action == 'R': c += 1
    if r < 0 or r >= rows or c < 0 or c >= cols or maze[r][c] == 1:
        return state, -1  # Wall penalty
    if (r, c) == goal:
        return (r, c), 10  # Goal reward
    return (r, c), -0.1   # Step cost

# Training Loop
for _ in range(episodes):
    state = start
    while state != goal:
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = actions[np.argmax(Q[state[0], state[1]])]
        
        next_state, reward = step(state, action)
        a_idx = actions.index(action)
        best_next = np.max(Q[next_state[0], next_state[1]])
        Q[state[0], state[1], a_idx] += alpha * (reward + gamma * best_next - Q[state[0], state[1], a_idx])
        state = next_state
    epsilon = max(0.05, epsilon * 0.995)

# Test Learned Path
state = start
path = [state]
steps = 0
while state != goal and steps < 25:
    action = actions[np.argmax(Q[state[0], state[1]])]
    state, _ = step(state, action)
    path.append(state)
    steps += 1

print("Learned Path:", path)
