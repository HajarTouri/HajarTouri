import numpy as np

# Policy matrix representing directions
policy = np.array([
    ['S', 'S', '', ''],
    ['E', 'S', '', 'N'],
    ['', 'E', 'E', 'N'],
])

# Reward for moving into an empty cell
rmove = -0.1      # penalty
passreward = 40   # Reward for passing
volcanoreward = -50  # Reward for landing on volcano
gamma = 0.9       # discount factor
slipProb = 0.5   # Probability of slipping

# Reward matrix
rewards = np.array([
    [rmove, rmove, volcanoreward, passreward],
    [rmove, rmove, volcanoreward, rmove],
    [2, rmove, rmove, rmove],
])
ends = np.array([
    [False, False, True, True],
    [False, False, True, False],
    [True, False, False, False],
])

op_policy = {}  # Dictionary to store optimal policy

def move(pos, policy, slProb):
    # Helper function to decrement position
    def decrement(x):
        return max(x - 1, 0)

    # Helper function to increment position
    def increment(x, to):
        return min(x + 1, to)

    # Check for slipping
    slip = np.random.random() <= slProb  # Generate an error of less than 0.2

    pos = list(pos)
    # If slipping, select a random direction
    if slip:
        next_step = np.random.random()
        if next_step >= 0.75:               # N
            pos[0] = decrement(pos[0])
        elif next_step >= 0.5:              # S
            pos[0] = increment(pos[0], 2)
        elif next_step >= 0.25:             # W
            pos[1] = decrement(pos[1])
        else:                               # E
            pos[1] = increment(pos[1], 3)
    else:
        # If not slipping, move according to the given action
        if policy == 'N':
            pos[0] = decrement(pos[0])
        elif policy == 'S':
            pos[0] = increment(pos[0], 2)
        elif policy == 'W':
            pos[1] = decrement(pos[1])
        else:
            pos[1] = increment(pos[1], 3)
    return pos

current = (1, 0)  # Starting point
utility = 0.      # Initialization

# Run the loop for 10 iterations
for i in range(10):
    print(policy[current], end=" ")
    current = tuple(move(current, policy[current], slipProb))
    utility += rewards[current[0], current[1]]  # Correct indexing for rewards
    print(rewards[current[0], current[1]], current[0] + 1, current[1] + 1)
    if ends[current]:
        break

print(utility)  # The gain

move((2, 1), 'S', 0.3)  # state with which we will begin

# Initialize the value matrix to 0
value = np.zeros((3, 4))

actions = ['S', 'E', 'N', 'W']

for _ in range(1000):  # Increase the number of iterations for better convergence
    for i in range(3):
        for j in range(4):
            current = (i, j)
            if ends[current]:
                value[current] = rewards[current[0], current[1]]  # Correct indexing for rewards
            else:
                pciat = np.array([tuple(move(current, pcy, -1)) for pcy in actions])

                pcy_termes = [(rewards[pcy[0], pcy[1]] + gamma * value[pcy[0], pcy[1]]) for pcy in pciat]

                pcy_termes_opt = np.max(pcy_termes)

                op_policy[current] = actions[np.argmax(pcy_termes)]

                value[current] = (1 - slipProb) * pcy_termes_opt + slipProb * np.mean(
                    [(rewards[move(current, a, 0)[0], move(current, a, 0)[1]] +
                      gamma * value[move(current, a, 0)[0], move(current, a, 0)[1]]) for a in actions])

# Print optimal policy
print("Optimal Policy:")
for i in range(3):
    for j in range(4):
        if ends[i, j]:
            print("T", end="\t")
        else:
            print(op_policy[(i, j)], end="\t")
    print()

# Print optimal values
print("Optimal Values:")
print(value)
