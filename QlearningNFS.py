
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Paramters for learning
env = gym.make('Breakout-v0')
frame = env.reset()
episodes = 50000
steps_max = 200
epsilon = 1.0
epsilon_max = 1.0
epsilon_min = 0.01
rate_decay = 0.001
rewards = []
rewards_wo_mean = []
gamma = 0.95
Q_Table = {}
lr = 0.7
# Preprocessing the image
def preprocess(img):
    return img[::4, ::4]

# Paddle tracker
def paddle_tracker(frame):
    column_number = []
    for j in range(2, 38):
        if 200 in frame[48, j, :]:
            column_number.append(j)
    first_point = column_number[0]
    last_point = column_number[-1]
    return [first_point, last_point]

# Ball tracker
def ball_tracker(frame):
    row_number = 0
    column_number = 0
    for row in range(24, 48):
        if 200 in frame[row, :, :]:
            row_number = row
            break

    for j in range(40):
        if 200 in frame[row_number, j, :]:
            column_number = j
            break

    return [row_number, column_number]

# Ball movement
def ball_move(old_frame, new_frame):

    ball_oldcoordinates = ball_tracker(old_frame)
    ball_newcoordinates = ball_tracker(new_frame)

    if ball_oldcoordinates[0] > ball_newcoordinates[0] and ball_oldcoordinates[1] < ball_newcoordinates[1]:
        ball_moving = "Upright"
    elif ball_oldcoordinates[0] > ball_newcoordinates[0] and ball_oldcoordinates[1] > ball_newcoordinates[1]:
        ball_moving = "Upleft"
    elif ball_oldcoordinates[0] > ball_newcoordinates[0] and ball_oldcoordinates[1] == ball_newcoordinates[1]:
        ball_moving = "Upstraight"
    if ball_oldcoordinates[0] < ball_newcoordinates[0] and ball_oldcoordinates[1] < ball_newcoordinates[1]:
        ball_moving = "Downright"
    elif ball_oldcoordinates[0] < ball_newcoordinates[0] and ball_oldcoordinates[1] > ball_newcoordinates[1]:
        ball_moving = "Downleft"
    elif ball_oldcoordinates[0] < ball_newcoordinates[0] and ball_oldcoordinates[1] == ball_newcoordinates[1]:
        ball_moving = "Downstraight"

    elif ball_oldcoordinates[0] == ball_newcoordinates[0] and ball_oldcoordinates[1] < ball_newcoordinates[1]:
        ball_moving = "moveright"
    elif ball_oldcoordinates[0] == ball_newcoordinates[0] and ball_oldcoordinates[1] > ball_newcoordinates[1]:
        ball_moving = "moveleft"
    else:
        ball_moving = "stationary"

    return ball_moving

for episode in range(episodes):
    step = 0
    done = False
    rewards_total = []
    env.reset()
    frame, reward, done, info = env.step(1)
    steps_max = 100
    old_frame = preprocess(frame)
    ball_row = ball_tracker(old_frame)[0]
    ball_column = ball_tracker(old_frame)[1]
    paddle_firstposition = paddle_tracker(old_frame)[0]
    paddle_lastposition = paddle_tracker(old_frame)[1]
    ball_movement = ball_move(old_frame, old_frame)
    old_state_value = str(ball_row) + str(ball_column) + str(paddle_firstposition) + str(paddle_lastposition) + str(
        ball_movement)
    for step in range(steps_max):
        num_random = random.random()

        # get random action
        if num_random > epsilon:
            if old_state_value in Q_Table:
                action = Q_Table[old_state_value].index(max(Q_Table[old_state_value]))
            else:
                action = env.action_space.sample()
        else:
            action = env.action_space.sample()
        # print(env.action_space.n)
        # Take the action (a) and observe the outcome state(s') and reward (r)
        frame, reward, done, info = env.step(action)
        new_frame = preprocess(frame)
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        # qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        ball_row = ball_tracker(new_frame)[0]
        ball_row_old = ball_tracker(old_frame)[0]
        if ball_row<=23:
            continue
        ball_column = ball_tracker(new_frame)[1]
        paddle_firstposition = paddle_tracker(new_frame)[0]
        paddle_lastposition = paddle_tracker(new_frame)[1]
        ball_movement = ball_move(old_frame, new_frame)
        if ball_row > ball_row_old:
             if ball_column in range(paddle_firstposition,paddle_lastposition):
                reward = reward + 10
        #         Q_dict["ball_row"].append(ball_row)
        #         Q_dict["ball_column"].append(ball_column)
        #         Q_dict["paddle_firstposition"].append(paddle_firstposition)
        #         Q_dict["paddle_lastposition"].append(paddle_lastposition)
        #         Q_dict["ball_movement"].append(ball_movement)
        #old_state_value = state_value
        if done == True:
            reward = reward - 2
            state_value = str(ball_row) + str(ball_column) + str(paddle_firstposition) + str(paddle_lastposition) + str(ball_movement)
            index = int(action)
            if old_state_value in Q_Table:
                if state_value not in Q_Table:
                    Q_Table[state_value] = [0,0,0,0]
                    Q_Table[old_state_value][action] = Q_Table[old_state_value][action] + lr * (reward + gamma * max(Q_Table[state_value]) - Q_Table[old_state_value][action])
                else:
                    Q_Table[old_state_value][action] = Q_Table[old_state_value][action] + lr * (reward + gamma * max(Q_Table[state_value]) - Q_Table[old_state_value][action])
            else:
                if state_value not in Q_Table:
                    Q_Table[state_value] = [0,0,0,0]
                    Q_Table[old_state_value] = [0, 0, 0, 0]
                    Q_Table[old_state_value][action] = Q_Table[old_state_value][action] + lr * (reward + gamma * max(Q_Table[state_value]) - Q_Table[old_state_value][action])
                else:
                    Q_Table[old_state_value] = [0, 0, 0, 0]
                    Q_Table[old_state_value][action] = Q_Table[old_state_value][action] + lr * (reward + gamma * max(Q_Table[state_value]) - Q_Table[old_state_value][action])
            break
        state_value = str(ball_row) + str(ball_column) + str(paddle_firstposition) + str(paddle_lastposition) + str(
            ball_movement)
        index = int(action)
        if old_state_value in Q_Table:
            if state_value not in Q_Table:
                Q_Table[state_value] = [0,0,0,0]
                Q_Table[old_state_value][action] = Q_Table[old_state_value][action] + lr * (
                    reward + gamma * max(Q_Table[state_value]) - Q_Table[old_state_value][action])
            else:
                Q_Table[old_state_value][action] = Q_Table[old_state_value][action] + lr * (
                    reward + gamma * max(Q_Table[state_value]) - Q_Table[old_state_value][action])
        else:
            if state_value not in Q_Table:
                Q_Table[state_value] = [0,0,0,0]
                Q_Table[old_state_value] = [0, 0, 0, 0]
                Q_Table[old_state_value][action] = Q_Table[old_state_value][action] + lr * (
                reward + gamma * max(Q_Table[state_value]) - Q_Table[old_state_value][action])
            else:
                Q_Table[old_state_value] = [0, 0, 0, 0]
                Q_Table[old_state_value][action] = Q_Table[old_state_value][action] + lr * (
                    reward + gamma * max(Q_Table[state_value]) - Q_Table[old_state_value][action])

        old_frame = new_frame
        old_state_value = state_value
        rewards_total.append(reward)
        # Our new state is state
        #env.render()
        # If done (if we're dead) : finish episode
       
            # Reduce epsilon (because we need less and less exploration)
    #if episode % 1000 == 0:
        #print(pd.DataFrame(Q_Table).T)
    epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-rate_decay * episode)
    rewards.append(np.mean(rewards_total))
    rewards_wo_mean.append(rewards_total)
    # print(rewards)




# In[4]:



import pickle
output = open('finalQ2nfsgs.pkl', 'wb')
pickle.dump(game_score, output)
output.close()