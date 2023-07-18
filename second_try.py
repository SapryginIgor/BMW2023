import math
import random

import numpy as np
import time
import tqdm
import cv2
import gymnasium as gym
import os.path

cell_size = -1
model_to_map = {}

# мб здесь проблема, но тут вроде просто. передаю верхние левые и нижние правые углы
def rectangle_intersection(top_left1, bottom_right1, top_left2, bottom_right2):
    x1 = max(top_left1[0], top_left2[0])
    x2 = min(bottom_right1[0], bottom_right2[0])
    y1 = max(top_left1[1], top_left2[1])
    y2 = min(bottom_right1[1], bottom_right2[1])
    if x1 < x2 and y1 < y2:
        return True
    return False


# видимо пока работает плохо
def bumped(agent_pos, angle):
    for corner in corners:
        marker_corner = corner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = marker_corner
        dot = np.array(model_to_map[agent_pos])
        if angle == 0:
            robot_top_left = dot + np.array([-2.5, -2.5]) * cell_size
            robot_top_right = dot + np.array([7.5, 3.5]) * cell_size
        elif angle == 90:
            robot_top_left = dot + np.array([-2.5, -2.5]) * cell_size
            robot_top_right = dot + np.array([3.5, 7.5]) * cell_size
        elif angle == 180:
            robot_top_left = dot + np.array([-6.5, -2, 5]) * cell_size
            robot_top_right = dot + np.array([-3.5, 3.5]) * cell_size
        elif angle == 270:
            robot_top_left = dot + np.array([-2.5, -6.5]) * cell_size
            robot_top_right = dot + np.array([3.5, 3.5]) * cell_size
        if rectangle_intersection(robot_top_left, robot_top_right, topLeft, topRight):
            return True
    return True





def get_len(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)



def initialize_q_table(state_space, action_space):
  Qtable = np.zeros((state_space, action_space))
  return Qtable


# def greedy_policy(Qtable, state):
#     # Exploitation: take the action with the highest state, action value
#     action = np.argmax(Qtable[state][:])
#
#     return action
#
#
# def epsilon_greedy_policy(Qtable, state, epsilon):
#     # Randomly generate a number between 0 and 1
#     random_num = random.uniform(0, 1)
#     # if random_num > greater than epsilon --> exploitation
#     if random_num > epsilon:
#         # Take the action with the highest value given a state
#         # np.argmax can be useful here
#         action = greedy_policy(Qtable, state)
#     # else --> exploration
#     else:
#         action = env.action_space.sample()
#
#     return action
#
#
# def get_policy(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps):
#     if not os.path.exists('arr'):
#         Qtable = initialize_q_table(state_space, action_space)
#     else:
#         file = open("arr", "rb")
#         Qtable = np.load(file)
#     for episode in range(n_training_episodes):
#         # Reduce epsilon (because we need less and less exploration)
#         epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
#         # Reset the environment
#         state, info = env.reset()
#         step = 0
#         terminated = False
#         truncated = False
#
#         # repeat
#         for step in range(max_steps):
#             # Choose the action At using epsilon greedy policy
#             action = epsilon_greedy_policy(Qtable, state, epsilon)
#
#             # Take action At and observe Rt+1 and St+1
#             # Take the action (a) and observe the outcome state(s') and reward (r)
#             new_state, reward, terminated, truncated, info = env.step(action)
#
#             # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
#             Qtable[state][action] = Qtable[state][action] + learning_rate * (
#                         reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])
#
#             # If terminated or truncated finish the episode
#             if terminated or truncated:
#                 break
#
#             # Our next state is the new state
#             state = new_state
#     return Qtable


def cells_initialization(initial_marker):
    top_left, top_right, bottom_right, bottom_left = initial_marker
    cell_centers = {}
    for i in range(height):
        for j in range(width):
            v = model_to_map[i * width + j]
            cell_center = v + (bottom_left - top_left)/2 + (top_right - top_left)/2
            cell_centers[i * width + j] = cell_center.astype('int32')
            cv2.circle(detected_markers, (int(cell_center[0]),int(cell_center[1])), 4, (0, 255, 0), -1)
    return cell_centers

def make_frozen_lake(initial_marker):
    top_left, top_right, bottom_right, bottom_left = initial_marker
    frozen_lake = np.full((height, width), 'F')
    for dot in marker_centers:
        x0, y0 = dot
        x_const, y_const = (bottom_left - top_left)/2 + (top_right - top_left)/2 + top_left
        x_a, y_a = top_right - top_left
        x_b, y_b = bottom_left - top_left
        j = round(((x0-x_const)*(y_b/x_b) - (y0-y_const))/((x_a * y_b )/x_b - y_a))
        i = round(((x0-x_const)*(y_a/y_a) - (y0-y_const))/((x_b * y_a)/x_a - y_b))
        frozen_lake[i][j] = 'H'
        for d1 in range(0, 2):
            for d2 in range(0, 2):
                frozen_lake[i + d1][j + d2] = 'H'

    # frozen_lake[0, 0] = 'S'
    frozen_lake[height - 1, width - 1] = 'G'
    return frozen_lake


def build_map(id, corners, ids, height, width):
    for (marker, num) in zip(corners, ids):
        if num == id:
            marker_corner = marker
            break

    marker_corner = marker_corner.reshape((4, 2))
    (top_left, top_right, bottom_right, bottom_left) = marker_corner
    top_left = np.array(top_left)
    top_right = np.array(top_right)
    bottom_left = np.array(bottom_left)
    bottom_right = np.array(bottom_right)
    for i in range(height):
        for j in range(width):
            dot = top_left + (top_right - top_left) * j + (bottom_left - top_left)*i
            model_to_map[i * width + j] = dot.astype('int32')
    return (top_left, top_right, bottom_right, bottom_left)



def aruco_display(corners, ids, image):
    marker_centers = []
    if len(corners) > 0:
        ids = ids.flatten()
        for (marker_corner, markerID) in zip(corners, ids):
            corners = marker_corner.reshape((4, 2))
            (top_left, top_right, bottom_right, bottom_left) = corners

            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))

            cv2.line(image, top_left, top_right, (0, 255, 0), 2)
            cv2.line(image, top_right, bottom_right, (0, 255, 0), 2)
            cv2.line(image, bottom_right, bottom_left, (0, 255, 0), 2)
            cv2.line(image, bottom_left, top_left, (0, 255, 0), 2)

            c_x = int((top_left[0] + bottom_right[0]) / 2.0)
            c_y = int((top_left[1] + bottom_right[1]) / 2.0)
            cv2.circle(image, (c_x, c_y), 4, (0, 0, 255), -1)
            marker_centers.append((c_x, c_y))
            cv2.putText(image, str(markerID), (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
    return image, marker_centers


arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

arucoParams = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

cap = cv2.VideoCapture(2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 1-й этап: размечаем карту по одному кубику

height = 6
width = 10


ret, img = cap.read()

h, w, _ = img.shape

width = 1000
height = int(width * (h / w))
img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

corners, ids, rejected = detector.detectMarkers(img)
initial_marker = build_map(8, corners, ids, height, width)
while True:
    ret, img = cap.read()

    h, w, _ = img.shape

    width = 1000
    height = int(width * (h / w))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    corners, ids, rejected = detector.detectMarkers(img)

    detected_markers, marker_centers = aruco_display(corners, ids, img)
    for val in model_to_map.values():
        cv2.circle(detected_markers, tuple(val), 4, (0, 0, 255), -1)
    cv2.imshow("Image", detected_markers)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


# 2-й этап: выставляем все кубики

cell_centers = cells_initialization(initial_marker)
#
while True:
    ret, img = cap.read()

    h, w, _ = img.shape

    width = 1000
    height = int(width * (h / w))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    corners, ids, rejected = detector.detectMarkers(img)

    detected_markers, marker_centers = aruco_display(corners, ids, img)
    for v in model_to_map.values():
        cv2.circle(detected_markers, v, 4, (0, 0, 255), -1)
    for v in cell_centers.values():
        cv2.circle(detected_markers, v, 4, (0, 255, 0), -1)
    cv2.imshow("Image", detected_markers)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


cv2.destroyAllWindows()
cap.release()



frozen_lake = make_frozen_lake(initial_marker)
print(frozen_lake)
#
# env = gym.make('FrozenLake-v1', desc=frozen_lake, is_slippery=False)
#
# state_space = env.observation_space.n
# print("There are ", state_space, " possible states")
#
# action_space = env.action_space.n
# print("There are ", action_space, " possible actions")
#
#
#
# # Training parameters
# n_training_episodes = 30000  # Total training episodes
# learning_rate = 0.7          # Learning rate
#
# # Evaluation parameters
# n_eval_episodes = 100        # Total number of test episodes
#
# # Environment parameters
# env_id = "FrozenLake-v1"     # Name of the environment
# max_steps = 99               # Max steps per episode
# gamma = 0.95                 # Discounting rate
# eval_seed = []               # The evaluation seed of the environment
#
# # Exploration parameters
# max_epsilon = 1.0             # Exploration probability at start
# min_epsilon = 0.05            # Minimum exploration probability
# decay_rate = 0.0005            # Exponential decay rate for exploration prob
#
# Qtable_frozenlake = get_policy(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps)
#
# file = open("arr", "wb")
# # save array to the file
# np.save(file, Qtable_frozenlake)
# # close the file
# file.close()
#
#
# print(frozen_lake)
# print(Qtable_frozenlake)
#
# if not model_to_map:
#     print("Something's wrong!\n")








