import math
import random

import numpy as np
import time
import cv2

cell_size = -1


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


class World:
    def __init__(self, m, n, start=0, goal=-1):
        if goal == -1:
            goal = m * n - 1 - 2 * m - 2
        self.goal = goal
        self.grid = np.zeros(
            (m, n))  # на данный момент бесполезно. существует на случай, если мы захотим препятствия отмечать на карте
        self.m = m
        self.n = n
        self.angle = 0
        self.state_space_plus = [i for i in range(n * m)]
        self.action_space = {'U': -self.m, 'D': self.m, 'L': -1, 'R': 1}
        self.possible_actions = np.array(['U', 'D', 'L', 'R'])
        self.agent_pos = start

    def get_coords(self):
        return self.agent_pos // self.m, self.agent_pos % self.n

    def set_state(self, new_state, new_angle):
        x, y = self.get_coords()
        self.grid[x][y] = 0
        self.angle = new_angle
        self.agent_pos = new_state
        x, y = self.get_coords()
        self.grid[x][y] = 1

    def move_is_offgrid(self, new_state, old_state):
        if new_state not in self.state_space_plus:
            return True
        elif old_state % self.m == 0 and new_state % self.m == self.m - 1:
            return True
        elif old_state % self.m == self.m - 1 and new_state % self.m == 0:
            return True
        elif old_state % self.n == 0 and new_state % self.n == self.n - 1:
            return True
        elif old_state % self.n == self.n - 1 and new_state % self.n == 0:
            return True
        return False

    def step(self, action):
        result_state = self.agent_pos + self.action_space[action]
        angle = env.angle
        # 'переворачиваемся', если мы не под нужным углом
        if (angle == 0 or angle == 180) and action == 'U':
            angle = 270
        elif (angle == 0 or angle == 180) and action == 'D':
            angle = 90
        elif (angle == 270 or angle == 90) and action == 'L':
            angle = 180
        elif (angle == 270 or angle == 90) and action == 'R':
            angle = 0
        if result_state != self.goal:
            reward = -1
        else:
            reward = 20
        if self.move_is_offgrid(result_state, self.agent_pos):
            return self.agent_pos, reward, (result_state == self.goal)
        if bumped(result_state, angle):
            reward = -20
        else:
            self.set_state(result_state, angle)
        return result_state, reward, (result_state == self.goal)

    def reset(self):
        self.agent_pos = 0
        self.grid = np.zeros((self.n, self.m))
        return self.agent_pos

    def action_space_sample(self):
        return np.random.choice(self.possible_actions)


def max_action(Q, state, actions):
    values = np.array([Q[state, a] for a in actions])
    action = np.argmax(values)
    return actions[action]


def get_len(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def aruco_display(corners, ids, rejected, image):
    model_to_map = {}
    cell_size = -1
    if len(corners) > 0:

        ids = ids.flatten()
        marker_corner = corners[0]
        marker_corner = marker_corner.reshape((4, 2))
        (top_left, top_right, bottom_right, bottom_left) = marker_corner
        cell_size = get_len(top_right, bottom_right) / 4

        for i in range(int(height // cell_size)):
            for j in range(int(width // cell_size)):
                dotx = int((cell_size * j) + (cell_size * (j + 1)) // 2)
                doty = int((cell_size * i) + (cell_size * (i + 1)) // 2)
                model_to_map[i * (width // cell_size) + j] = (dotx, doty)
                cv2.circle(image, (dotx, doty), 4, (0, 0, 255), -1)
        cv2.circle(image, (0, 0), 4, (0, 0, 255), -1)
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

            cv2.putText(image, str(markerID), (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    return image, model_to_map, width // cell_size, height // cell_size


arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

arucoParams = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

cap = cv2.VideoCapture(2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, img = cap.read()

    h, w, _ = img.shape

    width = 1000
    height = int(width * (h / w))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    corners, ids, rejected = detector.detectMarkers(img)

    detected_markers, model_to_map, map_width, map_height = aruco_display(corners, ids, rejected, img)

    cv2.imshow("Image", detected_markers)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

if not model_to_map:
    print("Something's wrong!\n")

env = World(int(map_width), int(map_height))
lr = 0.1
gamma = 1
eps = 1
Q = dict()
for state in env.state_space_plus:
    for action in env.possible_actions:
        Q[state, action] = 0

num_episodes = 50000
total_rewards = np.zeros(num_episodes)
for i in range(num_episodes):
    observation = env.reset()
    done = False
    ep_reward = 0
    while not done:
        rand = random.uniform(0, 1)
        action = max_action(Q, observation, env.possible_actions) if rand < (1 - eps) else env.action_space_sample()
        new_observation, reward, done = env.step(action)
        ep_reward += reward
        greedy_action = max_action(Q, new_observation, env.possible_actions)
        Q[observation, action] = Q[observation, action] + lr * (
                reward + gamma * Q[new_observation, greedy_action] - Q[observation, action])
        observation = new_observation
    if eps - (2 / num_episodes) > 0:
        eps -= 2 / num_episodes
    else:
        eps = 0.05
    total_rewards[i] = ep_reward

while True:
    cv2.imshow("Image", detected_markers)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# observation = env.reset()
# done = False
# while not done:
#     action = max_action(Q, observation, env.possible_actions)
#     new_observation, reward, done = env.step(action)
#     observation = new_observation
#     # реальное движение происходит где-то здесь


cv2.destroyAllWindows()
cap.release()
