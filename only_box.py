from environmemts.env_acd import Active_vision_env
from learn.dqn_learn import DQN
import os
import random
import cv2
import numpy as np
import torch
from yolov3 import extract_feature
import matplotlib.pyplot as plt

MEMORY_CAPACITY = 500000  # 一起改
MAXIMUN_STEPS = 100
TRAIN_TIMES = 300000
RAMDOM_EXPLORE_TIMES = 200000

os.makedirs("train_record", exist_ok=True)


def run_acd():
    dqn = DQN()

    for episode in range(TRAIN_TIMES):
        reward_list = []
        loss = []
        # initial observation
        if episode % 1 == 0:
            print("episode:", episode, "initial observation....")
        steps = 0
        train_set, img, thing_label, diff, curr_bbox = env.reset(episode)  # observation:
        curr_bbox = list(map(float, curr_bbox))  # turn to tensor
        curr_bbox = torch.FloatTensor(curr_bbox).squeeze()
        while True:
            # RL choose action based on observation
            curr_s = bbox_depth(curr_bbox, img, train_set)

            # choose action
            if dqn.memory_counter <= RAMDOM_EXPLORE_TIMES:
                action = random.randint(0, 6)
            else:
                action = dqn.choose_action(curr_s)
                if type(action) is np.ndarray:
                    action = action[0]

            # RL take action and get next observation and reward
            reward, next_img, next_diff, next_bbox = env.step(train_set, img, thing_label, diff, action, curr_bbox)
            next_s = bbox_depth(next_bbox, next_img, train_set)

            s = curr_s
            s_ = next_s
            r = reward
            a = action

            reward_list.append(str(reward))

            dqn.store_transition(s, a, r, s_)
            if dqn.memory_counter > MEMORY_CAPACITY:
                # print("---------------------Learning-------------------------")
                loss.append(dqn.learn())  # 记忆库满了就进行学习

            if dqn.memory_counter >= MEMORY_CAPACITY + 1:  # 中止驗證
                if stopping_criterion(next_diff, steps):
                    if episode % 100 == 0:
                        filename = os.path.join("train_record", str(episode) + '.txt')
                        with open(filename, 'w') as f:
                            for i in range(len(loss)):
                                f.write(str(loss[i]) + "\n")
                            f.close()
                    break

            steps += 1
            img = next_img
            curr_bbox = next_bbox
            diff = next_diff
        if episode > 150000:
            print("Reward:", reward_list)
        if episode%100000==0:
            print("save the ckp")
            filename = str(episode)+'only_box_dqn.pkl'
            torch.save(dqn, filename)
    print("save the net")
    torch.save(dqn, 'only_box_dqn.pkl')

def stopping_criterion(next_diff, steps):
    if next_diff == 1 or steps >= MAXIMUN_STEPS:
        return True


def bbox_depth(bbox, img, train_set):
    d_img = img[0:-5] + "3.png"
    d_img = os.path.join(train_set, 'high_res_depth', d_img)
    depth_img = cv2.imread(d_img)

    if type(bbox) is torch.Tensor:
        bbox = bbox.numpy()
    w = (bbox[2] - bbox[0]) / 1920
    h = (bbox[3] - bbox[1]) / 1080
    x = (bbox[2] + bbox[0]) / 2
    y = (bbox[3] + bbox[1]) / 2
    d = depth_img[int(y)][int(x)][0] / 255
    x = x / 1920
    y = y / 1080
    bbox = np.array([x, y, w, h, d])
    bbox = torch.from_numpy(bbox).unsqueeze(0)
    bbox = bbox.type(torch.FloatTensor)
    return bbox


if __name__ == '__main__':
    env = Active_vision_env()
    run_acd()
