from environmemts.env_acd import Active_vision_env
from learn.dqn_learn import DQN
import os
import random
import cv2
import numpy as np
import torch
from yolov3 import extract_feature


MEMORY_CAPACITY = 100
os.makedirs("train_record", exist_ok=True)

def run_acd():
    dqn = DQN()

    for episode in range(100):
        action_list = []
        reward_list = []
        # initial observation
        print("episode:", episode, "initial observation....")
        steps = 0
        train_set, img, thing_label, diff, curr_bbox = env.reset(episode) # observation:
        while True:
            # RL choose action based on observation
            # read curr image
            inimg = os.path.join(train_set, 'jpg_rgb', img)
            curr_s = extract_feature.extract_img_feature(inimg)
            # print("curr_s", curr_s.size())
            # choose action
            action = dqn.choose_action(curr_s)

            # RL take action and get next observation and reward
            reward, next_img, next_diff, next_bbox = env.step(train_set, img, thing_label, diff, action)
            # read next image
            inextimg = os.path.join(train_set, 'jpg_rgb', next_img)  # read next img
            next_s = extract_feature.extract_img_feature(inextimg)

            s = curr_s
            s_ = next_s
            r = reward
            a = action

            action_list.append(str(action))
            reward_list.append(str(reward))

            dqn.store_transition(s, a, r, s_)
            # print("counter:", dqn.memory_counter)
            # print('next_diff:', next_diff, 'steps', steps)
            if dqn.memory_counter > MEMORY_CAPACITY:
                # print("start learning...")
                dqn.learn()  # 记忆库满了就进行学习

            if stopping_criterion(next_diff, steps):
                filename = os.path.join("train_record", str(episode)+'.txt')
                with open(filename, 'w') as f:
                    for i in range(len(action_list)):
                        f.write(action_list[i])
                        f.write(reward_list[i]+"\n")
                    f.close()
                print("stop")
                break
            if steps%10 == 0:
                print("steps =", steps)
            print("action:", action, "cur:", os.path.basename(inimg), diff,
                  "next:", os.path.basename(inextimg), next_diff)
            print("Reward:", reward)
            steps += 1
            img = next_img
            diff = next_diff
    print("save the net")
    torch.save(dqn, 'dqn.pkl')


def stopping_criterion(next_diff, steps):
    if next_diff == 1 or steps >= MEMORY_CAPACITY + 100:
        return True


if __name__ == '__main__':
    env = Active_vision_env()
    cuda_gpu = torch.cuda.is_available()
    if (cuda_gpu):
        run_acd()