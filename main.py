from environmemts.env_acd import Active_vision_env
from learn.dqn_learn import DQN
import os
import random
import cv2
import numpy as np
import torch
from yolov3 import extract_feature
import matplotlib.pyplot as plt


MEMORY_CAPACITY = 15
MAXIMUN_STEPS = 10
TRAIN_TIMES = 3
RAMDOM_EXPLORE_TIMES = 5

os.makedirs("train_record", exist_ok=True)

def run_acd():
    dqn = DQN()

    for episode in range(TRAIN_TIMES):
        action_list = []
        reward_list = []
        loss = []
        # initial observation
        print("episode:", episode, "initial observation....")
        steps = 0
        train_set, img, thing_label, diff, curr_bbox = env.reset(episode) # observation:
        curr_bbox = list(map(float, curr_bbox))               #turn to tensor
        curr_bbox = torch.FloatTensor(curr_bbox).unsqueeze(0)
        while True:
            # RL choose action based on observation
            # read curr image
            inimg = os.path.join(train_set, 'jpg_rgb', img)
            curr_s = extract_feature.extract_img_feature(inimg)
            curr_s = curr_s.reshape(1, 8112*39)
            curr_s = dqn.decrease_net(curr_s)
            curr_s = torch.cat([curr_s, curr_bbox], 1) #combine feature and bbox
            # choose action
            if dqn.memory_counter <= RAMDOM_EXPLORE_TIMES:
                action = random.randint(0, 6)
            else:
                action = dqn.choose_action(curr_s)
                if type(action) is np.ndarray:
                    action = action[0]
            # action = dqn.choose_action(curr_s) # action value check
            # if type(action) is np.ndarray:
            #     action = action[0]

            # RL take action and get next observation and reward
            reward, next_img, next_diff, next_bbox = env.step(train_set, img, thing_label, diff, action, curr_bbox)
            if type(next_bbox) is not torch.Tensor:
                next_bbox = list(map(float, next_bbox))  # turn to tensor
                next_bbox = torch.FloatTensor(next_bbox).unsqueeze(0)  # turn to tensor


            # read next image
            inextimg = os.path.join(train_set, 'jpg_rgb', next_img)  # read next img
            next_s = extract_feature.extract_img_feature(inextimg)
            next_s = next_s.reshape(1, 8112 * 39)
            next_s = dqn.decrease_net(next_s)
            next_s = torch.cat([next_s, next_bbox], 1)

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
                # print("---------------------Learning-------------------------")
                loss.append(dqn.learn().detach().numpy())  # 记忆库满了就进行学习


            if dqn.memory_counter >= MEMORY_CAPACITY+1: #中止驗證
                if stopping_criterion(next_diff, steps):
                    filename = os.path.join("train_record", str(episode) + '.txt')
                    with open(filename, 'w') as f:
                        for i in range(len(action_list)):
                            f.write(action_list[i])
                            f.write(", ")
                            f.write(reward_list[i] + "\n")
                        f.close()

                    # print(loss)
                    # plt.plot(loss)
                    # plt.ylabel('Loss')
                    # plt.xlabel('Times')
                    # img_name = os.path.join("train_record", str(episode) + '.png')
                    # plt.savefig(img_name)
                    # print("stop")
                    break

            if steps%100 == 0:
                print("---------------------steps:", steps, "-----------------------")
            print("steps =", steps, "action:", action, "Reward:", reward, "next_diff:", next_diff)
            steps += 1
            img = next_img
            curr_bbox = next_bbox
            diff = next_diff
    print("save the net")
    torch.save(dqn, 'dqn.pkl')


def stopping_criterion(next_diff, steps):
    if next_diff == 1 or steps >= MAXIMUN_STEPS:
        return True


if __name__ == '__main__':
    env = Active_vision_env()
    cuda_gpu = torch.cuda.is_available()
    if (cuda_gpu):
        run_acd()
