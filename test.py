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
    train_set, img, thing_label, diff, curr_bbox = env.reset(0)
    inimg = os.path.join(train_set, 'jpg_rgb', img)
    curr_s, detections = extract_feature.extract_img_feature(inimg)
    print("curr_s", curr_s)
    print("curr_s shape", (np.array(curr_s)).shape)

def stopping_criterion(next_diff, steps):
    if next_diff == 1 or steps >= MAXIMUN_STEPS:
        return True


if __name__ == '__main__':
    env = Active_vision_env()

    run_acd()
