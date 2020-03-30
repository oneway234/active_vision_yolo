from __future__ import division

from yolov3.models import *
from yolov3.utils.utils import *
from yolov3.utils.datasets import *
import cv2
import os
import sys
import time
import datetime
import argparse

import torch
from torch.autograd import Variable


ckp = "/home/wei/active vision/PyTorch-YOLOv3/checkpoints329/yolov3_ckpt_55.pth" #weight
def extract_img_feature(img_name):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="yolov3/config/yolov3-custom.cfg",
                        help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default=ckp, help="path to weights file")
    parser.add_argument("--class_path", type=str, default="yolov3/data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, default=ckp, help="path to checkpoint model")
    opt = parser.parse_args()
    # print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #loas darknet model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    # Set in evaluation mode
    model.eval()

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    input_imgs = read_img(img_name)
    input_imgs = Variable(input_imgs.type(Tensor))
    with torch.no_grad():
        x, detections = model(input_imgs)
        x = detections
            # x = torch.unsqueeze(torch.FloatTensor(detections), 0)
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
    return x.cpu()

def read_img(img):
    inimg = cv2.imread(img)
    inimg = cv2.resize(inimg, (416, 416))
    inimg = np.transpose(inimg, (2, 0, 1))
    inimg = torch.unsqueeze(torch.FloatTensor(inimg), 0)
    return inimg

if __name__ == '__main__':
    extract_img_feature()

