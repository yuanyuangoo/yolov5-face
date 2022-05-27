# -*- coding: UTF-8 -*-
import os
from matplotlib import axes
import torchvision
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy

from models.experimental import attempt_load
from utils.datasets import letterbox
from additional import apply_classifier, show_results
from utils.general import check_img_size, non_max_suppression_face, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords






def detect_one(model, image_path, device):
    # Load model
    img_size = 1280
    conf_thres = 0.2
    iou_thres = 0.5

    mask_model = torchvision.models.resnet18(pretrained=False)
    num_ftrs = mask_model.fc.in_features
    num_classes = 2
    mask_model.fc = torch.nn.Linear(num_ftrs, num_classes)
    mask_model.load_state_dict(torch.load('../pytorch-image-classification/mask_cls.pt'))
    mask_model.eval()
    mask_model = mask_model.to(device)


    age_model = torchvision.models.resnet50(pretrained=False)
    num_ftrs = age_model.fc.in_features
    num_classes = 19
    age_model.fc = torch.nn.Linear(num_ftrs, num_classes)
    age_model.load_state_dict(torch.load('../pytorch-image-classification/age_4.pt', map_location=device))
    age_model.eval()
    age_model = age_model.to(device)

    orgimg = cv2.imread(image_path)  # BGR
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' + image_path
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
    # print(img0[-200,-200,:], img0.shape)

    img = letterbox(img0, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    # print(img[:,:,-200,-200])
    # exit()
    pred = model(img)[0]
    # print(pred)

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)
    # print(pred)
    # exit()
    names = ['mask', 'nomask']
    # print(len(pred[0]))
    # exit()
    pred = apply_classifier(pred, mask_model, age_model, img, img0)

    print('img.shape: ', img.shape)
    print('orgimg.shape: ', orgimg.shape)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()

            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                # cls = det[j, 5].cpu().numpy()
                landmarks = det[j, 5:15].view(-1).tolist()
                class_num = int(det[j, 15].cpu().numpy().item())
                age_num = int(det[j, 16].cpu().numpy().item())

                orgimg = show_results(orgimg, xyxy, conf,landmarks, class_num, age_num)

    cv2.imwrite('result.jpg', orgimg)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./yolov5m-face.pt', help='model.pt path(s)')
    parser.add_argument('--image', type=str, default='./_113298561_a6a17a72-013c-4d04-ae38-542e6f78ea44.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.weights, device)
    detect_one(model, opt.image, device)
