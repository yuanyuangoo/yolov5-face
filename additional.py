import os
import cv2
import numpy as np
import torch

from utils.general import xyxy2xywh, xywh2xyxy, scale_coords


def apply_classifier(x, mask_model, age_model, img, im0):
    # Apply a second stage classifier to YOLO outputs
    # Example model = torchvision.models.__dict__['efficientnet_b0'](pretrained=True).to(device).eval()
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()
            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                im = np.ascontiguousarray(
                    im, dtype=np.float32)  # uint8 to float32
                # cv2.imwrite('example%i.jpg' % j, cutout)
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                im = (im - pretrained_means) / pretrained_stds

                # BGR to RGB, to 3x416x416
                im = im[:, :, ::-1].transpose(2, 0, 1)
                ims.append(im)
            ims = np.asarray(ims)
            mask_scores = mask_model(torch.Tensor(ims).to(d.device))

            mask_cls = mask_scores.argmax(1)  # mask classifier prediction
            age_scores = age_model(torch.Tensor(ims).to(d.device))  # age classifier prediction

            #mid value of each group
            mid = torch.tensor([1.5, 24.5, 18.5, 23, 28, 4, 33, 38, 43,
                                48, 53, 58, 7, 63, 68, 73, 78, 85, 10.5])
            mid = torch.unsqueeze(mid, dim=1).to(d.device)
            normalized_age_score = torch.softmax(age_scores, dim=1)
            # age = age_scores.argmax(1)
            #age weigted by age_scores
            age = torch.matmul(normalized_age_score, mid)

            x[i][:, 15] = mask_cls
            x[i] = torch.cat((x[i], age), 1).detach()
    return x


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
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


def show_results(img, xyxy, conf, landmarks, class_num, age_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])

    # clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
    center = (x1/2+x2/2, (y1+y2)/2)


    mask_group=os.listdir('../test_data/ML_test_DataSet/Face_Mask_Dataset/Test/')
    mask_group.sort()
    mask_group=['with_mask', 'without_mask']

    if 'out' in mask_group[class_num]:
        label = str(conf)[:5]+'_'+str(age_num)
        
        cv2.ellipse(img,
                    (center,
                    (x2-x1, y2-y1),
                     0),
                    color=(0, 255, 0),
                    thickness=2)
    else:
        label = str(conf)[:5]+'_'+'Mask'
        cv2.rectangle(img, (x1,y1), (x2, y2), (0,0,255), thickness=tl, lineType=cv2.LINE_AA)


    tf = max(tl - 1, 1)  # font thickness


    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img