import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random, ascontiguousarray

from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


class YOLODetector(object):

    def __init__(self, device: str, weights: str = "yolov7.pt",
                 img_size=640, stride=32): # imgsz: int = 640,
        print("cuda is available" if torch.cuda.is_available() else "only cpu detected")
        self.img_size = img_size
        self.stride = stride
        self.device = torch.device(device) # select_device(device)
        self.half = self.device.type != 'cpu'
        # Load model
        # weights = R"saved_weights\epoch_000.pt"
        self.model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(self.model .stride.max())  # model stride
        imgsz = check_img_size(img_size, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model .names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = imgsz
        self.old_img_b = 1

    def pred_pipeline_detected_and_bboxes(self, img_in: np.ndarray):
        """
        Pipeline to predict objects in image
        :param img_in:
        :return: orig. image, image with bounding boxes, detections (bbox, conf, class)
        """
        img = img_in.copy()
        img, img0 = self.__image_resize(img)
        img, det = self.__detect_image(img, img0)
        img_out = self.draw_bbox(img, det)
        return img_in, img_out, det

    def pred_pipeline_bboxes(self, img: np.ndarray):
        img, img0 = self.__image_resize(img)
        _, det = self.__detect_image(img, img0)
        return det

    def __detect_image(self, img, im0, conf_thres: int = 0.3, iou_thres=0.45):
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Warmup
        if self.device.type != 'cpu' and (
                self.old_img_b != img.shape[0] or
                self.old_img_h != img.shape[2] or
                self.old_img_w != img.shape[3]):

            for i in range(3):
                self.model(img)[0]

        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        det = pred[0]

        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        return im0, det

    def __t_detect_image(self, img, im0, conf_thres: int = 0.2, iou_thres=0.45):
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Warmup
        if self.device.type != 'cpu' and (
                self.old_img_b != img.shape[0] or
                self.old_img_h != img.shape[2] or
                self.old_img_w != img.shape[3]):

            for i in range(3):
                self.model(img)[0]

        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        det = pred[0]

        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        im0 = self.draw_bbox(im0, det)

        return im0



    def draw_bbox(self, im0, det):
        # Write results
        for *xyxy, conf, cls in reversed(det):
            label = f'{self.names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)
        return im0


    def __image_resize(self, img0):

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = ascontiguousarray(img)
        return img, img0

    def plot_one_box(self, xyxy: list, conf, cls_indx,
                     face_conf, face_name, img,
                     color=None, label=None, line_thickness=1, thresh: float = 0.3) -> None:
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = self.colors[int(cls_indx)]
        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if True:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

            _c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, _c2, color, -1, cv2.LINE_AA)  # filled

            # class confidence rectangle
            cс_rec_c1 = c1
            cс_rec_c2 = int(c1[0] + conf * (c2[0] - c1[0])), c1[1] - 10
            cv2.rectangle(img, cс_rec_c1, cс_rec_c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, self.names[int(cls_indx)], (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        if face_conf > thresh:
            # face confidence rectangle
            _w = c2[0] - c1[0]
            _h = 10
            fс_rec_c1 = (c1[0] + int(_w/2), c2[1])
            fс_rec_c2 = (int(c1[0] + _w/2 + face_conf * _w), c2[1] + _h)
            cv2.rectangle(img, fс_rec_c1, fс_rec_c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, face_name + " " + str(face_conf), (c1[0], c2[1] + 10), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


