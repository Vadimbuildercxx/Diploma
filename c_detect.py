import argparse
import time
from pathlib import Path

import cv2
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
                 imgsz: int = 640, img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'
        # Load model
        self.model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(self.model .stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model .module.names if hasattr(self.model , 'module') else self.model .names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = imgsz
        self.old_img_b = 1

    def pred_pipeline(self, img):
        img, img0 = self.__image_resize(img)
        img = self.__detect_image(img, img0)
        cv2.imshow("image", img)
        cv2.waitKey()
        return img

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
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                self.model(img)[0]

        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img)[0]

        # print(pred.shape)
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        det = pred[0]

        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

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


if __name__ == '__main__':

    detector = YOLODetector("cpu", R"saved_weights\best.pt")
    img_path = R"C:\Users\vadim\Downloads\photo_2023-02-15_15-53-00.jpg"
    img = cv2.imread(img_path)

    detector.pred_pipeline(img)


