import random

import cv2
import threading
import torch
import numpy as np

import face_embeddings
from c_detect import YOLODetector
#import face_embeddings
import os
from PIL import Image

caps = [0, 200, 300, 500, 600, 700, 800, 900, 910, 1000, 1100, 1200, 1300,
        1400, 1410, 1500, 1600, 1610, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400]

img_me = Image.open(R"C:\Users\vadim\AI\YOLOV\testData\test_images\bateman\bateman_face\bateman_fun.jpg")

class CameraManager(object):

    def __init__(self):
        self.cameras: list[Camera] = []

    def create_cam(self, device_id, name=None):
        cam = Camera(device_id, name)
        self.cameras.append(cam)

    def run_all(self):
        [cam.start() for cam in self.cameras]

    def n_alive_cam_threads(self) -> int:
        return sum([cam.is_alive() for cam in self.cameras])


class Camera(threading.Thread):
    def __init__(self, source, name: str = None, detect_faces: bool = True) -> None:
        """
        Camera thread realise thread from cam or video source
        :param source - id of cam device or video source
        :param name - name of camera. Automatically generated when None (Camera+id)
        """
        threading.Thread.__init__(self)
        assert isinstance(source, str) or isinstance(source, int), "Wrong source type!"
        self.source = source

        if name is None:
            self.name = "Camera " + str(source[:-5] if isinstance(source, str) else source)
        else:
            self.name = name

        self.detector = YOLODetector('cuda', R"saved_weights\best.pt", img_size=1280)
        self._is_running = False
        self.__detect_faces = detect_faces
        if self.__detect_faces:
            self.face_similarity = face_embeddings.FaceComparator(dataset_path="person_faces")

    def extract_image_area(self, image_in: np.ndarray, detections):
        image = image_in.copy()
        # print("extract_image_area func ", image.shape)
        results = []
        for *xyxy, conf, cls in reversed(detections):
            x1, y1, x2, y2 = xyxy
            x1, y1, x2, y2 = x1.type(torch.int).item(), y1.type(torch.int).item(), \
                                    x2.type(torch.int).item(), y2.type(torch.int).item()

            area = image[y1:y2, x1:x2]
            if (area.shape[0] > 0) and (area.shape[0] > 0):
                similarity_score, path = self.face_similarity.find_face(area)
                _dict = {"face_score": similarity_score, "face_path": path,
                        "coords": [x1, y1, x2, y2], "conf": conf, "cls": cls}

            results.append(_dict)

        return results

    @property
    def is_running(self) -> bool:
        return self._is_running

    @is_running.setter
    def is_running(self, value: bool) -> None:
        assert isinstance(value, bool), 'Argument of wrong type!'
        self._is_running = value

    def run(self) -> None:
        """
        Extend threading. Thread run method. Start video capture from device id: cam_id
        :return: None
        """
        print(f"Cam thread on {threading.current_thread().name} is running... \n")
        cv2.namedWindow(self.name)
        self.is_running = True

        if isinstance(self.source, int):
            cam_source = self.source + cv2.CAP_DSHOW
        elif isinstance(self.source, str):
            cam_source = self.source

        cam = cv2.VideoCapture(cam_source)
        if cam.isOpened():  # try to get the first frame
            r_val, frame = cam.read()
        else:
            r_val = False
            print(f"!!! Cannot open cam thread on {threading.current_thread().name} ... \n")
        while r_val:
            if self.__detect_faces:
                detections = self.detector.pred_pipeline_bboxes(frame)
                face_preds = self.extract_image_area(frame, detections)
                for pred in face_preds:
                    _face_score, _path, _xyxy, _conf, _cls = (pred["face_score"], pred["face_path"],
                                                                    pred["coords"], pred["conf"], pred["cls"])
                    _person_name = _path.split('/')[-1].split('.')[0]
                    self.detector.plot_one_box(_xyxy, float(_conf.item()),
                                               _cls, float(_face_score.item()),
                                               _person_name, frame)
            else:
                orig, frame, detections = self.detector.pred_pipeline_detected_and_bboxes(frame)


            cv2.imshow(self.name, frame)
            r_val, frame = cam.read()
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
        self.is_running = False
        cv2.destroyWindow(self.name)






