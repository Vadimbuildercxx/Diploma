import cv2
import threading
from yolov7.c_detect import YOLODetector


caps = [0, 200, 300, 500, 600, 700, 800, 900, 910, 1000, 1100, 1200, 1300,
        1400, 1410, 1500, 1600, 1610, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400]


class CameraManager(object):

    def __init__(self):
        self.cameras: list[Camera] = []

    def create_cam(self, device_id: int, name=None):
        cam = Camera(device_id, name)
        self.cameras.append(cam)

    def run_all(self):
        for cam in self.cameras:
            cam.run()


class Camera(object):

    def __init__(self, device_id: int, name=None, camera_id: int = -1):
        self.camera_id = camera_id
        self.device_id = device_id
        if name is None:
            self.name = "Camera " + str(device_id)
        else:
            self.name = name

        self.thread = CamThread(self.name, self.device_id)

    def run(self):
        self.thread.run()


class CamThread(object):
    def __init__(self, cam_name, device_id):
        threading.Thread.__init__(self)
        self.detector = YOLODetector("cpu", R"saved_weights\best.pt")
        self.cam_name = cam_name
        self.cam_id = device_id

    def run(self):
        print("Starting cam: " + self.cam_name)
        threading.Thread(target=self.cam_thread, name=self.cam_name).start()

    def cam_thread(self):
        print(f"Cam thread on {threading.current_thread().name} is running... \n")
        cv2.namedWindow(self.cam_name)

        cam_id = self.cam_id + cv2.CAP_DSHOW
        cam = cv2.VideoCapture(cam_id)
        if cam.isOpened():  # try to get the first frame
            r_val, frame = cam.read()
        else:
            r_val = False
            print(f"!!! Cannot open cam thread on {threading.current_thread().name} ... \n")
        while r_val:
            frame = self.detector.pred_pipeline(frame)
            cv2.imshow(self.cam_name, frame)
            r_val, frame = cam.read()
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
        cv2.destroyWindow(self.cam_name)


