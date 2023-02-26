import ai_cameras
import torch

def main():

    cam_manager = ai_cameras.CameraManager()
    cam_manager.create_cam(device_id=R"C:\Users\vadim\AI\YOLOV\testData\test_videos\American Psycho - Business Card scene [HD - 720p] (online-video-cutter.com)_v.mp4")
    cam_manager.create_cam(device_id=R"C:\Users\vadim\AI\YOLOV\testData\test_videos\Видеонаблюдение на предприятии - рабочая зона.mp4")
    # # cam_manager.create_cam(device_id=2)
    cam_manager.run_all()
    print(cam_manager.n_alive_cam_threads())

    #cam = ai_cameras.Camera()
    # cam = ai_cameras.Camera()
    # cam.run()
# # Press the green button in the gutter to run the script.


if __name__ == '__main__':

    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
