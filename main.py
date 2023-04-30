import ai_cameras
import torch

def main():

    cam_manager = ai_cameras.CameraManager()
    cam_manager.create_cam(device_id=R"C:\Users\Vadim\Documents\TestVideos\BB_dde3fde4-9086-4d2f-93c1-583b21d22d10_preview.mp4")
    #cam_manager.create_cam(device_id=0)
    #cam_manager.create_cam(device_id="http://207.255.200.10:50000/SnapshotJPEG?Resolution=640x480&Quality=Clarity&COUNTER")
    cam_manager.run_all()
    print(cam_manager.n_alive_cam_threads())

    #cam = ai_cameras.Camera()
    # cam = ai_cameras.Camera()
    # cam.run()
# # Press the green button in the gutter to run the script.


if __name__ == '__main__':

    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
