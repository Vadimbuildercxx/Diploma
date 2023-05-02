import ai_cameras
import torch
from config_loader import JsonReader


def main():
    loader = JsonReader('../Diploma/Diploma/Files/startup_config.json')
    json = loader.read()
    cam_manager = ai_cameras.CameraManager()

    for path in json["Sources"]:
        cam_manager.create_cam(path, json["Weights"], name=None, detect_faces=json["IsDetectFaces"],
                               send_time_sec=json["SendEveryXsec"], dataset_path="person_faces",
                               img_size=json["ImageSize"], stride=json["Stride"])

    cam_manager.run_all()
    print(cam_manager.n_alive_cam_threads())

    # cam = ai_cameras.Camera()
    # cam = ai_cameras.Camera()
    # cam.run()


# # Press the green button in the gutter to run the script.


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
