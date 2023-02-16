import ai_cameras


def main():

    cam_manager = ai_cameras.CameraManager()
    cam_manager.create_cam(0)
    # cam_manager.create_cam(1)
    # cam_manager.create_cam(2)
    cam_manager.run_all()

# # Press the green button in the gutter to run the script.


if __name__ == '__main__':
    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
