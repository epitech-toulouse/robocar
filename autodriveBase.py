from control.Manager import Manager
from autodrive.autodriveStateBase import AutoDriveStateBase

if __name__ == "__main__":
    manager = Manager(AutoDriveStateBase())
    try:
        manager.loop()
    except KeyboardInterrupt:
        manager.safe_stop()
