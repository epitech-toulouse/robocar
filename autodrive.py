from control.Manager import Manager
from autodrive.autodriveState import AutoDriveState

if __name__ == "__main__":
    try:
        manager = Manager(AutoDriveState)
        manager.loop()
    except KeyboardInterrupt:
        manager.safe_stop()
