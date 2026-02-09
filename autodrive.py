from control.Manager import Manager
from autodrive.autodrive import AutoDriveState

if __name__ == "__main__":
    manager = Manager(AutoDriveState())
    try:
        manager.loop()
    except KeyboardInterrupt:
        manager.safe_stop()
