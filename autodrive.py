from control.Manager import Manager
from autodrive.autodriveState import AutoDriveState

if __name__ == "__main__":
    manager = Manager(AutoDriveState)

    manager.loop()
