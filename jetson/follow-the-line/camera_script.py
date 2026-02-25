import cv2
import depthai as dai
import os

device = None
videoQueue = None

def init_camera():
    global device, videoQueue
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setPreviewSize(128, 128)
    cam.setInterleaved(False)
    cam.setFps(12)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("video")
    cam.preview.link(xout.input)

    if not os.path.exists("images/"):
        os.makedirs("images/")

    device = dai.Device(pipeline)
    videoQueue = device.getOutputQueue(name="video", maxSize=4, blocking=True)

def get_image_array():
    frame = None
    while videoQueue.has() or frame is None:
        videoIn = videoQueue.get()
        frame = videoIn.getCvFrame()

    if frame is None:
        return None

    frame = cv2.resize(frame, (128, 128))
    return frame
