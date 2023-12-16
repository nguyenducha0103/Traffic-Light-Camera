
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import StreamingResponse
import uvicorn
import argparse
# from camera_service.cameraip.v1_0.thread_manager import ThreadControler
# from camera_service.webcam.v1_0.thread_manager import ThreadControler
import numpy as np
import cv2
import time 

app = FastAPI()
from camera_read import CameraReading
# import cv2
import sys
# appname = "live"
# restream_url = {
#     "rtmp": f"rtmp://10.70.39.204:1935/{appname}/webcam",
#     "flv": f"http://10.70.39.204:7001/{appname}/webcam.flv",
#     "hls": f"http://10.70.39.204:7002/{appname}/webcam.m3u8",
# }
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", default='rtsp://admin:Vnpt@123@123.25.190.36:553/ch1/main/av_stream',
    help="camera source")
ap.add_argument("-i", "--ip", type=str, default="0.0.0.0",
    help="ip address of the device")
ap.add_argument("-o", "--port", type=int, default=2000,
    help="ephemeral port number of the server (1024 to 65535)")

args = vars(ap.parse_args())

source = args['source']
stopstream_image = np.zeros((100,100))
_, jpeg_stopimage = cv2.imencode('.jpg',stopstream_image)
queue_camera = {
    'frame': None,
    'source': source
}

camera = CameraReading(queue_camera)
camera.daemon = True
camera.start()


@app.get("/")
async def video_feed():
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace;boundary=frame")


def gen():
    """Video streaming generator function."""
    while True:
        # time.sleep(0.01)
        frame = queue_camera['frame']
        if frame.shape == ():
            jpeg = jpeg_stopimage
        else:
            _, jpeg = cv2.imencode('.jpg', frame)
            
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
host_ip = str("http://") + str(s.getsockname()[0])+":"+str(args["port"])+"/"
s.close()

print(f'Uvicorn running on {host_ip} (Press CTRL+C to quit)')
if __name__ == '__main__':
    
	# start a thread that will perform motion detection
    uvicorn.run(app, host=args["ip"], port=args["port"], access_log=False)