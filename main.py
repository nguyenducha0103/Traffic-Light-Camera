
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import StreamingResponse
import uvicorn

import numpy as np
import cv2
import time
from collections import deque

import time
from PIL import Image
import io

from thread_manager import ThreadManager

app = FastAPI()

queue_frame = deque(maxlen=10)
queue_person = deque(maxlen=10)
queue_restream = deque(maxlen=10)

stopstream_image = np.zeros((1920,1080,3))
_, jpeg_stopimage = cv2.imencode('.jpg', stopstream_image)

queue_camera = deque(maxlen=5)

source = 'rtsp://admin:Vnpt@123@192.168.0.61:551/ch1/main/av_stream'

thread_manager = ThreadManager(source, queue_camera, queue_frame, queue_person, queue_restream)


cameras = {}
camera = thread_manager.cameraread
cameras[0] = camera

thread_manager.start()


@app.get("/{stream_id}")
async def video_feed(stream_id: int):
    if stream_id not in cameras:
        return "Not exist!"
    
    return StreamingResponse(gen(cameras[stream_id]), media_type="multipart/x-mixed-replace;boundary=frame")

@app.get('/stop/{stream_id}')
async def stop_stream(stream_id: int):
    if stream_id in cameras:
        cameras[stream_id].stop()
        cameras[stream_id] = None
        cameras.pop(stream_id)

        return "stopped"
    
    return "Not exist!"


def gen(camera):
    """Video streaming generator function."""
    print(camera)
    while True:
        # t1 = time.time()
        if len(queue_frame):
            frame = queue_frame.popleft()
            if frame.shape == ():
                jpeg = jpeg_stopimage
            else:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # _, jpeg = cv2.imencode('.jpg', frame)
                buffer = io.BytesIO()
                frame.save(buffer, format="JPEG")
                # Check first few bytes
                jpeg = buffer.getvalue()
                
        else:
            time.sleep(0.001)
            continue
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        
if __name__ == '__main__':
    
	# start a thread that will perform motion detection
    uvicorn.run(app, host='0.0.0.0', port=5689, access_log=False)
    
