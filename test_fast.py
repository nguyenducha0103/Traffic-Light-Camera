
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import StreamingResponse
import uvicorn

import numpy as np
import cv2
import time
from collections import deque

import time
import io


app = FastAPI()


@app.get("/")
async def video_feed():
    return 'Hello world'
        
if __name__ == '__main__':
    
	# start a thread that will perform motion detection
    uvicorn.run(app, host='0.0.0.0', port=2000, access_log=False)
    
