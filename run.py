import time
import numpy as np
import os
import glob

for img_path in glob.glob('./person_list/*jpg'):
    name = os.path.basename(img_path)
    name_img = name.split('.jpg')[0]

    os.mkdir(os.path.join('./person_list/',name_img))
    os.rename(img_path, os.path.join('./person_list/', name_img, name ))