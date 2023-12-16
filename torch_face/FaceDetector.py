import cv2
import numpy as np
from onnx_face.onnx_runtime import ONNXBase
from math import ceil
from itertools import product as product
import time


def draw(img, bbox, lmk):
    for b, l in zip(bbox, lmk):
        box = [int(x) for x in b[:4]]
        
        cv2.rectangle(img,(box[0], box[1]), (box[2], box[3]), (255,255,0),1 )
        
        for p in l:
            p = [int(x) for x in p]
            
            cv2.circle(img, (p[0], p[1]), 1, (0, 0, 255), 4)
            cv2.circle(img, (p[0], p[1]), 1, (0, 0, 255), 4)
            cv2.circle(img, (p[0], p[1]), 1, (0, 0, 255), 4)
            cv2.circle(img, (p[0], p[1]), 1, (0, 0, 255), 4)
            cv2.circle(img, (p[0], p[1]), 1, (0, 0, 255), 4)
        
    cv2.imwrite('test_drawed.jpg', img)

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

class PriorBox(object):
    def __init__(self, image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.clip = False
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = np.array(anchors).reshape(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), axis=1)
    return landms


class FaceDetector(ONNXBase):
    def __init__(self, weight_path, input_shape=(640, 640)):
        super().__init__(weight_path)

        self.input_shape = input_shape

        self.priors = PriorBox(input_shape).forward()

    def preprocess(self, image):
        input_width, input_height = self.input_shape
        image = cv2.resize(image, (input_width, input_height))
        image = np.float32(image)
        image -= (123, 117, 104)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image
    
    def postprocess(self, result, img_shape, confidence_threshold = 0.5, nms_threshold = 0.45):
        
        loc, conf, landms = result
        
        im_height, im_width = img_shape
        top_k = 50
        keep_top_k = 40
        
        variance = [0.1, 0.2]
        scale = np.array([im_width, im_height, im_width, im_height])

        
        boxes = decode(np.array(loc).squeeze(0), self.priors, variance)
        
        boxes = boxes * scale
        # boxes = boxes.detach().cpu().numpy()
        scores = np.array(conf).squeeze(0)[:, 1]
        landms = decode_landm(np.array(landms).squeeze(0), self.priors, variance)
        
        scale1 = np.array([im_width, im_height, im_width, im_height,
                                im_width, im_height, im_width, im_height,
                                im_width, im_height])
        landms = landms * scale1

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        # t2 = time.time()
        keep = py_cpu_nms(dets, nms_threshold)
        # print('time nms', time.time() - t2)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS

        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        bbox_keep, landmark = [], []
        
        for det,landm in zip(dets,landms):
            
            bbox_keep.append(det)
            landmark.append(np.array(landm).reshape(5,2).astype(np.int16))
            
        return np.array(bbox_keep), np.array(landmark)
    
    def detect(self, image):
        img_shape = image.shape[:2]
        # preprocess image from numpy array
        image = self.preprocess(image)
        result = self.infer(image)

        bboxes, lmks = self.postprocess(result, img_shape=img_shape)
        return bboxes, lmks
    
