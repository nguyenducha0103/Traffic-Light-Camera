import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import numpy as np
import sys

import cv2

def preprocess(img, input_shape, letter_box=False):
    """Preprocess an image before TRT YOLO inferencing.
    # Args
        img: int8 numpy array of shape (img_h, img_w, 3)
        input_shape: a tuple of (H, W)
        letter_box: boolean, specifies whether to keep aspect ratio and
                    create a "letterboxed" image for inference
    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    if letter_box:
        img_h, img_w, _ = img.shape
        new_h, new_w = input_shape[0], input_shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
            offset_h = (input_shape[0] - new_h) // 2
        else:
            new_w = int(img_w * new_h / img_h)
            offset_w = (input_shape[1] - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    else:
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    
    return img

def _nms_boxes(detections, nms_threshold):
    """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding
    boxes with their confidence scores and return an array with the
    indexes of the bounding boxes we want to keep.
    # Args
        detections: Nx7 numpy arrays of
                    [[x, y, w, h, box_confidence, class_id, class_prob],
                     ......]
    """
    x_coord = detections[:, 0]
    y_coord = detections[:, 1]
    width = detections[:, 2]
    height = detections[:, 3]
    box_confidences = detections[:, 4] * detections[:, 6]

    areas = width * height
    ordered = box_confidences.argsort()[::-1]

    keep = list()
    while ordered.size > 0:
        # Index of the current element:
        i = ordered[0]
        keep.append(i)
        xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
        yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
        xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
        yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)
        iou = intersection / union
        indexes = np.where(iou <= nms_threshold)[0]
        ordered = ordered[indexes + 1]

    keep = np.array(keep)
    return keep

def convert_points(boxes, old_size, new_size):
    points = boxes.astype(float)

    old_h, old_w = old_size
    new_h, new_w = new_size

    r_w, r_h = 1.0 / old_w * new_w, 1.0 / old_h * new_h
    points[:, [0, 2]] *= r_w
    points[:, [1, 3]] *= r_h
    
    return points.astype(int)


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

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

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def postprocess(predictions, img_w, img_h, input_shape, nms_thr = 0.45, score_thr = 0.25):
    """Postprocess for output from YOLOv7"""
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.

    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr)
    print('dets',dets)
    if dets is not None:
        boxes = convert_points(dets[:,:4], input_shape, (img_h, img_w))
        scores = dets[:,4]
        labels = dets[:,5]

        detected_objects = []
        for box, score, label in zip(boxes, scores, labels):
            box = box.astype(np.int16)
            detected_objects.append([box[0],box[1],box[2],box[3],score, int(label)])
        return np.array(detected_objects)
    else:
        return np.empty((0,6))

def processing_bbox_outside(bbox, img_w, img_h):
    x1,y1,x2,y2 = [int (i) for i in bbox]
    
    x1,y1,x2,y2 = \
        x1 if x1 >= 0 else 0,\
            y1 if y1 >= 0 else 0,\
            x2 if x2 <= img_w else img_w,\
            y2 if y2 <= img_h else img_h
    bbox = [x1,y1,x2,y2]
    return bbox


class TritonVehicle(object):
    def __init__(self, model_name = 'yolov7_vehicle'):
        self.url = '10.70.39.40:8001'
        self.model = model_name
        self.width = 640
        self.height = 640
        self.inputs = []
        self.outputs = []
        self.inputs.append(grpcclient.InferInput('images', [1, 3, self.height, self.width], "FP32"))
        self.outputs.append(grpcclient.InferRequestedOutput('output'))
        
        try:
            self.triton_client = grpcclient.InferenceServerClient(
                url=self.url,
                verbose=False,
                ssl=False,
                root_certificates=None,
                private_key=None,
                certificate_chain=None)
        except Exception as e:
            print("context creation failed: " + str(e))
            sys.exit()
        if not self.triton_client.is_server_live():
            print("FAILED : is_server_live")
            sys.exit(1)

        if not self.triton_client.is_server_ready():
            print("FAILED : is_server_ready")
            sys.exit(1)
        
        if not self.triton_client.is_model_ready(self.model):
            print("FAILED : is_model_ready")
            sys.exit(1)
    
    def model_info(self):
        # Model metadata
        try:
            metadata = self.triton_client.get_model_metadata(self.model)
            print(metadata)
        except InferenceServerException as ex:
            if "Request for unknown model" not in ex.message():
                print("FAILED : get_model_metadata")
                print("Got: {}".format(ex.message()))
                sys.exit(1)
            else:
                print("FAILED : get_model_metadata")
                sys.exit(1)

        # Model configuration
        try:
            config = self.triton_client.get_model_config(self.model)
            if not (config.config.name == self.model):
                print("FAILED: get_model_config")
                sys.exit(1)
            print(config)
        except InferenceServerException as ex:
            print("FAILED : get_model_config")
            print("Got: {}".format(ex.message()))
            sys.exit(1)
            
    def detect(self, input_image, conf_thresh=0.5, nms_thresh=0.45):
        
        input_image_buffer = preprocess(input_image, [self.height, self.width])
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        self.inputs[0].set_data_from_numpy(input_image_buffer)

        results = self.triton_client.infer(model_name=self.model,
                                    inputs=self.inputs,
                                    outputs=self.outputs,
                                    client_timeout=None)

        result = results.as_numpy('output')[0]
        detected_objects = postprocess(result, input_image.shape[1], input_image.shape[0], (self.height, self.width), nms_thresh, conf_thresh)
        print('detected_objects',detected_objects)
        return detected_objects
