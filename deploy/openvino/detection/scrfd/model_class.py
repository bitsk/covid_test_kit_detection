import numpy as np
from numpy.lib import stride_tricks
import yaml
import cv2

from pathlib import Path

from ..base_detector import BaseFaceDetector

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

class SCRFD(BaseFaceDetector):
    def __init__(self, ie, config: dict):
        super().__init__(config)
        xml_file = config['xml_file']
        bin_file = config['bin_file']
        self.net = ie.read_network(xml_file, bin_file)
        self.batch_size = config['batch_size']
        self.net.batch_size = self.batch_size
        self.input_name = config['input_name']
        self.output_names = config['output_names']
        
        self.input_layer = next(iter(self.net.input_info))
        print(f"Net input shape: {self.net.input_info[self.input_name].tensor_desc.dims}")

        self.input_width = 640
        self.input_height= 480
        # self.input_width = 640

        self.net.reshape({self.input_layer: (1, 3, self.input_height, self.input_width)})
        print(f"Net input shape: {self.net.input_info[self.input_name].tensor_desc.dims}")

        self.model = ie.load_network(network = self.net, device_name = config["device"])

        self.max_num = config['max_num'] # 1

        self.center_cache = {}
        self.conf_threshold = config['conf_threshold']
        self.nms_thresh = config['nms_threshold']
        self.model_input_shape = None
        self.resize_scale = None
        self._init_vars()

    def _init_vars(self):
        # input_cfg = self.session.get_inputs()[0]
        # input_shape = input_cfg.shape
        # if isinstance(input_shape[2], str):
        #     self.input_size = None
        # else:
        #     self.input_size = tuple(input_shape[2:4][::-1])
        # input_name = input_cfg.name
        outputs = self.output_names
        # if len(outputs[0].shape) == 3:
        #     self.batched = True
        # output_names = []
        # for o in outputs:
        #     output_names.append(o.name)
        # self.input_name = input_name
        # self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs)==6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs)==9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs)==10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs)==15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    # def prepare(self, ctx_id, **kwargs):
    #     if ctx_id<0:
    #         self.session.set_providers(['CPUExecutionProvider'])
    #     nms_thresh = kwargs.get('nms_thresh', None)
    #     if nms_thresh is not None:
    #         self.nms_thresh = nms_thresh
    #     input_size = kwargs.get('input_size', None)
    #     if input_size is not None:
    #         if self.input_size is not None:
    #             print('warning: det_size is already set in scrfd model, ignore')
    #         else:
    #             self.input_size = input_size

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # img = np.float32(image)
        # target_size = self.config["image_size"]
        # im_shape = img.shape
        # img_width = im_shape[1]
        # img_height = im_shape[0]
        # self.resize_scale = [float(target_size) / float(img_width), float(target_size) / float(img_height)]
        # img = cv2.resize(img, (target_size, target_size))
        # img -= np.asanyarray((127.5, 127.5, 127.5), dtype=np.float32)
        # img /= 128.0
        # img = image.transpose((2, 0, 1))
        # img = np.expand_dims(img, axis=0)

        # input_size = tuple([self.config["image_size"],self.config["image_size"]])
        input_size = tuple([image.shape[1], image.shape[0]])
        # print("input image size is:", input_size)
        blob = cv2.dnn.blobFromImage(image, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        return blob

    def _predict_raw(self, image: np.ndarray):
        # img = image.transpose((2, 0, 1))
        # img = np.expand_dims(img, axis=0)
        self.model.requests[0].infer(inputs={self.input_name: image})
        pred = self.model.requests[0].outputs
        # pred = (pred[self.output_names[0]][0], pred[self.output_names[1]][0], pred[self.output_names[2]][0])
        return pred

    def _postprocess(self, raw_prediction):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        fmc = self.fmc
       

        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = raw_prediction[self.output_names[idx]][0]
            scores.tofile("C:/Users/btx00/Desktop/scrfd_covid_kit/save_bin/openvino/"+self.output_names[idx]+".bin")
            bbox_preds = raw_prediction[self.output_names[idx+fmc]][0]
            bbox_preds.tofile("C:/Users/btx00/Desktop/scrfd_covid_kit/save_bin/openvino/"+self.output_names[idx+fmc]+".bin")
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = raw_prediction[self.output_names[idx+fmc*2]][0]
                kps_preds.tofile("C:/Users/btx00/Desktop/scrfd_covid_kit/save_bin/openvino/"+self.output_names[idx+fmc*2]+".bin")
                kps_preds = kps_preds * stride
        # print(scores_list)
            # print("score shape:", scores.shape)
            # print("bbox_preds shape:", bbox_preds.shape)
            # print("kps_preds shape:", kps_preds.shape)

            height = self.input_height // stride #FIXME: if dynamic input size need modify this
            width = self.input_width // stride #FIXME: if dynamic input size need modify this
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                #solution-1, c style:
                #anchor_centers = np.zeros( (height, width, 2), dtype=np.float32 )
                #for i in range(height):
                #    anchor_centers[i, :, 1] = i
                #for i in range(width):
                #    anchor_centers[:, i, 0] = i

                #solution-2:
                #ax = np.arange(width, dtype=np.float32)
                #ay = np.arange(height, dtype=np.float32)
                #xv, yv = np.meshgrid(np.arange(width), np.arange(height))
                #anchor_centers = np.stack([xv, yv], axis=-1).astype(np.float32)

                #solution-3:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                #print(anchor_centers.shape)

                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                if self._num_anchors>1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
                if len(self.center_cache)<100:
                    self.center_cache[key] = anchor_centers
            # print(anchor_centers.shape)

            pos_inds = np.where(scores>=self.conf_threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
            ## post-processing
            scores = np.vstack(scores_list)
            scores_ravel = scores.ravel()
            order = scores_ravel.argsort()[::-1]
            bboxes = np.vstack(bboxes_list) #FIXME: dynamic input need / det_scale
            if self.use_kps:
                kpss = np.vstack(kpss_list) #FIXME: dynamic input need/ det_scale
            pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
            pre_det = pre_det[order, :]
            keep = self.nms(pre_det)
            det = pre_det[keep, :]
            if self.use_kps:
                kpss = kpss[order,:,:]
                kpss = kpss[keep,:,:]
            else:
                kpss = None
            if self.max_num > 0 and det.shape[0] > self.max_num:
                area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                        det[:, 1])
                img_center = self.input_width // 2, self.input_height // 2 # Notice: input_width, input_height swap here?
                offsets = np.vstack([
                    (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - img_center[0]
                ])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                # if metric=='max':
                # if None:
                #     values = area
                # else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
                bindex = np.argsort(
                    values)[::-1]  # some extra weight on the centering
                bindex = bindex[0: self.max_num]
                det = det[bindex, :]
                if kpss is not None:
                    kpss = kpss[bindex, :]
        return det, kpss

    def predict(self, image: np.ndarray):
        image = self._preprocess(image)
        self.model_input_shape = image.shape
        raw_pred = self._predict_raw(image)
        bboxes, landms = self._postprocess(raw_pred)
        # converted_landmarks = []
        # for landmarks_set in landms:
        #     x_landmarks = []
        #     y_landmarks = []
        #     for i, lm in enumerate(landmarks_set):
        #         if i % 2 == 0:
        #             x_landmarks.append(lm)
        #         else:
        #             y_landmarks.append(lm)
        #     converted_landmarks.append(x_landmarks + y_landmarks)

        # landmarks = np.array(converted_landmarks)
        # return bboxes, landmarks
        return bboxes, landms

 
    def nms(self, dets):
        thresh = self.nms_thresh
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

if __name__ == "__main__":
    from openvino.inference_engine import IECore
    detector_scrfd_config = {
    'image_size': 640,
    'nms_threshold': 0.6,
    'conf_threshold': 0.5,
    'minface': 10,
    'device': 'CPU',
    'batch_size': 1,
    'max_num': 1,
    'input_name': 'input.1',
    'output_names': ['score_8','score_16','score_32','bbox_8','bbox_16','bbox_32','kps_8','kps_16','kps_32'],
    'xml_file': 'C:/Users/btx00/Desktop/scrfd_covid_kit/models/scrfd_500m_bnkps_fast_covid-kit.xml',
    'bin_file': 'C:/Users/btx00/Desktop/scrfd_covid_kit/models/scrfd_500m_bnkps_fast_covid-kit.bin'
    }
    ie = IECore()
    detector = SCRFD(ie, detector_scrfd_config)
    image = cv2.imread('C:/Users/btx00/Desktop/scrfd_covid_kit/images/hotgen_covid-19_05.jpg')
    bboxes, landmarks = detector.predict(image)
    print(bboxes)
    print(landmarks)

    