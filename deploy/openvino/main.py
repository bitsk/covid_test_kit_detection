# try:
#     import ncnn
# except ImportError:
#     import ncnn_replace as ncnn
import time
from detection.retinaface import RetinaFace
from detection.scrfd import SCRFD
from openvino.inference_engine import IECore
import cv2
import numpy as np

#['CPU', 'GNA', 'MYRIAD']
detector_scrfd_config = {
    'image_size': 640,
    'nms_threshold': 0.4,
    'conf_threshold': 0.5,
    'minface': 10,
    'device': 'CPU',
    'batch_size': 1,
    'max_num': 1,
    'input_name': 'input.1',
    'output_names': ['score_8','score_16','score_32','bbox_8','bbox_16','bbox_32','kps_8','kps_16','kps_32'],
    # 'xml_file': './models/scrfd_lpr_openvino_model/openvino_fp16_d/scrfd_500m_bnkps_shapedynimic_o.xml',
    # 'bin_file': './models/scrfd_lpr_openvino_model/openvino_fp16_d/scrfd_500m_bnkps_shapedynimic_o.bin'
    'xml_file': './models/covid0/scrfd_500m_bnkps_fast_covid-kit.xml',
    'bin_file': './models/covid0/scrfd_500m_bnkps_fast_covid-kit.bin'
}

class TCLPR(object):
    def __init__(self, detector_type='scrfd', draw_result=False, lpr_type='ncnn'):
        self.ie = IECore()
        self.detector_type = detector_type
        self.lpr_type = lpr_type
        self.detector = SCRFD(self.ie, detector_scrfd_config)
        self.draw_result = draw_result

    def __call__(self, frame):
        return self.predict(frame)
    
    def predict(self, frame):
        bboxes, landmarks = self.detector.predict(frame)
        for i, (box, landmark) in enumerate(zip(bboxes, landmarks)):
            xmin, ymin, xmax, ymax, score = box
            xmin = int(max(0, xmin))
            ymin = int(max(0, ymin))
            xmax = int(min(frame.shape[1], xmax))
            ymax = int(min(frame.shape[0], ymax))
            width = xmax - xmin
            height = ymax - ymin
            # xmin -= int(width * 0.1)
            # xmax += int(width * 0.1)
            # ymin -= int(height * 0.1)
            # ymax += int(height * 0.1)
            if self.draw_result:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            w = xmax - xmin + 1
            h = ymax - ymin + 1
            img_box = np.zeros((h, w, 3), dtype=np.uint8)
            img_box = frame[ymin:ymax + 1, xmin:xmax + 1, :].copy()
            if self.detector_type == 'rtf':
            # landms
                b = list(map(int, landmark))
                if self.draw_result:
                    cv2.circle(frame, (b[0], b[4]), 1, (0, 0, 255), 4)
                    cv2.circle(frame, (b[1], b[5]), 1, (0, 255, 255), 4)
                    # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                    cv2.circle(frame, (b[2], b[6]), 1, (0, 255, 0), 4)
                    cv2.circle(frame, (b[3], b[7]), 1, (255, 0, 0), 4)
                new_x1, new_y1 = b[2] - xmin, b[6] - ymin
                new_x2, new_y2 = b[3] - xmin, b[7] - ymin
                new_x3, new_y3 = b[1] - xmin, b[5] - ymin
                new_x4, new_y4 = b[0] - xmin, b[4] - ymin
                
            else:
                kps = landmark.astype(np.int32)
                if self.draw_result:
                    cv2.circle(frame, kps[2], 1, (0, 0, 255), 4)
                    cv2.circle(frame, kps[3], 1, (0, 255, 255), 4)
                    cv2.circle(frame, kps[1], 1, (0, 255, 0), 4)
                    cv2.circle(frame, kps[0], 1, (255, 0, 0), 4)
                new_x1, new_y1 = int(kps[0][0]) - xmin, int(kps[0][1]) - ymin
                new_x2, new_y2 = int(kps[1][0]) - xmin, int(kps[1][1]) - ymin
                new_x3, new_y3 = int(kps[2][0]) - xmin, int(kps[2][1]) - ymin
                new_x4, new_y4 = int(kps[3][0]) - xmin, int(kps[3][1]) - ymin
            points1 = np.float32([[new_x1, new_y1], [new_x2, new_y2], [new_x3, new_y3], [new_x4, new_y4]])
            points2 = np.float32([[0, 0], [16, 0], [16, 96], [0, 96]]) #NOTE 16 * 96
            M = cv2.getPerspectiveTransform(points1, points2)
            try:
                processed = cv2.warpPerspective(img_box, M, (16, 96))
                cv2.imshow('result area', processed)
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) 
                cv2.imshow('gray', gray)
                blur_img = cv2.medianBlur(gray,5) 
                cv2.imshow('blur_img', blur_img)
                bin_adapt = cv2.adaptiveThreshold(blur_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
                cv2.imshow('bin_adapt', bin_adapt)
                _, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                cv2.imshow('binary', binary)
            except cv2.error:
                return None
        return None

if __name__ == '__main__':
    tclpr = TCLPR('scrfd', True, 'openvino')
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        start = time.time()
        # frame = cv2.copyMakeBorder(frame, 0, 160, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        tclpr.predict(frame)
        end = time.time()
        print(end - start)
        # print(plate_str)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()