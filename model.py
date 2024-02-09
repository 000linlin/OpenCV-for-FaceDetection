import cv2
import numpy as np


class Haar_detection:
    def __init__(self):
        self.haar_detector = cv2.CascadeClassifier("./haar_info/haarcascade_frontalface_alt2.xml")

    # 检测人脸图像
    def detect(self, image, resize: tuple | bool = True, annotation=True):
        imageCopy = image.copy()
        resizeH, resizeW = image.shape[0], image.shape[1]
        if isinstance(resize, bool) and resize:
            H, W = image.shape[0], image.shape[1]
            resizeH = 300
            resizeW = int((W / H) * resizeH)
            image = cv2.resize(image, (resizeW, resizeH))
        elif isinstance(resize, tuple):
            H, W = image.shape[0], image.shape[1]
            resizeH, resizeW = resize
            image = cv2.resize(image, resize)
        else:
            H, W = resizeH, resizeW

        scaleH = H / resizeH
        scaleW = W / resizeW
        # 人脸检测
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_pos = self.haar_detector.detectMultiScale(gray_img)

        if annotation:
            for (x, y, w, h) in faces_pos:
                # 坐标(x, y), 宽w, 高h
                Rect = [
                    int(x * scaleW),
                    int(y * scaleH),
                    int((x + w) * scaleW),
                    int((y + h) * scaleH),
                ]
                imageCopy = cv2.rectangle(
                    imageCopy,
                    (Rect[0], Rect[1]),
                    (Rect[2], Rect[3]),
                    (255, 0, 0), 3
                )

            return imageCopy

        else:
            return faces_pos


class SSD_detection:
    def __init__(self, framework):
        if framework == "tf":
            self.ssd = cv2.dnn.readNet(
                model="./ssd_info/opencv_face_detector_uint8.pb",
                config="./ssd_info/opencv_face_detector.pbtxt",
                framework="tensorflow"
            )
            self.swapRB = True

        elif framework == "caffe":
            self.ssd = cv2.dnn.readNet(
                model="./ssd_info/res10_300x300_ssd_iter_140000_fp16.caffemodel",
                config="./ssd_info/deploy.prototxt",
                framework="caffe"
            )
            self.swapRB = False

        else:
            raise ValueError

    def detect(self, image, resize: tuple | bool = False, annotation=True):
        H, W = image.shape[:2]
        if isinstance(resize, bool) and resize:
            size = (500, 500)
        else:
            size = resize if isinstance(resize, tuple) else (0, 0)

        blobs = cv2.dnn.blobFromImage(
            image, scalefactor=1, size=size,
            mean=[104.0, 117.0, 123.0],
            swapRB=self.swapRB
        )

        self.ssd.setInput(blobs)
        detections = self.ssd.forward()  # detections.shape = [1, 1, 200, 7]

        if annotation:
            for i in range(0, detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf > 0.7:    # 筛选置信度高于0.7的人脸框
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    x1, y1, x2, y2 = box.astype("int")  # (x1, y1): 左下角坐标; (x2, y2): 右上角坐标

                    image = cv2.rectangle(
                        image,
                        (x1, y1), (x2, y2),
                        (255, 0, 0), 2  # 人脸标记
                    )
                    image = cv2.rectangle(
                        image,
                        (x1 - 2, y1 - 30), (x1 + 90, y1),
                        (255, 0, 0), -1
                    )

                    image = cv2.putText(
                        image, f"{conf:.3%}",
                        org=(x1 + 5, y1 - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6, color=(255, 255, 255),
                        thickness=2
                    )
            return image
        else:
            bboxes = []
            for i in range(0, detections.shape[2]):
                if detections[0, 0, i, 1] == 1.00:
                    conf = detections[0, 0, i, 2]
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    x1, y1, x2, y2 = box.astype("int")
                    bboxes.append([x1, y1, x2 - x1, y2 - y1, conf])

            return bboxes
