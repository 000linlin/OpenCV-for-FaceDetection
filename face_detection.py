import argparse
import time

import numpy as np

from model import Haar_detection, SSD_detection
import os
import cv2

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="人脸检测程序")
parser.add_argument("path", type=str, help="待检测图像或视频的路径")
parser.add_argument(
    "-v", "--video",
    action='store_true',
    help="是否检测视频流"
)
parser.add_argument(
    "-s", "--save",
    action='store_true',
    help="是否保存检测结果"
)
parser.add_argument(
    "-m", "--model",
    choices=["haar", "ssd"],
    default="ssd",
    help="支持haar cascade和SSD网络"
)
parser.add_argument(
    "-fw", "--framework",
    choices=["caffe", "tf"],
    default="caffe",
    help="基于两种框架[caffe||tensorflow]训练的SSD网络",
)
parser.add_argument(
    "-r", "--resize",
    type=lambda x: eval(x) if isinstance(eval(x), (tuple, bool)) else None,
    help="调整输入尺寸: bool or tuple"
)

args = parser.parse_args()
path = args.path
model = args.model
fw = args.framework
resize = args.resize
is_save = args.save
is_video = args.video

# 初始化分类器
face_detector = Haar_detection() if model == "haar" else SSD_detection(fw)

dirPath = "./data/result"
if not os.path.exists(dirPath):
    os.makedirs(dirPath)

if (is_video):
    capture = cv2.VideoCapture(path)
    outFile = os.path.basename(path)[:-4] + "_" + f"{model}" + ".mp4"
    ret, frame = capture.read()

    if is_save:
        vid_writer = cv2.VideoWriter(
            os.path.join(dirPath, outFile),
            cv2.VideoWriter_fourcc(*'mp4v'),  # 使用 MP4 编解码器
            30.0,
            (frame.shape[1], frame.shape[0]),
        )

    frame_count = 0
    tt = 0

    fps_list = []
    while True:
        # 获取每一帧
        ret, frame = capture.read()

        if not ret:
            break

        frame_count += 1
        t = time.time()

        # 人脸检测
        frame_detection = face_detector.detect(frame, resize=resize)

        tt += time.time() - t
        if not tt: tt = 1e-9
        FPS = frame_count / tt
        fps_list.append(FPS)

        cv2.putText(
            frame_detection, text=f"FPS: {FPS:.2f}",
            org=(10, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.3, color=(0, 0, 255),
            thickness=3, lineType=cv2.LINE_AA,
        )
        cv2.imshow("人脸检测", frame_detection)

        if is_save:
            vid_writer.write(frame_detection)
        if cv2.waitKey(5) == 27:
            break
        if frame_count == 1:
            tt = 0

    capture.release()
    cv2.destroyAllWindows()
    if is_save:
        vid_writer.release()
    if isinstance(resize, tuple):
        print(f"模型: {model}, 输入尺寸：{resize}, 平均FPS: {np.array(fps_list).mean():.2f}")

else:
    # 读取图像
    img = cv2.imread(path)
    # 人脸检测
    res = face_detector.detect(img, resize)

    plt.figure(figsize=(12, 7))
    img_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis("off")

    if is_save:
        plt.savefig(
            os.path.join(dirPath, f"test_{model}.jpg"),
            bbox_inches="tight",
            pad_inches=0.05, dpi=100
        )

    plt.show()
