import os.path

from model import Haar_detection, SSD_detection
import cv2
import matplotlib.pyplot as plt

def test_WF(face_detector, resize: bool | tuple = None):
    imageDir = "./data/WiderFace/test_image"
    imageListPath = "./data/WiderFace/test_image/test_image_list.txt"

    with open(imageListPath, "r") as f:
        fileNameList = f.readlines()
        plt.figure(figsize=(4, 3))

        pos = [1, 4, 7, 2, 5, 8, 3, 6, 9]
        for i in range(9):
            plt.subplot(3, 3, pos[i])
            fileName = fileNameList[i].strip()
            imagePath = os.path.join(imageDir, fileName)
            image = cv2.imread(imagePath)
            image = face_detector.detect(image, resize=resize)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis("off")

        plt.subplots_adjust(hspace=0, wspace=0.05)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="haar")
    parser.add_argument(
        "-r", "--resize",
        type=lambda x: eval(x) if isinstance(eval(x), (tuple, bool)) else None,
    )

    args = parser.parse_args()
    model = args.model
    resize = args.resize

    if model == "haar":
        detector = Haar_detection("./haar_info/haarcascade_frontalface_alt2.xml")
    else:
        detector = SSD_detection("caffe")
    test_WF(detector, resize)

    plt.savefig(
        f"./data/result/test_image_{model}.jpg",
        bbox_inches="tight",
        pad_inches=0.05, dpi=400
    )

    plt.show()


