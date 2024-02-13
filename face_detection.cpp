#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

using namespace std;

const string IMAGE_PATH = "./data/test.jpg";
const string VIDEO_PATH = "./data/rowing.mp4";

const string XML_PATH = "./haar_info/haarcascade_frontalface_alt2.xml";
const string CAFFE_CONFIG = "./ssd_info/deploy.prototxt";
const string CAFFE_MODEL = "./ssd_info/res10_300x300_ssd_iter_140000_fp16.caffemodel";
const string TF_CONFIG = "./ssd_info/opencv_face_detector.pbtxt";
const string TF_MODEL = "./ssd_info/opencv_face_detector_uint8.pb";

class Detector 
{
public:
    virtual void detect(
        cv::Mat image, 
        vector<cv::Rect> &faces, 
        cv::Size size
    ) = 0; 
};


// 哈尔级联
class Haar_detection: public Detector
{
private:
    cv::CascadeClassifier haar_detector;

public:
    Haar_detection(const string xmlPath){
        haar_detector.load(xmlPath);
    };

    void detect(
        cv::Mat image, 
        vector<cv::Rect> &faces, 
        cv::Size size = cv::Size(300, 300)
        ) override
    {   

        float scaleW, scaleH;
        cv::Mat resized_img, gray_img;
    
        if (size == cv::Size(0, 0)) {
            scaleW = scaleH = 1.0;
            resized_img = image;
        }
        else {
            scaleW = image.cols / (float)size.width;
            scaleH = image.rows / (float)size.height;
            cv::resize(image, resized_img, size);
        }

        cv::cvtColor(resized_img, gray_img, cv::COLOR_BGR2GRAY);

        // 进行人脸检测
        haar_detector.detectMultiScale(gray_img, faces);
        
        for (const auto& rect: faces) 
        {
            int x1 = (int)(rect.x * scaleW);
            int y1 = (int)(rect.y * scaleH);
            int x2 = (int)((rect.x + rect.width) * scaleW);
            int y2 = (int)((rect.y + rect.height) * scaleH);
            cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 3);    
        }
        
        return;
    };

};


// SSD网络
class SSD_detection: public Detector
{
private:
cv::dnn::Net ssd_detector;
bool swapRB = false;

public:
    SSD_detection(){ssd_detector = cv::dnn::readNetFromCaffe(CAFFE_CONFIG, CAFFE_MODEL);}
    SSD_detection(const string framework)
    {
        if (framework == "tf") {
            ssd_detector = cv::dnn::readNet(TF_MODEL, TF_CONFIG, framework);            
            swapRB = true;
        }

        else if(framework == "caffe"){
            ssd_detector = cv::dnn::readNet(CAFFE_MODEL, CAFFE_CONFIG, framework);
            swapRB = false;
        }

        else {throw invalid_argument("Invalid framework, only supporting 'caffe' or 'tf' ");}
    };

    void detect(
        cv::Mat image, 
        vector<cv::Rect> &faces, 
        cv::Size size = cv::Size(300, 300)
        ) override
    {
        cv::Mat blobs = cv::dnn::blobFromImage(
            image, 1, size,
            cv::Scalar(104.0, 177.0, 123.0),
            swapRB
        );

        ssd_detector.setInput(blobs);
        cv::Mat detections = ssd_detector.forward("detection_out");
        cv::Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());

        for (int i = 0; i < detectionMat.rows; i++)
        {
            float conf = detectionMat.at<float>(i, 2);
            if (conf > 0.6) {
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows);

                cv::rectangle(
                    image, 
                    cv::Point(x1, y1), cv::Point(x2, y2), 
                    cv::Scalar(255, 0, 0), 2
                );
                
                cv::rectangle(
                    image, 
                    cv::Point(x1 - 2, y1 - 30), cv::Point(x1 + 90, y1), 
                    cv::Scalar(255, 0, 0), -1
                );

                cv::putText(
                    image, cv::format("%.3f%%", conf * 100),
                    cv::Point(x1 + 5, y1 - 10), 
                    cv::FONT_HERSHEY_SIMPLEX, 
                    0.6, cv::Scalar(255, 255, 255), 2
                );
            }
        } 

    };
};



int main(int argc, char *argv[])
{
    string path, model, isVideo, size, framework; 
    model = "haar";
    path = IMAGE_PATH;
    isVideo = "";
    size = "(0,0)";
    framework = "caffe";

    switch (argc) 
    {
        case 1:
            break;
        case 2:
            path = argv[1];
            break;
        case 3:
            path = argv[1];
            model = argv[2];
            break;
        case 4: 
            path = argv[1];
            model = argv[2];
            isVideo = argv[3];
            break;
        case 5:
            path = argv[1];
            model = argv[2];
            isVideo = argv[3];
            size = argv[4];
            break;
        case 6:
            path = argv[1];
            model = argv[2];
            isVideo = argv[3];
            size = argv[4];
            framework = argv[5];
            break;
        default:
            throw invalid_argument("arguments: path|model|isVideo|size|framework");
    }

    size_t comma_pos = size.find(',');
    int W, H;
    if (comma_pos != string::npos) {
        W = stoi(size.substr(1, comma_pos));
        H = stoi(size.substr(comma_pos + 1, size.length() - 2));
    }
    else {throw invalid_argument("Invalid size");} 

    unique_ptr<Detector> face_detector;
    if (model == "haar") {
        face_detector = make_unique<Haar_detection>(XML_PATH);
    }
    else {face_detector = make_unique<SSD_detection>(framework);}

    if (isVideo == "T" || isVideo == "true") {
        if (path.substr(path.find_last_of('.')) != ".mp4") {
            throw invalid_argument("path should is a mp4 file");
        }
        cout << "video detection:" << endl;
    }
    else cout << "image detection:" << endl;

    cout << "model - " << model << endl;
    cout << "input size: (" << W << "," << H << ")" << endl;
    if (model == "ssd") cout << "framework - " << framework << endl;

    if (isVideo == "T" || isVideo == "true") 
    {
        cv::VideoCapture source;
        source.open(path);

        double tt = 0;
        double fps = 0;

        cv::Mat frame;
        cv::namedWindow("FaceDetection", cv::WINDOW_FREERATIO);
        while (true)
        {
            source >> frame;
            if (frame.empty()) break;

            double t = cv::getTickCount();
            
            vector<cv::Rect> faces;
            face_detector->detect(frame, faces, cv::Size(W, H));

            tt = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
            fps = 1/tt;

            cv::putText(
                frame, 
                cv::format("FPS = %.2f", fps), 
                cv::Point(10, 50), 
                cv::FONT_HERSHEY_SIMPLEX, 
                1.3, cv::Scalar(0, 0, 255), 3, cv::LINE_AA
            );

            cv::imshow("FaceDetection", frame);
            if(cv::waitKey(5) == 27) break;
        }
        source.release();
        cv::destroyAllWindows();

    }

    else
    {
        cv::Mat img = cv::imread(path);
        cv::namedWindow("FaceDetection", cv::WINDOW_FREERATIO);

        vector<cv::Rect> faces;
        face_detector->detect(img, faces, cv::Size(W, H));

        cv::imshow("FaceDetection", img);
        cv::waitKey();
    }

    return 0;
    
}