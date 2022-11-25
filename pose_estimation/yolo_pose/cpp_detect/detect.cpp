#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

//#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace dnn;
using namespace std;

struct NetConfig
{
    float person_conf_thres;
    float person_iou_thres;
    float kp_conf_thres;
    float kp_iou_thres;
    float conf_thres_kp_person;
    int overwrite_tal;
    bool use_kp_dets;
};

int endswith(string s, string sub)
{
    return s.rfind(sub) == (s.length() - sub.length()) ? 1 : 0;
}

int kp_face[5] = {0, 1, 2, 3, 4};
int segment[12][12] = {{5,6}, {5, 11}, {11, 12}, {12, 6}, {5, 7}, {7, 9},
                        {6, 8}, {8, 10},{11, 13}, {13, 15}, {12, 14}, {14, 16}};

class KAPAO  // model KAPAO 
{
    public:
        KAPAO(NetConfig config, string model_path);
        void detect(Mat& frame);
    private:
        const int width = 640;  // input image width
        const int height = 640; // input image height
        const int num_stride = 3;
        const bool keep_ratio = true;
        vector<string> class_names;
        int num_class;
        int num_lines;
        int num_face_pts;
        int* plines;

        Net net;
        NetConfig config;
        void drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid);
        Mat resize_image(Mat src_img, int* h_new, int* w_new, int* top, int* left);
        const float anchors[3][6] = {{10, 13, 16,30, 33,23}, {30, 61, 62, 45, 59, 119}, {116,90, 156, 198, 373, 326}};
};

KAPAO::KAPAO(NetConfig config, string model_path)
{
    this->config.person_conf_thres = config.person_conf_thres;
    this->config.person_iou_thres = config.person_iou_thres;
    this->config.kp_conf_thres = config.kp_conf_thres;
    this->config.kp_iou_thres = config.kp_iou_thres;
    this->config.conf_thres_kp_person = config.conf_thres_kp_person;
    this->config.overwrite_tal = config.overwrite_tal;
    this->config.use_kp_dets = config.use_kp_dets;

    this->net = readNetFromONNX(model_path);
    if (endswith(model_path, ".onnx"))
    {
        ifstream ifs("class.names");
        string line;
        while (getline(ifs, line)) this->class_names.push_back(line);
        this->num_lines = 14;
        this->num_face_pts = 0;
        // plines = (int*) crowd_segments;
    }
    else
    {
        cout << "Error: model_path: " << model_path << endl;
    }
    this->num_class = class_names.size();
}

Mat KAPAO::resize_image(Mat src_img, int* h_new, int* w_new, int* top, int* left)
{
    int src_h = src_img.rows, src_w = src_img.cols;
    *h_new = this->height;
    *w_new = this->width;
    Mat dst_img;
    if (this->keep_ratio && src_h != src_w) {
        float hw_scale = (float) src_h / src_w;
        if (hw_scale > 1)
        {
            *h_new = this->height;
            *w_new = int(this->width / hw_scale);
            resize(src_img, dst_img, Size(*w_new, *h_new), INTER_AREA);
            *left = int(this->width - *w_new) * 0.5;
            copyMakeBorder(dst_img, dst_img, 0, 0, *left, this->width - *w_new - *left, BORDER_CONSTANT, 114);
        }
        else{
            *h_new = (int)this->height * hw_scale;
            *w_new = this->width;
            resize(src_img, dst_img, Size(*w_new, *h_new), INTER_AREA);
            *top = (int)(this->height - *h_new) * 0.5;
            copyMakeBorder(dst_img, dst_img, *top, this->height - *h_new - *top, 0, 0, BORDER_CONSTANT, 114);
        }
    }
    else
    {
        resize(src_img, dst_img, Size(*w_new, *h_new), INTER_AREA);
    }
    return dst_img;
}

void KAPAO::drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid)
{
    // Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 1);
    // Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    label = this->class_names[classid] + ':' + label;
    // Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top-int(1.5*labelSize.height)),
                Point(left+int(1.5*labelSize.width), top+baseLine),
                Scalar(0, 255, 0), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
}

void KAPAO::detect(Mat& frame)
{
    int h_new=0, w_new=0, h_pad = 0, w_pad = 0;
    Mat dst_img = this->resize_image(frame, &h_new, &w_new, &h_pad, &w_pad);
    Mat blob = blobFromImage(dst_img, 1/255.0, Size(this->width, this->height),
                            Scalar(0, 0, 0), true, false);
    this->net.setInput(blob);
    Mat outs;
    // vector<Mat> outs;
    // this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
    this->net.forward(outs);
    // Mat output = this->net.forward();

    int num_proposal = outs.size[1];   // outs.shape vector([[Batch, num_grids, 57]])
    int element_size = outs.size[2];
    if (outs.dims > 2)
    {
        outs = outs.reshape(0, num_proposal);
    }
    const int num_coords = (element_size - this->num_class - 5) * 0.5;
    // generate proposals
    vector<float> person_confidences;
    vector<Rect> person_boxes;
    vector<int> person_classIds;
    vector<float>kp_confidences;
    vector<Rect> kp_boxes;
    vector<int> kp_classIds;
    vector<vector<float> > poses;

    // Get bounding box 
    float h_ratio = (float)frame.rows / h_new, w_ratio = (float)frame.cols / w_new;
    int grid_index = 0;  // the index of grids.
    float* pdata = (float*)outs.data;   // 1 dimension array： num_grids * 57 = [element57, element57, element57, ...]
    // float * pdata = outs.ptr<float>(); 
    for (int n=0; n < this->num_stride; ++n)
    {
        const float stride = pow(2, n+3);
        int num_grid_x = (int)ceil((this->width / stride));
        int num_grid_y = (int)ceil((this->height / stride));
        for (int q = 0; q < 3; ++q)
        {
            const float w_anchor = this->anchors[n][q*2];
            const float h_anchor = this->anchors[n][q*2 + 1];
            for (int i=0; i < num_grid_y; ++i)  // grid y index
            {
                for (int j=0; j < num_grid_x; ++j)  // grid x index
                {
                    float box_score = pdata[4];
                    Mat scores = outs.row(grid_index).colRange(5, 5 + this->num_class);
                    Point classIdPoint;
                    double max_class_score;
                    // Get the value and location of the maximum score
                    minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);
                    max_class_score *= box_score;
                    const int class_idx = classIdPoint.x;
                    
                    if  (class_idx == 0)   // person line
                    {
                        if (box_score > this->config.person_conf_thres && max_class_score > this->config.person_conf_thres)
                        {
                            float cx = (2.f * pdata[0] - 0.5f + j) * stride;  // center point x
                            float cy = (2.f * pdata[1] - 0.5f + i) * stride;  // center point y
                            float w = powf(2.f * pdata[2], 2.f) * w_anchor;  // bbox w
                            float h = powf(2.f * pdata[3], 2.f) * h_anchor;  // bbox h

                            int left = int((cx - w_pad - 0.5 * w) * w_ratio);
                            int top = int((cy - w_pad - 0.5 * h) * h_ratio);

                            person_confidences.push_back((float)max_class_score);
                            person_boxes.push_back(Rect(left, top, (int)(w*w_ratio), (int)(h*h_ratio)));
                            person_classIds.push_back(class_idx);

                            // Decode keypoints
                            vector<float> kp(3*num_coords, 0);
                            for (int k=0; k < num_coords; ++k)
                            {
                                float x = 4*pdata[5 + this->num_class + 2*k] - 2;
                                float y = 4*pdata[5 + this->num_class + 2*k + 1] - 2;
                                x *= w_anchor;
                                y *= h_anchor;
                                x += j * stride;
                                y += i * stride;
                                x = (x - w_pad) * w_ratio;
                                y = (y - h_pad) * h_ratio;
                                kp[3*k] = x;
                                kp[3*k + 1] = y;
                            }
                            poses.push_back(kp);
                        }
                    }
                    else
                    {
                        if (box_score > this->config.kp_conf_thres and max_class_score > this->config.kp_conf_thres)
                        {
                            float cx = (2.f * pdata[0] - 0.5f + j) * stride;
                            float cy = (2.f * pdata[1] - 0.5f + i) * stride;
                            float w = powf(2.f * pdata[2], 2.f) * w_anchor;
                            float h = powf(2.f * pdata[3], 2.f) * h_anchor;

                            int left = int((cx - w_pad - 0.5 * w) * w_ratio);
                            int top = int((cy - h_pad - 0.5 * h) * h_ratio);

                            kp_confidences.push_back((float)max_class_score);
                            kp_boxes.push_back(Rect(left, top, (int)(w*w_ratio), (int)(h*h_ratio)));
                            kp_classIds.push_back(class_idx);
                        }
                    }
                    grid_index++;
                    pdata += element_size;
                }
            }
        }
    }

    // Perform non-maximum suppression to eliminate redundant overlapping boxes with lower confidences.
    vector<int> person_indices;
    dnn::NMSBoxes(person_boxes, person_confidences, this->config.person_conf_thres, this->config.person_iou_thres, person_indices);
    vector<int> kp_indices;
    // TODO: all kinds of keypoint bounding boxes are NMS as the same class ?
    dnn::NMSBoxes(kp_boxes, kp_confidences, this->config.kp_conf_thres, this->config.kp_iou_thres, kp_indices);
    vector<int> pose_mask;
    for (int i = 0; i < person_indices.size(); ++i)
    {
        const int person_id = person_indices[i];
        if (person_confidences[person_id] > this->config.conf_thres_kp_person)
        {
            pose_mask.push_back(person_id);
        }
    }
    // Fuse keypoint bboxes and person keypoints
    for (int i=0; i < kp_indices.size(); ++i)
    {
        int idx = kp_indices[i];
        Rect box = kp_boxes[idx];
        float x = box.x + box.width * 0.5;
        float y = box.y + box.height * 0.5;
        float conf = kp_confidences[idx];
        int pt_id = kp_classIds[idx] - 1;
        int min_id = 0;
        float min_dist = 10000;
        for (int j=0; j < pose_mask.size(); ++j)
        {
            const int pose_id = pose_mask[j];
            const float dist = sqrt(powf(poses[pose_id][pt_id*3] - x, 2) + 
                                    powf(poses[pose_id][pt_id*3 + 1] - y, 2));
            
            if (dist < min_dist)
            {
                min_dist = dist;
                min_id = pose_id;
            }
        }
        if (conf > poses[min_id][pt_id*3 + 2] and min_dist < this->config.overwrite_tal)
        {
            poses[min_id][pt_id*3] = x;
            poses[min_id][pt_id*3 + 1] = y;
            poses[min_id][pt_id*3 + 2] = conf;
        }
    }

    // draw bbox
    for (int i=0; i < person_indices.size(); ++i)
    {
        int idx = person_indices[i];
        Rect box = person_boxes[idx];
        this->drawPred(person_confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height,
                        frame, person_classIds[idx]);
    }
    // draw line
    for (int i=0; i < pose_mask.size(); ++i)
    {
        for (int j=0; j < num_coords; ++j)
        {
            if (poses[pose_mask[i]][j*3 + 2] > 0)   // if vis > 0
            {
                int x = int(poses[pose_mask[i]][j*3]);
                int y = int(poses[pose_mask[i]][j*3 + 1]);
                circle(frame, Point(x, y), 1, Scalar(0, 255, 0), -1);
            }
        }
        for (int j=0; j < this->num_lines; ++ j)
        {
            int x1 = int(poses[pose_mask[i]][this->plines[2*j]*3]);
            int y1 = int(poses[pose_mask[i]][this->plines[2*j]*3 + 1]);
            int x2 = int(poses[pose_mask[i]][this->plines[2*j + 1]*3]);
            int y2 = int(poses[pose_mask[i]][this->plines[2*j + 1]*3 + 1]);
            line(frame, Point(x1, y1), Point(x2, y2), Scalar(255, 0, 255), 1);
        }
    }

}

int main()
{
    NetConfig cfg = {0.7, 0.45, 0.5, 0.45, 0.2, 25, true};
    string img_path = "/home/huangzhiyong/Project/kapao/res/crowdpose_100024.jpg";
    std::cout << "=> Loading image from " << img_path << std::endl;
    Mat src_img = imread(img_path);
    std::cout << "=> Loading Model ..." << std::endl;
    KAPAO model(cfg, "/home/huangzhiyong/Project/kapao/cpp_detect/model.onnx");
    std::cout << "=> Detecting ... " <<  std::endl;
    model.detect(src_img);
    std::cout << "=> Done ... " <<  std::endl;

    static const string window_name = "Keypoint and Pose as Object (KAPAO)";
    namedWindow(window_name, WINDOW_NORMAL);
    imshow(window_name, src_img);
    waitKey(0);
    destroyAllWindows();
}