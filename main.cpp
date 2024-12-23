#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;


class SCRFD
{
public:
	SCRFD(string modelpath, const float confThreshold, const float nmsThreshold);
	void detect(Mat& frame);
private:

	const float stride[3] = { 8.0, 16.0, 32.0 };
	const int inpWidth = 640;
	const int inpHeight = 640;
	float confThreshold;
	float nmsThreshold;
	const bool keep_ratio = true;
	Net net;
	Mat resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left);
};

SCRFD::SCRFD(string modelpath, const float conf_Threshold, const float nms_Threshold)
{
	this->confThreshold = conf_Threshold;
	this->nmsThreshold = nms_Threshold;
	this->net = readNet(modelpath);
}

Mat SCRFD::resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw)
	{
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1)
		{
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 0);
		}
		else
		{
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 0);
		}
	}
	else
	{
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void SCRFD::detect(Mat& frame)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat img = this->resize_image(frame, &newh, &neww, &padh, &padw);
	Mat blob;
	blobFromImage(img, blob, 1 / 128.0, Size(this->inpWidth, this->inpHeight), Scalar(127.5, 127.5, 127.5), true, false);
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	/////generate proposals
	vector<float> confidences;
	vector<Rect> boxes;
	vector< vector<int>> landmarks;
	float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
	int n = 0, i = 0, j = 0, k = 0, l = 0;
	for (n = 0; n < 3; n++)   
	{
		int num_grid_x = (int)(this->inpWidth / this->stride[n]);
		int num_grid_y = (int)(this->inpHeight / this->stride[n]);
		float* pdata_score = (float*)outs[n * 3].data;  ///score
		float* pdata_bbox = (float*)outs[n * 3 + 1].data;  ///bounding box
		float* pdata_kps = (float*)outs[n * 3 + 2].data;  ///face landmark
		for (i = 0; i < num_grid_y; i++)
		{
			for (j = 0; j < num_grid_x; j++)
			{
				for (k = 0; k < 4; k++)
				{
					if (pdata_score[0] > this->confThreshold)
					{
						const int xmin = (int)(((j - pdata_bbox[0]) * this->stride[n] - padw) * ratiow);
						const int ymin = (int)(((i - pdata_bbox[1]) * this->stride[n] - padh) * ratioh);
						const int width = (int)((pdata_bbox[2] + pdata_bbox[0])*this->stride[n] * ratiow);
						const int height = (int)((pdata_bbox[3] + pdata_bbox[1])*this->stride[n] * ratioh);
						confidences.push_back(pdata_score[0]);
						boxes.push_back(Rect(xmin, ymin, width, height));
						vector<int> landmark(8, 0);
						for (l = 0; l < 8; l+=2)
						{
							landmark[l] = (int)(((j + pdata_kps[l]) * this->stride[n] - padw) * ratiow);
							landmark[l + 1] = (int)(((i + pdata_kps[l + 1]) * this->stride[n] - padh) * ratioh);
						}
						landmarks.push_back(landmark);
					}
					pdata_score++;
					pdata_bbox += 4;
					pdata_kps += 8;
				}
			}
		}
	}

	vector<int> indices;
	dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
    ////draw bbox and kps
	for (i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(0, 0, 255), 2);
		for (k = 0; k < 8; k+=2)
		{
			circle(frame, Point(landmarks[idx][k], landmarks[idx][k + 1]), 1, Scalar(0, 255, 0), -1);
		}

		//Get the label for the class name and its confidence
		string label = format("%.2f", confidences[idx]);
		//Display the label at the top of the bounding box
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		int top = max(box.y, labelSize.height);
		//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
		putText(frame, label, Point(box.x, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	}
}

int main()
{
	SCRFD mynet("./carddetection_scrfd34gkps.onnx", 0.5, 0.5);
	string imgpath = "./testimgs/card_detection2.jpg";

	Mat srcimg = imread(imgpath);
	mynet.detect(srcimg);

	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}
