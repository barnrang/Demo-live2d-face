#pragma once

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <opencv/highgui.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <algorithm>

#include "LAppPal.hpp"

using namespace dlib;
using namespace std;

class LCameraManager
{
public:
	
	shape_predictor sp;
	frontal_face_detector detector;
	std::vector<cv::Point3d> model_points;

	double norm_height = 640.f;
	double norm_width = 800.f;
	double max_d = 640.f;
	cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << max_d, 0, norm_width/2.f, 0, max_d, norm_height/2.f, 0, 0, 1);
	std::vector<cv::Vec2d> norm_points;
	cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);
	cv::Mat invertYM = cv::Mat::zeros(4, 4, cv::DataType<double>::type);
	cv::Mat invertZM = cv::Mat::zeros(4, 4, cv::DataType<double>::type);
	cv::Mat transformationM = cv::Mat::zeros(4, 4, cv::DataType<double>::type);


	LCameraManager();
	~LCameraManager();

	int RunDebugFaceMark();
	cv::Vec3d GetHeadRotation(cv::VideoCapture& cap);
	cv::Vec3d RotationMatrixToEuler(cv::Mat &R);
	void GetEulerMatrix(cv::Mat &rotm, cv::Vec3d &euler);
	float GetMouthSize();
	float GetLEyeSize();
	float GetREyeSize();
};

