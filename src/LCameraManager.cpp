#include "LCameraManager.h"



LCameraManager::LCameraManager()
{
	cerr << "Initializing The camera" << endl;
	deserialize("C:\\Users\\DELL GAMING\\Desktop\\CubismSdkForNative-4-r.1\\CubismSdkForNative-4-r.1\\Samples\\shape_predictor_68_face_landmarks.dat") >> sp;

	detector = get_frontal_face_detector();

	model_points.push_back(cv::Point3d(-31.f, 72.f, 86.f));               // l eye
	model_points.push_back(cv::Point3d(31.f, 72.f, 86.f));          // r eye 
	model_points.push_back(cv::Point3d(0.f, 40.f, 114.f));       // nose
	model_points.push_back(cv::Point3d(-20.f, 15.f, 90.f));        // l mouth
	model_points.push_back(cv::Point3d(20.f, 15.f, 90.f)); // r mouth
	model_points.push_back(cv::Point3d(-69.f, 76.f, -2.f));      // l ear
	model_points.push_back(cv::Point3d(69.f, 76.f, -2.f));       // r ear

	for (int i = 0; i < 4; i++) {
		invertYM.at<double>(i, i) = 1;
		invertZM.at<double>(i, i) = 1;
	}

	invertYM.at<double>(0, 0) = 1;
	invertYM.at<double>(1, 1) = -1;
	invertYM.at<double>(2, 2) = 1;
	invertYM.at<double>(3, 3) = 1;

	invertZM.at<double>(0, 0) = 1;
	invertZM.at<double>(1, 1) = 1;
	invertZM.at<double>(2, 2) = -1;
	invertZM.at<double>(3, 3) = 1;

	for (int i = 0; i < 68; i++) {
		norm_points.push_back(cv::Vec2d(0, 0));
	}

	cerr << "Initialize complete" << endl;
}


LCameraManager::~LCameraManager()
{
}

int LCameraManager::RunDebugFaceMark()
{
	cv::VideoCapture cap = cv::VideoCapture(0);
	try
	{
		if (!cap.isOpened())
		{
			cerr << "Unable to connect to camera" << endl;
			return 1;
		}

		image_window win;

		// Load face detection and pose estimation models.
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;

		// Grab and process frames until the main window is closed by the user.
		while (true)
		{
			// Grab a frame
			cv::Mat temp;

			//cap.read(temp);

			//cv::imshow("Output", temp);

			//if (cv::waitKey(5) >= 0)
			//	break;

			//continue;
			
			if (!cap.read(temp))
			{
				break;
			}

			LAppPal::PrintLog("Image size w=%d h=%d", temp.size().width, temp.size().height);

			if (temp.empty()) {
				LAppPal::PrintLog("Empty\n");
			}
			// Turn OpenCV's Mat into something dlib can deal with.  Note that this just
			// wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
			// long as temp is valid.  Also don't do anything to temp that would cause it
			// to reallocate the memory which stores the image as that will make cimg
			// contain dangling pointers.  This basically means you shouldn't modify temp
			// while using cimg.
			cv_image<bgr_pixel> cimg(temp);

			// Detect faces 
			std::vector<rectangle> faces = detector(cimg);
			// Find the pose of each face.
			std::vector<full_object_detection> shapes;
			for (unsigned long i = 0; i < faces.size(); ++i)
				shapes.push_back(sp(cimg, faces[i]));


			if (shapes.size() > 0) {
				//point x = shapes[0].part(0);



				for (int i = 0; i < 68; i++) {
					point tmp = shapes[0].part(i);
					norm_points[i] = cv::Vec2d((double)tmp.x(), (double)tmp.y());
				}

				double norm_scale = (norm_points[30][1] - norm_points[8][1]) / (norm_height / 2.f);
				cv::Vec2d norm_diff = norm_points[30] * norm_scale - cv::Vec2d(norm_width / 2.f, norm_height / 2.f);

				for (int i = 0; i < 68; i++) {
					norm_points[i] = norm_points[i] * norm_scale + norm_diff;
				}

				std::vector<cv::Vec2d> landmark;
				landmark.push_back((norm_points[38] + norm_points[41]) / 2.);
				landmark.push_back((norm_points[43] + norm_points[46]) / 2.);
				landmark.push_back(norm_points[33]);
				landmark.push_back(norm_points[48]);
				landmark.push_back(norm_points[54]);
				landmark.push_back(norm_points[0]);
				landmark.push_back(norm_points[16]);

				cv::Mat rotation_vector;
				cv::Mat translation_vector;

				cv::solvePnP(model_points, landmark, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

				//cv::Vec3d euler_angle = cv::Vec3d(rotation_vector.at<double>(0, 0), rotation_vector.at<double>(0, 1), rotation_vector.at<double>(0, 2)) * 180.f / pi;
				cv::Vec3d euler_angle;
				cv::Mat rotm;
				cv::Rodrigues(rotation_vector, rotm);

				GetEulerMatrix(rotm, euler_angle);


				GetLEyeSize();
				GetREyeSize();
				LAppPal::PrintLog("euler angle %f %f %f\n", euler_angle[0], euler_angle[1], euler_angle[2]);
			}

			//cerr << sp.num_parts() << endl;


			//if (shapes.size() > 0) {
			//	for (int i = 0; i < 68; i++) {
			//		long x = shapes[0].part(i).x();
			//		long y = shapes[0].part(i).y();
			//		cv::circle(temp, cv::Point(x, y), 2, cv::Scalar(0, 255, 0));
			//		cv::putText(temp, to_string(i), cv::Point(x, y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 0), 1.0);
			//	}

			//	cv::imwrite("image.jpg", temp);
			//	break;
			//}



			// Display it all on the screen
			win.clear_overlay();
			//if (shapes.size() > 0) {
			//	draw_circle(cimg, point(shapes[0].part(38).x(), shapes[0].part(38).y()));
			//}
			win.set_image(cimg);

			win.add_overlay(render_face_detections(shapes));
		}
	}
	catch (serialization_error& e)
	{
		cout << "You need dlib's default face landmarking model file to run this example." << endl;
		cout << "You can get it from the following URL: " << endl;
		cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
		cout << endl << e.what() << endl;
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}
}

cv::Vec3d LCameraManager::GetHeadRotation(cv::VideoCapture& cap)
{
	try
	{
		if (!cap.isOpened())
		{
			cerr << "Unable to connect to camera" << endl;
			return 1;
		}

		// Load face detection and pose estimation models.

		// Grab and process frames until the main window is closed by the user.
			// Grab a frame
		cv::Mat temp;

		//cap.read(temp);

		//cv::imshow("Output", temp);

		//if (cv::waitKey(5) >= 0)
		//	break;

		//continue;

		if (!cap.read(temp))
		{
			return cv::Vec3d();
		}

		LAppPal::PrintLog("Image size w=%d h=%d", temp.size().width, temp.size().height);

		if (temp.empty()) {
			LAppPal::PrintLog("Empty\n");
		}
		// Turn OpenCV's Mat into something dlib can deal with.  Note that this just
		// wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
		// long as temp is valid.  Also don't do anything to temp that would cause it
		// to reallocate the memory which stores the image as that will make cimg
		// contain dangling pointers.  This basically means you shouldn't modify temp
		// while using cimg.
		cv_image<bgr_pixel> cimg(temp);

		// Detect faces 
		std::vector<rectangle> faces = detector(cimg);
		// Find the pose of each face.
		std::vector<full_object_detection> shapes;
		for (unsigned long i = 0; i < faces.size(); ++i)
			shapes.push_back(sp(cimg, faces[i]));


		if (shapes.size() > 0) {
			//point x = shapes[0].part(0);



			for (int i = 0; i < 68; i++) {
				point tmp = shapes[0].part(i);
				norm_points[i] = cv::Vec2d((double)tmp.x(), (double)tmp.y());
			}

			double norm_scale = (norm_points[30][1] - norm_points[8][1]) / (norm_height / 2.f);
			cv::Vec2d norm_diff = norm_points[30] * norm_scale - cv::Vec2d(norm_width / 2.f, norm_height / 2.f);

			for (int i = 0; i < 68; i++) {
				norm_points[i] = norm_points[i] * norm_scale + norm_diff;
			}

			std::vector<cv::Vec2d> landmark;
			landmark.push_back((norm_points[38] + norm_points[41]) / 2.);
			landmark.push_back((norm_points[43] + norm_points[46]) / 2.);
			landmark.push_back(norm_points[33]);
			landmark.push_back(norm_points[48]);
			landmark.push_back(norm_points[54]);
			landmark.push_back(norm_points[0]);
			landmark.push_back(norm_points[16]);

			cv::Mat rotation_vector;
			cv::Mat translation_vector;

			cv::solvePnP(model_points, landmark, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

			//cv::Vec3d euler_angle = cv::Vec3d(rotation_vector.at<double>(0, 0), rotation_vector.at<double>(0, 1), rotation_vector.at<double>(0, 2)) * 180.f / pi;
			cv::Vec3d euler_angle;
			cv::Mat rotm;
			cv::Rodrigues(rotation_vector, rotm);

			GetEulerMatrix(rotm, euler_angle);

			LAppPal::PrintLog("euler angle %f %f %f\n", euler_angle[0], euler_angle[1], euler_angle[2]);
			return euler_angle;
		}

	}
	catch (serialization_error& e)
	{
		cout << "You need dlib's default face landmarking model file to run this example." << endl;
		cout << "You can get it from the following URL: " << endl;
		cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
		cout << endl << e.what() << endl;
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}


	return cv::Vec3d(-999,-1,-1);
}



cv::Vec3d LCameraManager::RotationMatrixToEuler(cv::Mat & R)
{
	//assert
	double sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

	bool singular = sy < 1e-6; // If

	double x, y, z;
	if (!singular)
	{
		x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
		y = atan2(-R.at<double>(2, 0), sy);
		z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
	}
	else
	{
		x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
		y = atan2(-R.at<double>(2, 0), sy);
		z = 0;
	}
	return cv::Vec3d(x, y, z);
}

void LCameraManager::GetEulerMatrix(cv::Mat &rotm, cv::Vec3d &euler)
{
	cv::Mat cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ;
	double* _r = rotm.ptr<double>();
	double projMatrix[12] = { _r[0],_r[1],_r[2],0,
						  _r[3],_r[4],_r[5],0,
						  _r[6],_r[7],_r[8],0 };

	cv::decomposeProjectionMatrix(cv::Mat(3, 4, CV_64FC1, projMatrix),
		cameraMatrix,
		rotMatrix,
		transVect,
		rotMatrixX,
		rotMatrixY,
		rotMatrixZ,
		euler);
}

float LCameraManager::GetMouthSize()
{
	float ratio = abs(norm_points[62][1] - norm_points[66][1]) / abs(norm_points[51][1] - norm_points[62][1]);
	//LAppPal::PrintLog("mouth open ratio %f\n", ratio);
	if (ratio > 0 && ratio < 1.7) return 0.;
	if (ratio >= 1.7 && ratio < 2) return 0.3;
	if (ratio >= 2 && ratio < 3) return 0.5;
	if (ratio >= 3 && ratio < 4) return 0.7;
	if (ratio >= 4) return 1.0;
	if (ratio <= 0) return 0;
	return ratio;

	//float size = abs(norm_points[48][0] - norm_points[54][0]) / (abs(norm_points[31][0] - norm_points[35][0]) * 1.8f);
	//size -= 1;
	//size *= 4.f;

	//return std::clamp(size, -1.0f, 1.0f);
}

float LCameraManager::GetLEyeSize()
{
	//float ratio = abs(norm_points[43][1] - norm_points[47][1]) / abs(norm_points[43][0] - norm_points[47][0]);
	float ratio = (norm_points[44][1] - norm_points[46][1]) / (norm_points[42][0] - norm_points[45][0]);
	LAppPal::PrintLog("LEye open ratio %f\n", ratio);
	
	return ratio;
}

float LCameraManager::GetREyeSize()
{
	float ratio = abs(norm_points[38][1] - norm_points[40][1]) / abs(norm_points[36][0] - norm_points[39][0]);
	LAppPal::PrintLog("REye open ratio %f\n", ratio);

	return ratio;
}
