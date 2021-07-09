/*
 * OBJECT DETECTION OPENVINO CLASS
 *
 * Copyright (c) 2020-2021 Alberto José Tudela Roldán <ajtudela@gmail.com>
 * 
 * This file is part of object_detection_openvino project.
 * 
 * All rights reserved.
 *
 */

#ifndef OBJECT_DETECTION_OPENVINO_H
#define OBJECT_DETECTION_OPENVINO_H

// C++
#include <chrono>
#include <vector>
#include <string>

// ROS
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <visualization_msgs/Marker.h>
#include <vision_msgs/Detection2D.h>
#include <vision_msgs/Detection3D.h>

#include <object_detection_openvino/detectionObject.h>
#include <object_detection_openvino/yoloParams.h>

// OpenCV
#include <cv_bridge/cv_bridge.h>

// OpenVINO
#include <inference_engine.hpp>
#include <samples/ocv_common.hpp>

#ifdef WITH_EXTENSIONS
    #include <ext_list.hpp>
#endif

#define COCO_CLASSES		80

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

class ObjectDetectionOpenvino{
	public:
		ObjectDetectionOpenvino(ros::NodeHandle& node, ros::NodeHandle& node_private);
		~ObjectDetectionOpenvino();

	private:
		// ROS related
		ros::NodeHandle node_, nodePrivate_;
		image_transport::ImageTransport imageTransport_;
		image_transport::SubscriberFilter colorSub_, depthSub_;
		image_transport::Publisher detectionColorPub_;
		ros::Subscriber infoSub_;
		ros::Publisher detectionInfoPub_, detectionsPub_, markersPub_;

		typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicyTwoImage;
		message_filters::Synchronizer<SyncPolicyTwoImage> syncTwoImage_;

		std::string inputName_, networkType_;
		std::string modelFileName_, binFileName_, labelFileName_;
		std::string colorFrameId_, infoTopic_, colorTopic_, depthTopic_, detectionImageTopic_, detectionInfoTopic_, detectionsTopic_, deviceTarget_;
		std::vector<std::string> labels_;

		float fx_, fy_, cx_, cy_;
		bool showFPS_, useDepth_, outputImage_;

		void getParams();
		void infoCallback(const sensor_msgs::CameraInfo::ConstPtr& infoMsg);
		void oneImageCallback(sensor_msgs::Image::ConstPtr colorImageMsg);
		void twoImageCallback(sensor_msgs::Image::ConstPtr colorImageMsg, sensor_msgs::Image::ConstPtr depthImageMsg);
		void cameraCallback(const std::vector<sensor_msgs::Image::ConstPtr>& imageMsg);
		void showHistogram(cv::Mat image, cv::Scalar mean);
		vision_msgs::Detection2D createDetection2DMsg(DetectionObject object, std_msgs::Header header);
		vision_msgs::Detection3D createDetection3DMsg(cv_bridge::CvImagePtr depthImage, DetectionObject object, std_msgs::Header header);
		visualization_msgs::Marker createBBox3dMarker(int id, geometry_msgs::Pose center, geometry_msgs::Vector3 size, float colorRGB[3], std_msgs::Header header);
		visualization_msgs::Marker createLabel3dMarker(int id, std::string label, geometry_msgs::Pose pose, float colorRGB[3], std_msgs::Header header);
		void publishImage(cv::Mat image);

		// OpenVino related
		std::map<std::string, YoloParams> yoloParams_;
		cv::Mat nextFrame_, currFrame_;
		InferenceEngine::InferRequest::Ptr asyncInferRequestCurr_, asyncInferRequestNext_;
		InferenceEngine::OutputsDataMap outputInfo_;
		InferenceEngine::InputsDataMap inputInfo_;
		InferenceEngine::CNNNetwork cnnNetwork_;
		InferenceEngine::Core core_;
		float thresh_, iouThresh_;

		int getColor(int c, int x, int max);
		static int entryIndex(int side, int lcoords, int lclasses, int location, int entry);
		double intersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2);
		void frameToBlob(const cv::Mat &frame, InferenceEngine::InferRequest::Ptr &inferRequest, const std::string &inputName, bool autoResize = false);
		void parseSSDOutput(const InferenceEngine::Blob::Ptr &blob, const unsigned long height, const unsigned long width, const float threshold, std::vector<DetectionObject> &objects);
		void parseYOLOV3Output(const InferenceEngine::CNNNetwork &cnnNetwork, const std::string &outputName, const InferenceEngine::Blob::Ptr &blob, const unsigned long resizedImgH, const unsigned long resizedImgW, const unsigned long originalImgH, const unsigned long originalImgW, const float threshold, std::vector<DetectionObject> &objects);
};
#endif
