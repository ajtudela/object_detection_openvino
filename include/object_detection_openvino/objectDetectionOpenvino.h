/*
 * OBJECT DETECTION OPENVINO ROS NODE
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
#include <std_srvs/Empty.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

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

#define YOLO_SCALE_13		13
#define YOLO_SCALE_26		26
#define YOLO_SCALE_52		52
#define COCO_CLASSES		80

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

class ObjectDetectionOpenvino{
	public:
		ObjectDetectionOpenvino(ros::NodeHandle& node, ros::NodeHandle& node_private);
		~ObjectDetectionOpenvino();

	private:
		// ROS related
		ros::NodeHandle node_, nodePrivate_;
		ros::ServiceServer paramsSrv_;
		image_transport::ImageTransport imageTransport_;
		image_transport::SubscriberFilter colorSub_, depthSub_;
		image_transport::Publisher detectionColorPub_;
		ros::Subscriber infoSub_;
		ros::Publisher detectionInfoPub_, detection2DPub_;

		typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicyTwoImage;
		message_filters::Synchronizer<SyncPolicyTwoImage> syncTwoImage_;

		std::string inputName_, networkType_;
		std::string modelFileName_, binFileName_, labelFileName_;
		std::string colorFrameId_, infoTopic_, colorTopic_, depthTopic_, detectionImageTopic_, detectionInfoTopic_, detection2DTopic_, deviceTarget_;
		std::vector<std::string> labels_;

		float fx_, fy_, cx_, cy_;
		bool showFPS_, useDepth_, outputImage_;

		void initialize() { std_srvs::Empty empt; updateParams(empt.request, empt.response); }
		bool updateParams(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res);
		void infoCallback(const sensor_msgs::CameraInfo::ConstPtr& infoMsg);
		void oneImageCallback(sensor_msgs::Image::ConstPtr colorImageMsg);
		void twoImageCallback(sensor_msgs::Image::ConstPtr colorImageMsg, sensor_msgs::Image::ConstPtr depthImageMsg);
		void cameraCallback(const std::vector<sensor_msgs::Image::ConstPtr>& imageMsg);
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
		void parseSSDOutput(const InferenceEngine::CNNLayerPtr &layer, const InferenceEngine::Blob::Ptr &blob, const unsigned long height, const unsigned long width, const float threshold, std::vector<DetectionObject> &objects);
		void parseYOLOV3Output(const YoloParams &params, const std::string &outputName, const InferenceEngine::Blob::Ptr &blob, const unsigned long resizedImgH, const unsigned long resizedImgW, const unsigned long originalImgH, const unsigned long originalImgW, const float threshold, std::vector<DetectionObject> &objects);
};
#endif
