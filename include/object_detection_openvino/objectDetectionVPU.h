/*
 * OBJECT DETECTION VPU CLASS
 *
 * Copyright (c) 2020-2021 Alberto José Tudela Roldán <ajtudela@gmail.com>
 * 
 * This file is part of object_detection_openvino project.
 * 
 * All rights reserved.
 *
 */

#ifndef OBJECT_DETECTION_VPU_H
#define OBJECT_DETECTION_VPU_H

// C++
#include <chrono>
#include <string>
#include <vector>

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
#include <object_detection_openvino/openvino.h>

// OpenCV
#include <cv_bridge/cv_bridge.h>

#define COCO_CLASSES		80

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

class ObjectDetectionVPU{
	public:
		ObjectDetectionVPU(ros::NodeHandle& node, ros::NodeHandle& node_private);
		~ObjectDetectionVPU();

	private:
		ros::NodeHandle node_, nodePrivate_;
		image_transport::ImageTransport imageTransport_;
		image_transport::SubscriberFilter colorSub_, depthSub_;
		image_transport::Publisher detectionColorPub_;
		ros::Subscriber depthInfoSub_;
		ros::Publisher detectionInfoPub_, detectionsPub_, markersPub_;

		typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicyTwoImage;
		message_filters::Synchronizer<SyncPolicyTwoImage> syncTwoImage_;

		Openvino openvino_;
		float fx_, fy_, cx_, cy_;
		float iouThres_;
		bool showFPS_, useDepth_, outputImage_;
		std::string networkType_;
		std::string modelFileName_, binFileName_, labelFileName_;
		std::string colorFrameId_, colorTopic_, depthInfoTopic_, depthTopic_, detectionImageTopic_, detectionInfoTopic_, detectionsTopic_, deviceTarget_;
		std::vector<std::string> labels_;


		void getParams();
		void depthInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& infoMsg);
		void oneImageCallback(sensor_msgs::Image::ConstPtr colorImageMsg);
		void twoImageCallback(sensor_msgs::Image::ConstPtr colorImageMsg, sensor_msgs::Image::ConstPtr depthImageMsg);
		void cameraCallback(const std::vector<sensor_msgs::Image::ConstPtr>& imageMsg);
		void showHistogram(cv::Mat image, cv::Scalar mean);
		vision_msgs::Detection2D createDetection2DMsg(DetectionObject object, std_msgs::Header header);
		vision_msgs::Detection3D createDetection3DMsg(cv_bridge::CvImagePtr depthImage, DetectionObject object, std_msgs::Header header);
		visualization_msgs::Marker createBBox3dMarker(int id, geometry_msgs::Pose center, geometry_msgs::Vector3 size, float colorRGB[3], std_msgs::Header header);
		visualization_msgs::Marker createLabel3dMarker(int id, std::string label, geometry_msgs::Pose pose, float colorRGB[3], std_msgs::Header header);
		void publishImage(cv::Mat image);
};
#endif
