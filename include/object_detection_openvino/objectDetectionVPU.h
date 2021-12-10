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

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

// ROS
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
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

// Typedef for easier readability
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
typedef pcl::PointCloud<pcl::PointXYZRGB> pcloud;

class ObjectDetectionVPU{
	public:
		ObjectDetectionVPU(ros::NodeHandle& node, ros::NodeHandle& node_private);
		~ObjectDetectionVPU();

	private:
		ros::NodeHandle node_, nodePrivate_;
		image_transport::ImageTransport imageTransport_;
		image_transport::SubscriberFilter colorSub_;
		image_transport::Publisher detectionColorPub_;
		message_filters::Subscriber<sensor_msgs::PointCloud2> pointsSub_;
		ros::Publisher detectionInfoPub_, detectionsPub_, markersPub_;
		tf::TransformListener tfListener_;
		typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> SyncPolicyImagePCL;
		message_filters::Synchronizer<SyncPolicyImagePCL> syncImagePCL_;

		Openvino openvino_;
		float thresh_, iouThresh_;
		bool showFPS_, useDepth_, outputImage_;
		std::string deviceTarget_, networkType_;
		std::string modelFileName_, binFileName_, labelFileName_;
		std::string colorFrameId_, colorTopic_, pointCloudTopic_, detectionImageTopic_, detectionInfoTopic_, detectionsTopic_;
		cv::Mat nextFrame_, currFrame_;
		std::vector<std::string> labels_;

		void getParams();
		int getColor(int c, int x, int max);
		void colorImageCallback(sensor_msgs::Image::ConstPtr colorImageMsg);
		void colorPointCallback(sensor_msgs::Image::ConstPtr colorImageMsg, sensor_msgs::PointCloud2::ConstPtr pointsMsg);
		void showHistogram(cv::Mat image, cv::Scalar mean);
		vision_msgs::Detection2D createDetection2DMsg(DetectionObject object, std_msgs::Header header);
		vision_msgs::Detection3D createDetection3DMsg(sensor_msgs::PointCloud2 cloudPC2, pcloud::ConstPtr cloudPCL, DetectionObject object, std_msgs::Header header);
		visualization_msgs::Marker createBBox3dMarker(int id, geometry_msgs::Pose center, geometry_msgs::Vector3 size, float colorRGB[3], std_msgs::Header header);
		visualization_msgs::Marker createLabel3dMarker(int id, std::string label, geometry_msgs::Pose pose, float colorRGB[3], std_msgs::Header header);
		void publishImage(cv::Mat image);
};
#endif
