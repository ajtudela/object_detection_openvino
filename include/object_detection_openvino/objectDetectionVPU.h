/*
 * OBJECT DETECTION VPU CLASS
 *
 * Copyright (c) 2020-2022 Alberto José Tudela Roldán <ajtudela@gmail.com>
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


/// The number of the labels in the COCO database.
#define COCO_CLASSES		80

/// Typedef for easier readability.
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

/// Typedef for easier readability.
typedef pcl::PointCloud<pcl::PointXYZRGB> pcloud;

/**
 * @brief Implmentation of the interface between different neural networks frameworks and ROS.
 * 
 */
class ObjectDetectionVPU{
	public:

		/// Constructor which initialize the subscribers, the publishers and the inference engine .
		ObjectDetectionVPU(ros::NodeHandle& node, ros::NodeHandle& node_private);

		/// Delete all parameteres of the node.
		~ObjectDetectionVPU();

	private:

		/// Public ROS node handler.
		ros::NodeHandle node_;

		/// Private ROS node handler.
		ros::NodeHandle nodePrivate_;

		/// The received image.
		image_transport::ImageTransport imageTransport_;

		/// Subscriber to a color image.
		image_transport::SubscriberFilter colorSub_;

		/// Publisher of the image with the bounding boxes of the detected objects.
		image_transport::Publisher detectionColorPub_;

		/// Subscriber to a pointcloud. 
		message_filters::Subscriber<sensor_msgs::PointCloud2> pointsSub_;

		/// Publisher of the meta-information about the vision pipeline.
		ros::Publisher detectionInfoPub_;

		/// Publisher of the aray with the detected objects in 2 dimensions.
		ros::Publisher detections2DPub_;

		/// Publisher of the aray with the detected objects in 3 dimensions.
		ros::Publisher detections3DPub_;

		/// Publisher of markers with the bounding boxes and labels of the detected objects.
		ros::Publisher markersPub_;

		/// Listener of the transformations tree.
		tf::TransformListener tfListener_;

		/// Typedef for a ApproximateTime policy between an image message and a pointcloud message.
		typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> SyncPolicyImagePCL;

		/// Synchronizer of the image message and the pointcloud message.
		message_filters::Synchronizer<SyncPolicyImagePCL> syncImagePCL_;

		/// Intel OpenVino framework.
		Openvino openvino_;

		/// Value below which objects will be discarded.
		float thresh_;

		/// Value of the intersection over union threshold in a range between 0 and 1.
		float iouThresh_;

		/// Flag to enable / disable the number of FPS in the iamge.
		bool showFPS_;

		/// Flag to enable / disable the use of te pointcloud to perform depth analysis.
		bool useDepth_;

		/// Type of the neural network.
		std::string networkType_;

		/// Name of the device to load the neural network into.
		std::string deviceTarget_;

		/// The filename of the model in XML format.
		std::string modelFileName_;

		/// The filename of the model in BIN format.
		std::string binFileName_;

		/// The filename of the labels used in the model.
		std::string labelFileName_;

		/// Frame identifier of the color image.
		std::string colorFrameId_;

		/// Frame identifier of the camera.
		std::string cameraFrameId_;

		/// Topic of the color image.
		std::string colorTopic_;

		/// Topic of the pointcloud.
		std::string pointCloudTopic_;

		/// Topic of the image with the bounding boxes of the detected objects.
		std::string detectionImageTopic_;

		/// Topic of the information about the vision pipeline.
		std::string detectionInfoTopic_;

		/// Topic of the detected objects in 2 dimensions.
		std::string detections2DTopic_;

		/// Topic of the detected objects in 3 dimensions.
		std::string detections3DTopic_;

		/// Next image to be processed.
		cv::Mat nextFrame_;

		/// Curent image to be processed.
		cv::Mat currFrame_;

		/// Class labels of the neural network.
		std::vector<std::string> labels_;

		/**
		 * @brief Update the parameters of the node.
		 * 
		 */
		void getParams();

		/**
		 * @brief Get color of the class.
		 * 
		 * @param c Integer with the channel of the color.
		 * @param x Offset value.
		 * @param max Maximum number of classes.
		 * @return The color of the class. 
		 */
		int getColor(int c, int x, int max);

		/**
		 * @brief Publish the meta-information about the vision pipeline when a subscriber connect to it.
		 * 
		 * @param pub The publisher of the information.
		 */
		void connectInfoCallback(const ros::SingleSubscriberPublisher& pub);

		/**
		 * @brief Receive an image, perform 2d object detection on it and publish the detected objects.
		 * 
		 * @param colorImageMsg A message with an image to be processed.
		 */
		void colorImageCallback(const sensor_msgs::Image::ConstPtr& colorImageMsg);

		/**
		 * @brief Receive an image and a pointcloud, perform 2d object detection on it and publish the detected objects.
		 * 
		 * 
		 * @param colorImageMsg A message with an image to be processed.
		 * @param pointsMsg A message with a pointcloud to be processed.
		 */
		void colorPointCallback(const sensor_msgs::Image::ConstPtr& colorImageMsg, const sensor_msgs::PointCloud2::ConstPtr& pointsMsg);

		/**
		 * @brief Show the histogram of the image.
		 * 
		 * @param image The image from the sensor.
		 * @param mean The mean value of the pixels.
		 */
		void showHistogram(cv::Mat image, cv::Scalar mean);

		/**
		 * @brief Create a Detection2D message with the detected object.
		 * 
		 * @param[in] object The detected object.
		 * @param[in] header The header of the detected object.
		 * @param[out] detection2D The detection2D message with the detected object.
		 * @return True if the message was created correctly, false otherwise.
		 */
		bool createDetection2DMsg(DetectionObject object, std_msgs::Header header, vision_msgs::Detection2D& detection2D);

		/**
		 * @brief Create a detection3D message with the detected object.
		 * 
		 * @param[in] object The detected object.
		 * @param[in] header The header of the detected object.
		 * @param[in] cloudPC2 The pointcloud of the detected object in sensor_msgs::PointCloud2 format.
		 * @param[in] cloudPCL The pointcloud of the detected object in pcloud format.
		 * @param[out] detection3D The detection3D message with the detected object.
		 * @return True if the message was created correctly, false otherwise.
		 */
		bool createDetection3DMsg(DetectionObject object, std_msgs::Header header, const sensor_msgs::PointCloud2& cloudPC2, pcloud::ConstPtr cloudPCL, vision_msgs::Detection3D& detection3D);

		/**
		 * @brief Create a 3D marker with the bounding box of the object to be shown in Rviz.
		 * 
		 * @param[in] id The numeric identifier of the marker.
		 * @param[in] header The header of the marker.
		 * @param[in] colorRGB The color of the marker.
		 * @param[in] bbox The dimensions of the bounding box surrounding the object.
		 * @param[out] marker The 3d marker with the bounding of the detected object.
		 * @return True if the marker was created correctly, false otherwise.
		 */
		bool createBBox3DMarker(int id, std_msgs::Header header, float colorRGB[3], vision_msgs::BoundingBox3D bbox, visualization_msgs::Marker& marker);

		/**
		 * @brief Create a 3d marker with the label of the object to be shown in Rviz.
		 * 
		 * @param[in] id The numeric identifier of the marker.
		 * @param[in] header The header of the marker.
		 * @param[in] colorRGB The color of the marker.
		 * @param[in] bbox The dimensions of the bounding box surrounding the object.
		 * @param[in] label The label class of the detected object.
		 * @param[out] marker The 3d marker with the label of the detected object.
		 * @return True if the marker was created correctly, false otherwise.
		 */
		bool createLabel3DMarker(int id, std_msgs::Header header, float colorRGB[3], vision_msgs::BoundingBox3D bbox, std::string label, visualization_msgs::Marker& marker);

		/**
		 * @brief Publish the image with the bounding boxes of the detected objects.
		 * 
		 * @param image The image with the bounding boxes of the detected objects.
		 */
		void publishImage(cv::Mat image);
};
#endif
