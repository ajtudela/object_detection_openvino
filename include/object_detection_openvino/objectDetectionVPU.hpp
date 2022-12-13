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

#ifndef OBJECT_DETECTION_OPENVINO__OBJECT_DETECTION_VPU_HPP_
#define OBJECT_DETECTION_OPENVINO__OBJECT_DETECTION_VPU_HPP_

// C++
#include <chrono>
#include <string>
#include <vector>

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

// ROS
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"
#include "tf2_ros/buffer.h"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "image_transport/image_transport.hpp"
#include "image_transport/subscriber_filter.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/time_synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "visualization_msgs/msg/marker_array.hpp"
#include "vision_msgs/msg/vision_info.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"

// Object detection openvino
#include "object_detection_openvino/detectionObject.hpp"
#include "object_detection_openvino/openvino.hpp"

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
class ObjectDetectionVPU : public rclcpp::Node{
	public:

		/// Constructor which initialize the subscribers, the publishers and the inference engine.
		ObjectDetectionVPU(std::string node_name);

		/// Delete all parameteres of the node.
		~ObjectDetectionVPU();

	private:

		/// Subscriber to a color image.
		image_transport::SubscriberFilter color_sub_;

		/// Publisher of the image with the bounding boxes of the detected objects.
		image_transport::Publisher detection_color_pub_;

		/// Subscriber to a pointcloud. 
		message_filters::Subscriber<sensor_msgs::msg::PointCloud2> points_sub_;

		/// Publisher of the meta-information about the vision pipeline.
		rclcpp::Publisher<vision_msgs::msg::VisionInfo>::SharedPtr detection_info_pub_;

		/// Publisher of the aray with the detected objects in 2 dimensions.
		rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detections_2d_pub_;

		/// Publisher of the aray with the detected objects in 3 dimensions.
		rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr detections_3d_pub_;

		/// Publisher of markers with the bounding boxes and labels of the detected objects.
		rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;

		/// The buffer of the transformations tree.
		std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

		/// Typedef for a ApproximateTime policy between an image message and a pointcloud message.
		typedef message_filters::sync_policies::ApproximateTime<
			sensor_msgs::msg::Image, 
			sensor_msgs::msg::PointCloud2> ApproximatePolicy;

		/// Typedef for synchronizer of the image message and the pointcloud message.
		typedef message_filters::Synchronizer<ApproximatePolicy> ApproximateSync;

		/// Synchronizer of the image message and the pointcloud message.
		std::shared_ptr<ApproximateSync> approximate_sync_;

		/// Intel OpenVino framework.
		Openvino openvino_;

		/// Value below which objects will be discarded.
		float thresh_;

		/// Value of the intersection over union threshold in a range between 0 and 1.
		float iou_thresh_;

		/// Flag to enable / disable the number of FPS in the image.
		bool show_fps_;

		/// Flag to enable / disable the use of te pointcloud to perform depth analysis.
		bool use_depth_;

		/// Type of the neural network.
		std::string network_type_;

		/// Name of the device to load the neural network into.
		std::string device_target_;

		/// The filename of the model in XML format.
		std::string model_filename_;

		/// The filename of the model in BIN format.
		std::string bin_filename_;

		/// The filename of the labels used in the model.
		std::string label_filename_;

		/// Frame of the color image.
		std::string color_frame_;

		/// Frame of the camera.
		std::string camera_frame_;

		/// Topic of the color image.
		std::string color_topic_;

		/// Topic of the pointcloud.
		std::string pointcloud_topic_;

		/// Topic of the image with the bounding boxes of the detected objects.
		std::string detection_image_topic_;

		/// Topic of the information about the vision pipeline.
		std::string detection_info_topic_;

		/// Topic of the detected objects in 2 dimensions.
		std::string detections_2d_topic_;

		/// Topic of the detected objects in 3 dimensions.
		std::string detections_3d_topic_;

		/// Image to be processed.
		cv::Mat frame_;

		/// Class labels of the neural network.
		std::vector<std::string> labels_;

		/**
		 * @brief Update the parameters of the node.
		 * 
		 */
		void get_params();

		/**
		 * @brief Get color of the class.
		 * 
		 * @param c Integer with the channel of the color.
		 * @param x Offset value.
		 * @param max Maximum number of classes.
		 * @return The color of the class. 
		 */
		int get_color(int c, int x, int max);

		/**
		 * @brief Publish the meta-information about the vision pipeline when a subscriber connect to it.
		 */
		void connect_info_callback();

		/**
		 * @brief Receive an image, perform 2d object detection on it and publish the detected objects.
		 * 
		 * @param color_image_msg A message with an image to be processed.
		 */
		void color_image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & color_image_msg);

		/**
		 * @brief Receive an image and a pointcloud, perform 2d object detection on it and publish the detected objects.
		 * 
		 * 
		 * @param color_image_msg A message with an image to be processed.
		 * @param points_msg A message with a pointcloud to be processed.
		 */
		void color_point_callback(const sensor_msgs::msg::Image::ConstSharedPtr & color_image_msg, 
								const sensor_msgs::msg::PointCloud2::ConstSharedPtr & points_msg);

		/**
		 * @brief Show the histogram of the image.
		 * 
		 * @param image The image from the sensor.
		 * @param mean The mean value of the pixels.
		 */
		void show_histogram(cv::Mat image, cv::Scalar mean);

		/**
		 * @brief Create a Detection2D message with the detected object.
		 * 
		 * @param[in] object The detected object.
		 * @param[in] header The header of the detected object.
		 * @param[out] detection2D The detection2D message with the detected object.
		 * @return True if the message was created correctly, false otherwise.
		 */
		bool create_detection_2d_msg(DetectionObject object, 
									std_msgs::msg::Header header, 
									vision_msgs::msg::Detection2D& detection_2d);

		/**
		 * @brief Create a detection3D message with the detected object.
		 * 
		 * @param[in] object The detected object.
		 * @param[in] header The header of the detected object.
		 * @param[in] cloud_pc2 The pointcloud of the detected object in sensor_msgs::PointCloud2 format.
		 * @param[in] cloud_pcl The pointcloud of the detected object in pcloud format.
		 * @param[out] detection3D The detection3D message with the detected object.
		 * @return True if the message was created correctly, false otherwise.
		 */
		bool create_detection_3d_msg(DetectionObject object, 
									std_msgs::msg::Header header, 
									const sensor_msgs::msg::PointCloud2& cloud_pc2, 
									pcloud::ConstPtr cloud_pcl, 
									vision_msgs::msg::Detection3D& detection_3d);

		/**
		 * @brief Create a 3D marker with the bounding box of the object to be shown in Rviz.
		 * 
		 * @param[in] id The numeric identifier of the marker.
		 * @param[in] header The header of the marker.
		 * @param[in] color_rgb The color of the marker.
		 * @param[in] bbox The dimensions of the bounding box surrounding the object.
		 * @param[out] marker The 3d marker with the bounding of the detected object.
		 * @return True if the marker was created correctly, false otherwise.
		 */
		bool create_bbox_3d_marker(int id, 
								std_msgs::msg::Header header, 
								float color_rgb[3], 
								vision_msgs::msg::BoundingBox3D bbox, 
								visualization_msgs::msg::Marker& marker);

		/**
		 * @brief Create a 3d marker with the label of the object to be shown in Rviz.
		 * 
		 * @param[in] id The numeric identifier of the marker.
		 * @param[in] header The header of the marker.
		 * @param[in] color_rgb The color of the marker.
		 * @param[in] bbox The dimensions of the bounding box surrounding the object.
		 * @param[in] label The label class of the detected object.
		 * @param[out] marker The 3d marker with the label of the detected object.
		 * @return True if the marker was created correctly, false otherwise.
		 */
		bool create_label_3d_marker(int id, 
								std_msgs::msg::Header header, 
								float color_rgb[3], 
								vision_msgs::msg::BoundingBox3D bbox, 
								std::string label, 
								visualization_msgs::msg::Marker& marker);

		/**
		 * @brief Publish the image with the bounding boxes of the detected objects.
		 * 
		 * @param image The image with the bounding boxes of the detected objects.
		 */
		void publish_image(cv::Mat image);
};
#endif // OBJECT_DETECTION_OPENVINO__OBJECT_DETECTION_VPU_HPP_
