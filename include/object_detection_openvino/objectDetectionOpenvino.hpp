/*
 * OBJECT DETECTION OPENVINO ROS NODE
 *
 * Copyright (c) 2020 Alberto José Tudela Roldán <ajtudela@gmail.com>
 * 
 * This file is part of object_detection_openvino project.
 * 
 * All rights reserved.
 *
 */

#ifndef OBJECT_DETECTION_OPENVINO_H
#define OBJECT_DETECTION_OPENVINO_H

// C++
#include <math.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>

// ROS
#include <ros/ros.h>
#include <std_srvs/Empty.h>
#include <std_msgs/Header.h>
#include <std_msgs/Int8.h>
#include <geometry_msgs/Pose.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <visualization_msgs/MarkerArray.h>
#include <object_detection_openvino/BoundingBoxArray.h>
#include <object_detection_openvino/BoundingBox3dArray.h>

// OpenCv
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
#include <cv_bridge/cv_bridge.h>

// OpenVINO
#include <inference_engine.hpp>

#ifdef WITH_EXTENSIONS
    #include <ext_list.hpp>
#endif

#define YOLO_SCALE_13		13
#define YOLO_SCALE_26		26
#define YOLO_SCALE_52		52
#define COCO_CLASSES		80

float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

class ObjectDetectionOpenvino{
	public:
		ObjectDetectionOpenvino(ros::NodeHandle& node, ros::NodeHandle& node_private);
		~ObjectDetectionOpenvino();

		struct DetectionObject{
			int xmin, ymin, xmax, ymax, classId, id;
			float xMin3d, yMin3d, zMin3d, xMax3d, yMax3d, zMax3d;
			float confidence;
			std::string Class;

			DetectionObject(double x, double y, double h, double w, int classId, std::string Class, float confidence, float h_scale, float w_scale);
			DetectionObject(double x, double y, double h, double w, int classId, std::string Class, float confidence);
			bool operator<(const DetectionObject &s2) const;
			object_detection_openvino::BoundingBox BoundingBox(int id);
			object_detection_openvino::BoundingBox3d BoundingBox3d(int id);
		};

		ros::NodeHandle node_, nodePrivate_;
		ros::ServiceServer paramsSrv_;
		image_transport::ImageTransport imageTransport_;
		image_transport::SubscriberFilter imageSubscriberFilter_, depthSubscriberFilter_;
		message_filters::Subscriber<sensor_msgs::CameraInfo> cameraInfoSubscriber_;
		typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> MySyncPolicy;
		message_filters::Synchronizer<MySyncPolicy> sync;
		ros::Publisher boundingBoxesPublisher_, markerPublisher_, boundingBoxes3dPublisher_;
		image_transport::Publisher detectionImagePublisher_;
		
		std_msgs::Header imageHeader_;

		std::string inputName_, outputName_, networkType_;
		std::string modelFileName_, binFileName_, labelFileName_;
		std::string colorFrameId_, depthFrameId_, infoTopic_, colorTopic_, depthTopic_, imageDetectedTopic_, boundingBoxTopic_, boundingBox3dTopic_, deviceTarget_;
		std::vector<std::string> labels_;
		
		int detectionId_;
		float thresh_, iouThresh_;
		bool showFPS_, outputImage_, outputBoxes_, outputMarkers_;
		
		cv::Mat nextFrame_, currFrame_;
		InferenceEngine::InferRequest::Ptr async_infer_request_curr_, async_infer_request_next_;
		InferenceEngine::OutputsDataMap outputInfo_;
		InferenceEngine::InputsDataMap inputInfo_;
		InferenceEngine::CNNNetReader netReader_;
		InferenceEngine::Core core_;

		typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
		
		void initialize() { std_srvs::Empty empt; updateParams(empt.request, empt.response); }
		bool updateParams(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res);
		int getColor(int c, int x, int max);
		static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry);
		double IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2);
		void ParseYOLOV3Output(const InferenceEngine::CNNLayerPtr &layer, const InferenceEngine::Blob::Ptr &blob, const unsigned long resizedImgH, const unsigned long resizedImgW, const unsigned long originalImgH, const unsigned long originalImgW, const float threshold, std::vector<DetectionObject> &objects);
		void ParseSSDOutput(const InferenceEngine::CNNLayerPtr &layer, const InferenceEngine::Blob::Ptr &blob, const unsigned long height, const unsigned long width, const float threshold, std::vector<DetectionObject> &objects);
		void cameraCallback(const sensor_msgs::ImageConstPtr& colorImageMsg, const sensor_msgs::ImageConstPtr& depthImageMsg, const sensor_msgs::CameraInfo::ConstPtr& infoMsg);
		visualization_msgs::Marker createBoundingBox3dMarker(int id, geometry_msgs::Pose poseMin, geometry_msgs::Pose poseMax, float colorRGB[3], std::string targetFrame, ros::Time stamp);
		visualization_msgs::Marker createLabel3dMarker(int id, std::string label, geometry_msgs::Pose poseMin, geometry_msgs::Pose poseMax, float colorRGB[3], std::string targetFrame, ros::Time stamp);
		void publishImage(cv::Mat image);
		void publishBoundingImage(cv::Mat image);
};
#endif
