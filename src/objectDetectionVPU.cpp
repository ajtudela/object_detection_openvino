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

#include <limits>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/filters/crop_box.h>
#include <sensor_msgs/image_encodings.h>
#include <visualization_msgs/MarkerArray.h>
#include <vision_msgs/VisionInfo.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/Detection3DArray.h>
#include <vision_msgs/ObjectHypothesisWithPose.h>

#include "object_detection_openvino/objectDetectionVPU.h"

ObjectDetectionVPU::ObjectDetectionVPU(ros::NodeHandle& node, ros::NodeHandle& node_private): node_(node), nodePrivate_(node_private), 
											imageTransport_(nodePrivate_), syncImagePCL_(SyncPolicyImagePCL(5), colorSub_, pointsSub_){
	// Initialize ROS parameters
	getParams();

	// Initialize values for depth analysis
	useDepth_ = pointCloudTopic_.empty() ? false : true;

	// Initialize subscribers, create sync policy and synchronizer
	colorSub_.subscribe(imageTransport_, colorTopic_, 10);
	if (!useDepth_){
		colorSub_.registerCallback(boost::bind(&ObjectDetectionVPU::colorImageCallback, this,_1));
	}else{
		pointsSub_.subscribe(node_, pointCloudTopic_, 10);
		syncImagePCL_.connectInput(colorSub_, pointsSub_);
		syncImagePCL_.registerCallback(boost::bind(&ObjectDetectionVPU::colorPointCallback, this,_1,_2));
	}

	// Initialize publishers
	detectionInfoPub_ = nodePrivate_.advertise<vision_msgs::VisionInfo>(detectionInfoTopic_, 1, boost::bind(&ObjectDetectionVPU::connectInfoCallback, this, _1));
	detectionColorPub_ = imageTransport_.advertise(detectionImageTopic_, 1);
	detections2DPub_ = nodePrivate_.advertise<vision_msgs::Detection2DArray>(detections2DTopic_, 1);

	if (useDepth_){
		detections3DPub_ = nodePrivate_.advertise<vision_msgs::Detection3DArray>(detections3DTopic_, 1);
		markersPub_ = nodePrivate_.advertise<visualization_msgs::MarkerArray>("markers", 1);
	}

	// Set target device
	openvino_.setTargetDevice(deviceTarget_);

	// Set network model
	openvino_.setNetworkModel(modelFileName_, binFileName_, labelFileName_);

	// Get labels
	labels_ = openvino_.getLabels();

	// Configuring input and output 
	openvino_.configureNetwork(networkType_);

	// Load model to the device 
	openvino_.loadModelToDevice(deviceTarget_);

	// Create async inference request
	openvino_.createAsyncInferRequest();
}

ObjectDetectionVPU::~ObjectDetectionVPU(){
	nodePrivate_.deleteParam("model_thresh");
	nodePrivate_.deleteParam("model_iou_thresh");

	nodePrivate_.deleteParam("model_xml");
	nodePrivate_.deleteParam("model_bin");
	nodePrivate_.deleteParam("model_labels");
	nodePrivate_.deleteParam("model_type");
	nodePrivate_.deleteParam("device_target");

	nodePrivate_.deleteParam("camera_frame");
	nodePrivate_.deleteParam("color_topic");
	nodePrivate_.deleteParam("points_topic");
	nodePrivate_.deleteParam("detection_image_topic");
	nodePrivate_.deleteParam("detection_info_topic");
	nodePrivate_.deleteParam("detections2d_topic");
	nodePrivate_.deleteParam("detections3d_topic");

	nodePrivate_.deleteParam("show_fps");
}

void ObjectDetectionVPU::getParams(){
	ROS_INFO("[Object detection VPU]: Reading ROS parameters");

	nodePrivate_.param<float>("model_thresh", thresh_, 0.3);
	nodePrivate_.param<float>("model_iou_thresh", iouThresh_, 0.4);

	nodePrivate_.param<std::string>("model_xml", modelFileName_, "");
	nodePrivate_.param<std::string>("model_bin", binFileName_, "");
	nodePrivate_.param<std::string>("model_labels", labelFileName_, "");
	nodePrivate_.param<std::string>("model_type", networkType_, "");
	nodePrivate_.param<std::string>("device_target", deviceTarget_, "CPU");

	nodePrivate_.param<std::string>("camera_frame", cameraFrameId_, "camera_link");
	nodePrivate_.param<std::string>("color_topic", colorTopic_, "/camera/color/image_raw");
	nodePrivate_.param<std::string>("points_topic", pointCloudTopic_, "");
	nodePrivate_.param<std::string>("detection_image_topic", detectionImageTopic_, "image_raw");
	nodePrivate_.param<std::string>("detection_info_topic", detectionInfoTopic_, "detection_info");
	nodePrivate_.param<std::string>("detections2d_topic", detections2DTopic_, "detections2d");
	nodePrivate_.param<std::string>("detections3d_topic", detections3DTopic_, "detections3d");

	nodePrivate_.param<bool>("show_fps", showFPS_, false);
}

void ObjectDetectionVPU::connectInfoCallback(const ros::SingleSubscriberPublisher& pub){
	ROS_INFO("[Object detection VPU]: Subscribed to vision info topic");

	// Create the key on the param server
	std::string classKey = std::string("class_labels");
	if (!nodePrivate_.hasParam(classKey)){
		nodePrivate_.setParam(classKey, labels_);
	}

	// Create and publish info
	vision_msgs::VisionInfo detectionInfo;
	detectionInfo.header.frame_id = cameraFrameId_;
	detectionInfo.header.stamp = ros::Time::now();
	detectionInfo.method = networkType_ + " detection";
	detectionInfo.database_version = 0;
	detectionInfo.database_location = nodePrivate_.getNamespace() + std::string("/") + classKey;

	detectionInfoPub_.publish(detectionInfo);
}

void ObjectDetectionVPU::colorImageCallback(const sensor_msgs::Image::ConstPtr& colorImageMsg){
	colorPointCallback(colorImageMsg, nullptr);
}

void ObjectDetectionVPU::colorPointCallback(const sensor_msgs::Image::ConstPtr& colorImageMsg, const sensor_msgs::PointCloud2::ConstPtr& pointsMsg){
	// Note: Only infer object if there's any subscriber
	if (detectionColorPub_.getNumSubscribers() == 0 && detections2DPub_.getNumSubscribers() == 0
		&& detections3DPub_.getNumSubscribers() == 0 && markersPub_.getNumSubscribers() == 0) return;
	ROS_INFO_ONCE("[Object detection VPU]: Subscribed to color image topic: %s", colorTopic_.c_str());

	// Read header
	colorFrameId_ = colorImageMsg->header.frame_id;

	// Create arrays to publish and format headers
	visualization_msgs::MarkerArray markerArray;

	vision_msgs::Detection2DArray detections2D;
	detections2D.header.frame_id = colorFrameId_;
	detections2D.header.stamp = colorImageMsg->header.stamp;

	vision_msgs::Detection3DArray detections3D;
	detections3D.header.frame_id = cameraFrameId_;
	detections3D.header.stamp = colorImageMsg->header.stamp;

	auto wallclock = std::chrono::high_resolution_clock::now();

	// Convert from ROS to CV image
	cv_bridge::CvImagePtr colorImageCv;
	try{
		colorImageCv = cv_bridge::toCvCopy(colorImageMsg, sensor_msgs::image_encodings::BGR8);
	}catch (cv_bridge::Exception& e){
		ROS_ERROR("[Object detection VPU]: cv_bridge exception: %s", e.what());
		return;
	}

	const size_t colorHeight = (size_t) colorImageCv->image.size().height;
	const size_t colorWidth  = (size_t) colorImageCv->image.size().width;

	// Copy data from image to input blob
	nextFrame_ = colorImageCv->image.clone();
	openvino_.frameToNextInfer(nextFrame_, false);

	// Tranform the pointcloud
	sensor_msgs::PointCloud2 localCloudPC2;
	pcloud::Ptr localCloudPCLPtr(new pcl::PointCloud<pcl::PointXYZRGB>);
	if (useDepth_){
		ROS_INFO_ONCE("[Object detection VPU]: Subscribed to pointcloud topic: %s", pointCloudTopic_.c_str());
		// Transform to camera frame
		try{
			pcl_ros::transformPointCloud(cameraFrameId_, *pointsMsg, localCloudPC2, tfListener_);
		}catch (tf::TransformException& ex){
			ROS_ERROR_STREAM("[Object detection VPU]: Transform error of sensor data: " << ex.what() << ", quitting callback");
			return;
		}
		// Convert to PCL
		pcl::fromROSMsg(localCloudPC2, *localCloudPCLPtr);
	}

	// Load network
	// In the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
	auto t0 = std::chrono::high_resolution_clock::now();
	openvino_.startNextAsyncInferRequest();
	auto t1 = std::chrono::high_resolution_clock::now();

	if (openvino_.isDeviceReady()){
		// Show FPS
		if (showFPS_){
			t1 = std::chrono::high_resolution_clock::now();
			ms detection = std::chrono::duration_cast<ms>(t1 - t0);

			t0 = std::chrono::high_resolution_clock::now();
			ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
			wallclock = t0;

			t0 = std::chrono::high_resolution_clock::now();

			std::ostringstream out;
			cv::putText(currFrame_, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
			out.str("");
			out << "Wallclock time ";
			out << std::fixed << std::setprecision(2) << wall.count() << " ms (" << 1000.f / wall.count() << " fps)";
			cv::putText(currFrame_, out.str(), cv::Point2f(0, 50), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

			out.str("");
			out << "Detection time  : " << std::fixed << std::setprecision(2) << detection.count()
				<< " ms ("
				<< 1000.f / detection.count() << " fps)";
			cv::putText(currFrame_, out.str(), cv::Point2f(0, 75), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
		}

		// Get detection objects
		std::vector<DetectionObject> objects = openvino_.getDetectionObjects(colorHeight, colorWidth, thresh_, iouThresh_);

		int detectionId = 0;

		/* Process objects */
		for (auto &object: objects){
			// Skip if confidence is less than the threshold
			if (object.confidence < thresh_) continue;

			auto label = object.classId;
			float confidence = object.confidence;

			ROS_DEBUG("[Object detection VPU]: %s tag (%.2f%%)", labels_[label].c_str(), confidence*100);

			// Improve bounding box
			object.xmin = object.xmin < 0 ? 0 : object.xmin;
			object.ymin = object.ymin < 0 ? 0 : object.ymin;
			object.xmax = object.xmax > colorWidth ? colorWidth : object.xmax;
			object.ymax = object.ymax > colorHeight ? colorHeight : object.ymax;

			// Color of the class
			int offset = object.classId * 123457 % COCO_CLASSES;
			float colorRGB[3];
			colorRGB[0] = getColor(2, offset, COCO_CLASSES);
			colorRGB[1] = getColor(1, offset, COCO_CLASSES);
			colorRGB[2] = getColor(0, offset, COCO_CLASSES);

			// Create detection2D and push to array
			vision_msgs::Detection2D detection2D;
			if (createDetection2DMsg(object, detections2D.header, detection2D)){
				detections2D.detections.push_back(detection2D);
			}

			if (useDepth_){
				// Create detection3D and push to array
				vision_msgs::Detection3D detection3D;

				if (createDetection3DMsg(object, detections3D.header, localCloudPC2, localCloudPCLPtr, detection3D)){
					// Create markers
					visualization_msgs::Marker vizMarker, labelMarker;

					createBBox3DMarker(detectionId, detections3D.header, colorRGB, detection3D.bbox, vizMarker);
					createLabel3DMarker(detectionId*10, detections3D.header, colorRGB, detection3D.bbox, labels_[label], labelMarker);

					detections3D.detections.push_back(detection3D);
					markerArray.markers.push_back(vizMarker);
					markerArray.markers.push_back(labelMarker);
				}
			}

			/* Image */
			// Text label
			std::ostringstream conf;
			conf << ":" << std::fixed << std::setprecision(3) << confidence;
			std::string labelText = (label < this->labels_.size() ? this->labels_[label] : std::string("label #") + std::to_string(label)) + conf.str();
			// Rectangles for class
			cv::rectangle(currFrame_, cv::Point2f(object.xmin-1, object.ymin), cv::Point2f(object.xmin + 180, object.ymin - 22), cv::Scalar(colorRGB[2], colorRGB[1], colorRGB[0]), cv::FILLED, cv::LINE_AA);
			cv::putText(currFrame_, labelText, cv::Point2f(object.xmin, object.ymin - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0), 1.5, cv::LINE_AA);
			cv::rectangle(currFrame_, cv::Point2f(object.xmin, object.ymin), cv::Point2f(object.xmax, object.ymax), cv::Scalar(colorRGB[2], colorRGB[1], colorRGB[0]), 4, cv::LINE_AA);

			detectionId++;
		}

		// Publish detections and markers
		publishImage(currFrame_);
		if (!objects.empty()){
			detections2DPub_.publish(detections2D);
			if (useDepth_){
				detections3DPub_.publish(detections3D);
				markersPub_.publish(markerArray);
			}
		}
	}

	// In the truly Async mode we swap the NEXT and CURRENT requests for the next iteration
	currFrame_ = nextFrame_;
	nextFrame_ = cv::Mat();
	openvino_.swapAsyncInferRequest();
}

void ObjectDetectionVPU::showHistogram(cv::Mat image, cv::Scalar mean){
	int histSize = 256;
	float range[] = { 0, histSize }; //the upper boundary is exclusive
	const float* histRange = { range };
	bool uniform = true, accumulate = false;
	cv::Mat depthHist;
	calcHist( &image, 1, 0, cv::Mat(), depthHist, 1, &histSize, &histRange, uniform, accumulate );
	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histSize );
	cv::Mat histImage(hist_h, hist_w,  CV_8UC3, cv::Scalar( 0,0,0) );
	normalize(depthHist, depthHist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
	for(int i = 1; i < histSize; i++){
		cv::line(histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(depthHist.at<float>(i-1)) ),
			  cv::Point( bin_w*(i), hist_h - cvRound(depthHist.at<float>(i)) ),
			  cv::Scalar( 0, 0, 255), 2, 8, 0  );
	}
	cv::line(histImage, cv::Point(mean[0], 0), cv::Point(mean[0], histImage.rows), cv::Scalar(0,255,0));
	cv::imshow("Histogram", histImage );
	cv::waitKey(10);
}

bool ObjectDetectionVPU::createDetection2DMsg(DetectionObject object, std_msgs::Header header, vision_msgs::Detection2D& detection2D){
	detection2D.header = header;

	// Class probabilities
	vision_msgs::ObjectHypothesisWithPose hypo;
	hypo.id = object.classId;
	hypo.score = object.confidence;
	detection2D.results.push_back(hypo);

	// 2D bounding box surrounding the object
	detection2D.bbox.center.x = (object.xmax + object.xmin) / 2;
	detection2D.bbox.center.y = (object.ymax + object.ymin) / 2;
	detection2D.bbox.size_x = object.xmax - object.xmin;
	detection2D.bbox.size_y = object.ymax - object.ymin;

	// The 2D data that generated these results
	cv::Mat croppedImage = currFrame_(cv::Rect(object.xmin, object.ymin, object.xmax - object.xmin, object.ymax - object.ymin));

	detection2D.source_img.header = header;
	detection2D.source_img.height = croppedImage.rows;
	detection2D.source_img.width = croppedImage.cols;
	detection2D.source_img.encoding = "bgr8";
	detection2D.source_img.is_bigendian = false;
	detection2D.source_img.step = croppedImage.cols * 3;
	size_t size = detection2D.source_img.step * croppedImage.rows;
	detection2D.source_img.data.resize(size);
	memcpy((char*)(&detection2D.source_img.data[0]), croppedImage.data, size);

	return true;
}

bool ObjectDetectionVPU::createDetection3DMsg(DetectionObject object, std_msgs::Header header, const sensor_msgs::PointCloud2& cloudPC2, pcloud::ConstPtr cloudPCL, vision_msgs::Detection3D& detection3D){
	// Calculate the center in 3D coordinates
	int centerX, centerY;
	centerX = (object.xmax + object.xmin) / 2;
	centerY = (object.ymax + object.ymin) / 2;

	int pclIndex = centerX + (centerY * cloudPC2.width);
	pcl::PointXYZRGB centerPoint = cloudPCL->at(pclIndex);

	if (std::isnan(centerPoint.x)) return false;

	// Calculate the bounding box
	float maxX, minX, maxY, minY, maxZ, minZ;
	maxX = maxY = maxZ = -std::numeric_limits<float>::max();
	minX = minY = minZ = std::numeric_limits<float>::max();

	for (int i = object.xmin; i < object.xmax; i++){
		for (int j = object.ymin; j < object.ymax; j++){
			pclIndex = i + (j * cloudPC2.width);
			pcl::PointXYZRGB point =  cloudPCL->at(pclIndex);

			if (std::isnan(point.x)) continue;
			if (fabs(point.x - centerPoint.x) > thresh_) continue;

			maxX = std::max(point.x, maxX);
			maxY = std::max(point.y, maxY);
			maxZ = std::max(point.z, maxZ);
			minX = std::min(point.x, minX);
			minY = std::min(point.y, minY);
			minZ = std::min(point.z, minZ);
		}
	}

	// Header
	detection3D.header = header;

	// 3D bounding box surrounding the object
	detection3D.bbox.center.position.x = centerPoint.x;
	detection3D.bbox.center.position.y = centerPoint.y;
	detection3D.bbox.center.position.z = centerPoint.z;
	detection3D.bbox.center.orientation.x = 0.0;
	detection3D.bbox.center.orientation.y = 0.0;
	detection3D.bbox.center.orientation.z = 0.0;
	detection3D.bbox.center.orientation.w = 1.0;
	detection3D.bbox.size.x = maxX - minX;
	detection3D.bbox.size.y = maxY - minY;
	detection3D.bbox.size.z = maxZ - minZ;

	// Class probabilities
	// We use the pose as the min Z of the bounding box
	// Because this is not a "real" detection 3D
	vision_msgs::ObjectHypothesisWithPose hypo;
	hypo.id = object.classId;
	hypo.score = object.confidence;
	hypo.pose.pose = detection3D.bbox.center;
	hypo.pose.pose.position.z -= detection3D.bbox.size.z / 2.0;
	detection3D.results.push_back(hypo);

	// The 3D data that generated these results:
	// Cropping the cloud
	pcl::CropBox<pcl::PointXYZRGB> boxFilter;
	pcloud::Ptr croppedCloudPCLPtr(new pcl::PointCloud<pcl::PointXYZRGB>);
	boxFilter.setMin(Eigen::Vector4f(minX, minY, minZ, 1.0));
	boxFilter.setMax(Eigen::Vector4f(maxX, maxY, maxZ, 1.0));
	boxFilter.setInputCloud(cloudPCL);
	boxFilter.filter(*croppedCloudPCLPtr);
	// Convert to sensor_msgs
	sensor_msgs::PointCloud2 croppedCloudPC2;
	pcl::toROSMsg(*croppedCloudPCLPtr, croppedCloudPC2);
	detection3D.source_cloud = croppedCloudPC2;

	return true;
}

bool ObjectDetectionVPU::createBBox3DMarker(int id, std_msgs::Header header, float colorRGB[3], vision_msgs::BoundingBox3D bbox, visualization_msgs::Marker& marker){
	marker.header = header;
	marker.ns = "boundingBox3d";
	marker.id = id;
	marker.type = visualization_msgs::Marker::CUBE;
	marker.action = visualization_msgs::Marker::ADD;
	marker.lifetime = ros::Duration(0.15);
	marker.pose = bbox.center;
	marker.scale = bbox.size;
	marker.color.r = colorRGB[0] / 255.0;
	marker.color.g = colorRGB[1] / 255.0;
	marker.color.b = colorRGB[2] / 255.0;
	marker.color.a = 0.2f;

	return true;
}

bool ObjectDetectionVPU::createLabel3DMarker(int id, std_msgs::Header header, float colorRGB[3], vision_msgs::BoundingBox3D bbox, std::string label, visualization_msgs::Marker& marker){
	marker.header = header;
	marker.ns = "label3d";
	marker.id = id;
	marker.text = label;
	marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
	marker.action = visualization_msgs::Marker::ADD;
	marker.lifetime = ros::Duration(0.15);
	marker.pose = bbox.center;
	marker.pose.position.z += bbox.size.z / 2.0 + 0.05;
	marker.scale.z = 0.3;
	marker.color.r = colorRGB[0] / 255.0;
	marker.color.g = colorRGB[1] / 255.0;
	marker.color.b = colorRGB[2] / 255.0;
	marker.color.a = 0.8f;

	return true;
}

void ObjectDetectionVPU::publishImage(cv::Mat image){
	cv_bridge::CvImage outputImageMsg;
	outputImageMsg.header.stamp = ros::Time::now();
	outputImageMsg.header.frame_id = colorFrameId_;
	outputImageMsg.encoding = sensor_msgs::image_encodings::BGR8;
	outputImageMsg.image = image;

	detectionColorPub_.publish(outputImageMsg.toImageMsg());
}

int ObjectDetectionVPU::getColor(int c, int x, int max){
	float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

	float ratio = ((float)x/max)*5;
	int i = floor(ratio);
	int j = ceil(ratio);
	ratio -= i;
	float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];

	return floor(r*255);
}
