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

// C++
#include <limits>

// PCL
#include "pcl/filters/crop_box.h"

// ROS
#include "pcl_conversions/pcl_conversions.h"
#include "pcl_ros/transforms.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "tf2_sensor_msgs/tf2_sensor_msgs.h"

#include "object_detection_openvino/objectDetectionVPU.hpp"

ObjectDetectionVPU::ObjectDetectionVPU(std::string node_name) : 
										Node(node_name) , 
										openvino_(node_name){
	// Initialize ROS parameters
	get_params();

	// Initialize values for depth analysis
	use_depth_ = pointcloud_topic_.empty() ? false : true;

	// Initialize publishers
	detection_info_pub_ = this->create_publisher<vision_msgs::msg::VisionInfo>(
				detection_info_topic_, 
				rclcpp::QoS(1).transient_local());
	connect_info_callback();
	detection_color_pub_ = image_transport::create_publisher(this, detection_image_topic_);
	detections_2d_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(detections_2d_topic_, 1);

	if (use_depth_){
		detections_3d_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>(detections_3d_topic_, 1);
		markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("markers", 1);
	}

	// Initialize subscribers, create sync policy and synchronizer
	image_transport::TransportHints hints(this, "raw");
	color_sub_.subscribe(this, color_topic_, hints.getTransport());

	if (!use_depth_){
		color_sub_.registerCallback(std::bind(&ObjectDetectionVPU::color_image_callback, 
											this, 
											std::placeholders::_1));
	}else{
		points_sub_.subscribe(this, pointcloud_topic_);
		approximate_sync_ = std::make_shared<ApproximateSync>(
			ApproximatePolicy(5),
			color_sub_, 
			points_sub_);
		approximate_sync_->registerCallback(std::bind(&ObjectDetectionVPU::color_point_callback,
											this, 
											std::placeholders::_1, 
											std::placeholders::_2));
	}

	// Initialize transform buffer
	tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());

	// Set target device
	openvino_.set_target_device(device_target_);

	// Set network model
	openvino_.set_network_model(model_filename_, bin_filename_, label_filename_);

	// Get labels
	labels_ = openvino_.get_labels();

	// Configuring input and output
	openvino_.configure_network(network_type_);

	// Load model to the device 
	openvino_.load_model_to_device(device_target_);

	// Create async inference request
	openvino_.create_async_infer_request();
}

ObjectDetectionVPU::~ObjectDetectionVPU(){
}

void ObjectDetectionVPU::get_params(){
	// Model parameters
	this->declare_parameter("model_thresh", 0.3);
	this->get_parameter("model_thresh", thresh_);
	RCLCPP_INFO(this->get_logger(), "The parameter model_thresh is set to: [%f]", thresh_);
	this->declare_parameter("model_iou_thresh", 0.4);
	this->get_parameter("model_iou_thresh", iou_thresh_);
	RCLCPP_INFO(this->get_logger(), "The parameter model_iou_thresh is set to: [%f]", iou_thresh_);

	// Network parameters
	this->declare_parameter("model_xml", "");
	this->get_parameter("model_xml", model_filename_);
	RCLCPP_INFO(this->get_logger(), "The parameter model_xml is set to: [%s]", model_filename_.c_str());
	this->declare_parameter("model_bin", "");
	this->get_parameter("model_bin", bin_filename_);
	RCLCPP_INFO(this->get_logger(), "The parameter model_bin is set to: [%s]", bin_filename_.c_str());
	this->declare_parameter("model_labels", "");
	this->get_parameter("model_labels", label_filename_);
	RCLCPP_INFO(this->get_logger(), "The parameter model_labels is set to: [%s]", label_filename_.c_str());
	this->declare_parameter("model_type", "");
	this->get_parameter("model_type", network_type_);
	RCLCPP_INFO(this->get_logger(), "The parameter model_type is set to: [%s]", network_type_.c_str());
	this->declare_parameter("device_target", "CPU");
	this->get_parameter("device_target", device_target_);
	RCLCPP_INFO(this->get_logger(), "The parameter device_target is set to: [%s]", device_target_.c_str());

	this->declare_parameter("camera_frame", "camera_link");
	this->get_parameter("camera_frame", camera_frame_);
	RCLCPP_INFO(this->get_logger(), "The parameter camera_frame is set to: [%s]", camera_frame_.c_str());

	// Topics
	this->declare_parameter("color_topic", "/camera/color/image_raw");
	this->get_parameter("color_topic", color_topic_);
	RCLCPP_INFO(this->get_logger(), "The parameter color_topic is set to: [%s]", color_topic_.c_str());
	this->declare_parameter("points_topic", "");
	this->get_parameter("points_topic", pointcloud_topic_);
	RCLCPP_INFO(this->get_logger(), "The parameter points_topic is set to: [%s]", pointcloud_topic_.c_str());
	this->declare_parameter("detection_info_topic", "/detection_info");
	this->get_parameter("detection_info_topic", detection_info_topic_);
	RCLCPP_INFO(this->get_logger(), "The parameter detection_info_topic is set to: [%s]", detection_info_topic_.c_str());
	this->declare_parameter("detection_image_topic", "/detection_image");
	this->get_parameter("detection_image_topic", detection_image_topic_);
	RCLCPP_INFO(this->get_logger(), "The parameter detection_image_topic is set to: [%s]", detection_image_topic_.c_str());
	this->declare_parameter("detections_2d_topic", "/detections_2d");
	this->get_parameter("detections_2d_topic", detections_2d_topic_);
	RCLCPP_INFO(this->get_logger(), "The parameter detections_2d_topic is set to: [%s]", detections_2d_topic_.c_str());
	this->declare_parameter("detections_3d_topic", "/detections_3d");
	this->get_parameter("detections_3d_topic", detections_3d_topic_);
	RCLCPP_INFO(this->get_logger(), "The parameter detections_3d_topic is set to: [%s]", detections_3d_topic_.c_str());

	this->declare_parameter("show_fps", false);
	this->get_parameter("show_fps", show_fps_);
	RCLCPP_INFO(this->get_logger(), "The parameter show_fps is set to: [%s]", show_fps_ ? "true" : "false");
}

// TODO(ros2) Implement when SubscriberStatusCallback is available
void ObjectDetectionVPU::connect_info_callback(){
	RCLCPP_INFO(this->get_logger(), "Subscribed to vision info topic");

	// Create the key on the param server
	std::string class_key = std::string("class_labels");
	if (!this->has_parameter(class_key)){
		this->declare_parameter(class_key, labels_);
	}

	// Create and publish info
	vision_msgs::msg::VisionInfo detection_info;
	detection_info.header.frame_id = camera_frame_;
	detection_info.header.stamp = this->now();
	detection_info.method = network_type_ + " detection";
	detection_info.database_version = 0;
	detection_info.database_location = this->get_namespace() + std::string("/") + class_key;

	detection_info_pub_->publish(detection_info);
}

void ObjectDetectionVPU::color_image_callback(
		const sensor_msgs::msg::Image::ConstSharedPtr & color_image_msg){
	color_point_callback(color_image_msg, nullptr);
}

void ObjectDetectionVPU::color_point_callback(
		const sensor_msgs::msg::Image::ConstSharedPtr & color_image_msg, 
		const sensor_msgs::msg::PointCloud2::ConstSharedPtr & points_msg){
	// Note: Only infer object if there's any subscriber
	if (detection_color_pub_.getNumSubscribers() == 0 && 
		detections_2d_pub_->get_subscription_count() == 0 && 
		detections_3d_pub_->get_subscription_count() == 0 && 
		markers_pub_->get_subscription_count() == 0){
		return;
	}
	RCLCPP_INFO_ONCE(this->get_logger(), "Subscribed to color image topic: %s", color_topic_.c_str());

	// Read header
	color_frame_ = color_image_msg->header.frame_id;

	// Create arrays to publish and format headers
	visualization_msgs::msg::MarkerArray marker_array;

	vision_msgs::msg::Detection2DArray detections_2d;
	detections_2d.header.frame_id = color_frame_;
	detections_2d.header.stamp = color_image_msg->header.stamp;

	vision_msgs::msg::Detection3DArray detections_3d;
	detections_3d.header.frame_id = camera_frame_;
	detections_3d.header.stamp = color_image_msg->header.stamp;

	auto wallclock = std::chrono::high_resolution_clock::now();

	// Convert from ROS to CV image
	cv_bridge::CvImagePtr color_image_cv;
	try{
		color_image_cv = cv_bridge::toCvCopy(color_image_msg, sensor_msgs::image_encodings::BGR8);
	}catch (cv_bridge::Exception& e){
		RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
		return;
	}

	const size_t color_height = (size_t) color_image_cv->image.size().height;
	const size_t color_width  = (size_t) color_image_cv->image.size().width;

	// Copy data from image to input blob
	next_frame_ = color_image_cv->image.clone();
	openvino_.frame_to_next_infer(next_frame_, false);

	// Transform the pointcloud
	sensor_msgs::msg::PointCloud2 local_cloud_pc2;
	pcloud::Ptr local_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	if (use_depth_){
		RCLCPP_INFO_ONCE(this->get_logger(), "Subscribed to pointcloud topic: %s", pointcloud_topic_.c_str());
		// Transform to camera frame
		try{
			geometry_msgs::msg::TransformStamped tf_stamped =
				tf_buffer_->lookupTransform(
				camera_frame_, points_msg->header.frame_id,
				tf2_ros::fromMsg(points_msg->header.stamp));
			tf2::doTransform(*points_msg, local_cloud_pc2, tf_stamped);
		}catch (const tf2::TransformException & ex){
			RCLCPP_WARN(this->get_logger(), ex.what());
			return;
		}
		// Convert the cloud to PCL format
		pcl::fromROSMsg(local_cloud_pc2, *local_cloud);
	}

	// Load network
	// In the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
	auto t0 = std::chrono::high_resolution_clock::now();
	openvino_.start_next_async_infer_request();
	auto t1 = std::chrono::high_resolution_clock::now();

	if (openvino_.is_device_ready()){
		// Show FPS
		if (show_fps_){
			t1 = std::chrono::high_resolution_clock::now();
			ms detection = std::chrono::duration_cast<ms>(t1 - t0);

			t0 = std::chrono::high_resolution_clock::now();
			ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
			wallclock = t0;

			t0 = std::chrono::high_resolution_clock::now();

			std::ostringstream out;
			cv::putText(current_frame_, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
			out.str("");
			out << "Wallclock time ";
			out << std::fixed << std::setprecision(2) << wall.count() << " ms (" << 1000.f / wall.count() << " fps)";
			cv::putText(current_frame_, out.str(), cv::Point2f(0, 50), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

			out.str("");
			out << "Detection time  : " << std::fixed << std::setprecision(2) << detection.count()
				<< " ms ("
				<< 1000.f / detection.count() << " fps)";
			cv::putText(current_frame_, out.str(), cv::Point2f(0, 75), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
		}

		// Get detection objects
		std::vector<DetectionObject> objects = openvino_.get_detection_objects(color_height, color_width, thresh_, iou_thresh_);

		int detection_id = 0;

		/* Process objects */
		for (auto &object: objects){
			// Skip if confidence is less than the threshold
			if (object.confidence < thresh_) continue;

			auto label = object.classId;
			float confidence = object.confidence;

			RCLCPP_DEBUG(this->get_logger(), "%s tag (%.2f%%)", labels_[label].c_str(), confidence*100);

			// Improve bounding box
			object.xmin = object.xmin < 0 ? 0 : object.xmin;
			object.ymin = object.ymin < 0 ? 0 : object.ymin;
			object.xmax = object.xmax > color_width ? color_width : object.xmax;
			object.ymax = object.ymax > color_height ? color_height : object.ymax;

			// Color of the class
			int offset = object.classId * 123457 % COCO_CLASSES;
			float color_rgb[3];
			color_rgb[0] = get_color(2, offset, COCO_CLASSES);
			color_rgb[1] = get_color(1, offset, COCO_CLASSES);
			color_rgb[2] = get_color(0, offset, COCO_CLASSES);

			// Create detection2D and push to array
			vision_msgs::msg::Detection2D detection_2d;
			if (create_detection_2d_msg(object, detections_2d.header, detection_2d)){
				detections_2d.detections.push_back(detection_2d);
			}

			if (use_depth_){
				// Create detection3D and push to array
				vision_msgs::msg::Detection3D detection_3d;

				if (create_detection_3d_msg(object, detections_3d.header, local_cloud_pc2, local_cloud, detection_3d)){
					// Create markers
					visualization_msgs::msg::Marker marker_viz, marker_label;

					create_bbox_3d_marker(detection_id, detections_3d.header, color_rgb, detection_3d.bbox, marker_viz);
					create_label_3d_marker(detection_id*10, detections_3d.header, color_rgb, detection_3d.bbox, labels_[label], marker_label);

					detections_3d.detections.push_back(detection_3d);
					marker_array.markers.push_back(marker_viz);
					marker_array.markers.push_back(marker_label);
				}
			}

			/* Image */
			// Text label
			std::ostringstream conf;
			conf << ":" << std::fixed << std::setprecision(3) << confidence;
			std::string label_text = (label < this->labels_.size() ? this->labels_[label] : std::string("label #") + std::to_string(label)) + conf.str();
			// Rectangles for class
			cv::rectangle(current_frame_, cv::Point2f(object.xmin-1, object.ymin), cv::Point2f(object.xmin + 180, object.ymin - 22), cv::Scalar(color_rgb[2], color_rgb[1], color_rgb[0]), cv::FILLED, cv::LINE_AA);
			cv::putText(current_frame_, label_text, cv::Point2f(object.xmin, object.ymin - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0), 1.5, cv::LINE_AA);
			cv::rectangle(current_frame_, cv::Point2f(object.xmin, object.ymin), cv::Point2f(object.xmax, object.ymax), cv::Scalar(color_rgb[2], color_rgb[1], color_rgb[0]), 4, cv::LINE_AA);

			detection_id++;
		}

		// Publish detections and markers
		publish_image(current_frame_);
		if (!objects.empty()){
			detections_2d_pub_->publish(detections_2d);
			if (use_depth_){
				detections_3d_pub_->publish(detections_3d);
				markers_pub_->publish(marker_array);
			}
		}
	}

	// In the truly Async mode we swap the NEXT and CURRENT requests for the next iteration
	current_frame_ = next_frame_;
	next_frame_ = cv::Mat();
	openvino_.swap_async_infer_request();
}

void ObjectDetectionVPU::show_histogram(cv::Mat image, cv::Scalar mean){
	int hist_size = 256;
	float range[] = { 0, hist_size }; //the upper boundary is exclusive
	const float* hist_range = { range };
	bool uniform = true, accumulate = false;
	cv::Mat depth_hist;
	calcHist( &image, 1, 0, cv::Mat(), depth_hist, 1, &hist_size, &hist_range, uniform, accumulate );
	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound( (double) hist_w/hist_size );
	cv::Mat hist_image(hist_h, hist_w,  CV_8UC3, cv::Scalar( 0,0,0) );
	normalize(depth_hist, depth_hist, 0, hist_image.rows, cv::NORM_MINMAX, -1, cv::Mat() );
	for (int i = 1; i < hist_size; i++){
		cv::line(hist_image, cv::Point( bin_w*(i-1), hist_h - cvRound(depth_hist.at<float>(i-1)) ),
			cv::Point( bin_w*(i), hist_h - cvRound(depth_hist.at<float>(i)) ),
			cv::Scalar( 0, 0, 255), 2, 8, 0  );
	}
	cv::line(hist_image, cv::Point(mean[0], 0), cv::Point(mean[0], hist_image.rows), cv::Scalar(0,255,0));
	cv::imshow("Histogram", hist_image );
	cv::waitKey(10);
}

bool ObjectDetectionVPU::create_detection_2d_msg(DetectionObject object, 
									std_msgs::msg::Header header, 
									vision_msgs::msg::Detection2D& detection_2d){
	detection_2d.header = header;

	// Class probabilities
	vision_msgs::msg::ObjectHypothesisWithPose hypo_with_pose;
	hypo_with_pose.hypothesis.class_id = std::to_string(object.classId);
	hypo_with_pose.hypothesis.score = object.confidence;
	detection_2d.results.push_back(hypo_with_pose);

	// 2D bounding box surrounding the object
	detection_2d.bbox.center.x = (object.xmax + object.xmin) / 2;
	detection_2d.bbox.center.y = (object.ymax + object.ymin) / 2;
	detection_2d.bbox.size_x = object.xmax - object.xmin;
	detection_2d.bbox.size_y = object.ymax - object.ymin;

	return true;
}

bool ObjectDetectionVPU::create_detection_3d_msg(DetectionObject object, 
									std_msgs::msg::Header header, 
									const sensor_msgs::msg::PointCloud2& cloud_pc2, 
									pcloud::ConstPtr cloud_pcl, 
									vision_msgs::msg::Detection3D& detection_3d){
	// Calculate the center in 3D coordinates
	int center_x = (object.xmax + object.xmin) / 2;
	int center_y = (object.ymax + object.ymin) / 2;

	int index_pcl = center_x + (center_y * cloud_pc2.width);
	pcl::PointXYZRGB center_point = cloud_pcl->at(index_pcl);

	if (std::isnan(center_point.x)) return false;

	// Calculate the bounding box
	float maxX, minX, maxY, minY, maxZ, minZ;
	maxX = maxY = maxZ = -std::numeric_limits<float>::max();
	minX = minY = minZ = std::numeric_limits<float>::max();

	for (int i = object.xmin; i < object.xmax; i++){
		for (int j = object.ymin; j < object.ymax; j++){
			index_pcl = i + (j * cloud_pc2.width);
			pcl::PointXYZRGB point =  cloud_pcl->at(index_pcl);

			if (std::isnan(point.x)) continue;
			if (fabs(point.x - center_point.x) > thresh_) continue;

			maxX = std::max(point.x, maxX);
			maxY = std::max(point.y, maxY);
			maxZ = std::max(point.z, maxZ);
			minX = std::min(point.x, minX);
			minY = std::min(point.y, minY);
			minZ = std::min(point.z, minZ);
		}
	}

	// Header
	detection_3d.header = header;

	// 3D bounding box surrounding the object
	detection_3d.bbox.center.position.x = center_point.x;
	detection_3d.bbox.center.position.y = center_point.y;
	detection_3d.bbox.center.position.z = center_point.z;
	detection_3d.bbox.center.orientation.x = 0.0;
	detection_3d.bbox.center.orientation.y = 0.0;
	detection_3d.bbox.center.orientation.z = 0.0;
	detection_3d.bbox.center.orientation.w = 1.0;
	detection_3d.bbox.size.x = maxX - minX;
	detection_3d.bbox.size.y = maxY - minY;
	detection_3d.bbox.size.z = maxZ - minZ;

	// Class probabilities
	// We use the pose as the min Z of the bounding box
	// Because this is not a "real" detection 3D
	vision_msgs::msg::ObjectHypothesisWithPose hypo_with_pose;
	hypo_with_pose.hypothesis.class_id = std::to_string(object.classId);
	hypo_with_pose.hypothesis.score = object.confidence;
	hypo_with_pose.pose.pose = detection_3d.bbox.center;
	hypo_with_pose.pose.pose.position.z -= detection_3d.bbox.size.z / 2.0;
	detection_3d.results.push_back(hypo_with_pose);

	return true;
}

bool ObjectDetectionVPU::create_bbox_3d_marker(int id, 
								std_msgs::msg::Header header, 
								float color_rgb[3], 
								vision_msgs::msg::BoundingBox3D bbox, 
								visualization_msgs::msg::Marker& marker){
	marker.header = header;
	marker.ns = "boundingBox3d";
	marker.id = id;
	marker.type = visualization_msgs::msg::Marker::CUBE;
	marker.action = visualization_msgs::msg::Marker::ADD;
	marker.lifetime = rclcpp::Duration::from_seconds(0.03);
	marker.pose = bbox.center;
	marker.scale = bbox.size;
	marker.color.r = color_rgb[0] / 255.0;
	marker.color.g = color_rgb[1] / 255.0;
	marker.color.b = color_rgb[2] / 255.0;
	marker.color.a = 0.2f;

	return true;
}

bool ObjectDetectionVPU::create_label_3d_marker(int id, 
								std_msgs::msg::Header header, 
								float color_rgb[3], 
								vision_msgs::msg::BoundingBox3D bbox, 
								std::string label, 
								visualization_msgs::msg::Marker& marker){
	marker.header = header;
	marker.ns = "label3d";
	marker.id = id;
	marker.text = label;
	marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
	marker.action = visualization_msgs::msg::Marker::ADD;
	marker.lifetime = rclcpp::Duration::from_seconds(0.03);
	marker.pose = bbox.center;
	marker.pose.position.z += bbox.size.z / 2.0 + 0.05;
	marker.scale.z = 0.3;
	marker.color.r = color_rgb[0] / 255.0;
	marker.color.g = color_rgb[1] / 255.0;
	marker.color.b = color_rgb[2] / 255.0;
	marker.color.a = 0.8f;

	return true;
}

void ObjectDetectionVPU::publish_image(cv::Mat image){
	cv_bridge::CvImage output_msg;
	output_msg.header.stamp = this->now();
	output_msg.header.frame_id = color_frame_;
	output_msg.encoding = sensor_msgs::image_encodings::BGR8;
	output_msg.image = image;

	detection_color_pub_.publish(output_msg.toImageMsg());
}

int ObjectDetectionVPU::get_color(int c, int x, int max){
	float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

	float ratio = ((float)x/max)*5;
	int i = floor(ratio);
	int j = ceil(ratio);
	ratio -= i;
	float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];

	return floor(r*255);
}
