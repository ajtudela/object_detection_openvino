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

#include <iostream>
#include <iterator>
#include <boost/filesystem.hpp>

#include <ngraph/ngraph.hpp>

#include <sensor_msgs/image_encodings.h>
#include <visualization_msgs/MarkerArray.h>
#include <vision_msgs/VisionInfo.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/Detection3DArray.h>
#include <vision_msgs/ObjectHypothesisWithPose.h>

#include "object_detection_openvino/objectDetectionOpenvino.h"

/* Initialize the subscribers, the publishers and the inference engine */
ObjectDetectionOpenvino::ObjectDetectionOpenvino(ros::NodeHandle& node, ros::NodeHandle& node_private): node_(node), nodePrivate_(node_private), imageTransport_(nodePrivate_), 
																										syncTwoImage_(SyncPolicyTwoImage(5), colorSub_, depthSub_){
	// Initialize ROS parameters
	getParams();

	ROS_INFO_STREAM("[Object detection Openvino]: Loading Inference Engine" << InferenceEngine::GetInferenceEngineVersion());
	ROS_INFO_STREAM("[Object detection Openvino]: Device info: " << core_.GetVersions(deviceTarget_));

	// Load extensions for the plugin 
#ifdef WITH_EXTENSIONS
	if(deviceTarget_.find("CPU") != std::string::npos){
		/**
		 * cpu_extensions library is compiled from "extension" folder containing
		 * custom MKLDNNPlugin layer implementations. These layers are not supported
		 * by mkldnn, but they can be useful for inferring custom topologies.
		**/
		core_.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
	}else if(deviceTarget_.find("GPU") != std::string::npos){
		core_.SetConfig({{PluginConfigParams::KEY_DUMP_KERNELS, PluginConfigParams::YES}}, "GPU");
	}
#endif

	// Initialize values for depth analysis
	if(depthTopic_.empty()) useDepth_ = false;
	else useDepth_ = true;

	// Initialize subscribers, create sync policy and synchronizer
	infoSub_ = node_.subscribe<sensor_msgs::CameraInfo>(infoTopic_, 1, &ObjectDetectionOpenvino::infoCallback, this);
	colorSub_.subscribe(imageTransport_, colorTopic_, 10);
	if(!useDepth_){
		colorSub_.registerCallback(boost::bind(&ObjectDetectionOpenvino::oneImageCallback, this,_1));
	}else{
		depthSub_.subscribe(imageTransport_, depthTopic_, 10);
		syncTwoImage_.connectInput(colorSub_, depthSub_);
		syncTwoImage_.registerCallback(boost::bind(&ObjectDetectionOpenvino::twoImageCallback, this,_1,_2));
	}

	// Initialize publishers
	detectionColorPub_ = imageTransport_.advertise(detectionImageTopic_, 1);
	detectionInfoPub_ = nodePrivate_.advertise<vision_msgs::VisionInfo>(detectionInfoTopic_, 1);
	if(!useDepth_){
		detectionsPub_ = nodePrivate_.advertise<vision_msgs::Detection2DArray>(detectionsTopic_, 1);
	}else{
		detectionsPub_ = nodePrivate_.advertise<vision_msgs::Detection3DArray>(detectionsTopic_, 1);
		markersPub_ = nodePrivate_.advertise<visualization_msgs::MarkerArray>("markers", 1);
	}

	/* Read IR generated by the Model Optimizer (.xml, .bin, .labels files) */
	// Read network model
	ROS_INFO("[Object detection Openvino]: Loading network files");
	if(!boost::filesystem::exists(modelFileName_)){
		ROS_FATAL("[Object detection Openvino]: Network file doesn't exist.");
		ros::shutdown();
	}
	cnnNetwork_ = core_.ReadNetwork(modelFileName_, binFileName_);
	// Read labels (if any)
	std::ifstream inputFile(labelFileName_);
	std::copy(std::istream_iterator<std::string>(inputFile), std::istream_iterator<std::string>(), std::back_inserter(this->labels_));

	/* Configuring input and output */
	// Prepare input blobs
	ROS_INFO("[Object detection Openvino]: Checking that the inputs are as expected");
	inputInfo_ = InferenceEngine::InputsDataMap(cnnNetwork_.getInputsInfo());
	if(networkType_ == "YOLO"){
		if(inputInfo_.size() != 1){
			ROS_FATAL("[Object detection Openvino]: Only accepts networks that have only one input");
			ros::shutdown();
		}
		InferenceEngine::InputInfo::Ptr& input = inputInfo_.begin()->second;
		inputName_ = inputInfo_.begin()->first;
		input->setPrecision(InferenceEngine::Precision::U8);
		input->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
	}else if(networkType_ == "SSD"){
		if(inputInfo_.size() != 1 && inputInfo_.size() != 2 ){
			ROS_FATAL("[Object detection Openvino]: Only accepts networks with 1 or 2 inputs");
			ros::shutdown();
		}
		for(auto &input : inputInfo_){
			// First input contains images
			if(input.second->getTensorDesc().getDims().size() == 4){
				inputName_ = input.first;
				input.second->setPrecision(InferenceEngine::Precision::U8);
				input.second->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
			// Second input contains image info
			}else if (input.second->getTensorDesc().getDims().size() == 2){
				inputName_ = input.first;
				input.second->setPrecision(InferenceEngine::Precision::FP32);
			}else{
				throw std::logic_error("[Object detection Openvino]: Unsupported " + std::to_string(input.second->getTensorDesc().getDims().size()) + "D "
										"input layer '" + input.first + "'. Only 2D and 4D input layers are supported");
				ros::shutdown();
			}
		}
	}

	// Set batch size to 1
	ROS_INFO("[Object detection Openvino]: Batch size is forced to  1");
	InferenceEngine::ICNNNetwork::InputShapes inputShapes = cnnNetwork_.getInputShapes();
	InferenceEngine::SizeVector& inSizeVector = inputShapes.begin()->second;
	inSizeVector[0] = 1; 
	cnnNetwork_.reshape(inputShapes);

	// Prepare output blobs
	ROS_INFO("[Object detection Openvino]: Checking that the outputs are as expected");
	outputInfo_ = InferenceEngine::OutputsDataMap(cnnNetwork_.getOutputsInfo());
	if(networkType_ == "YOLO"){
		if(outputInfo_.size() != 3 && outputInfo_.size() != 2){
			ROS_FATAL("[Object detection Openvino]: Only accepts networks with three (YOLO) or two (tiny-YOLO) outputs");
			ros::shutdown();
		}

		for(auto &output : outputInfo_){
			output.second->setPrecision(InferenceEngine::Precision::FP32);
			output.second->setLayout(InferenceEngine::Layout::NCHW);
		}
	}else if(networkType_ == "SSD"){
		if(outputInfo_.size() != 1){
			throw std::logic_error("[Object detection Openvino]: Only accepts networks with one output");
		}

		for(auto &output : outputInfo_){
			output.second->setPrecision(InferenceEngine::Precision::FP32);
			output.second->setLayout(InferenceEngine::Layout::NCHW);
		}
	}

	// Load model to the device 
	ROS_INFO("[Object detection Openvino]: Loading model to the device");
	InferenceEngine::ExecutableNetwork network = core_.LoadNetwork(cnnNetwork_, deviceTarget_);

	// Create inference request
	ROS_INFO("[Object detection Openvino]: Create infer request");
	asyncInferRequestCurr_ = network.CreateInferRequestPtr();
	asyncInferRequestNext_ = network.CreateInferRequestPtr();
}

/* Delete all parameteres */
ObjectDetectionOpenvino::~ObjectDetectionOpenvino(){
	nodePrivate_.deleteParam("model_thresh");
	nodePrivate_.deleteParam("model_iou_thresh");

	nodePrivate_.deleteParam("model_xml");
	nodePrivate_.deleteParam("model_bin");
	nodePrivate_.deleteParam("model_labels");
	nodePrivate_.deleteParam("model_type");
	nodePrivate_.deleteParam("device_target");

	nodePrivate_.deleteParam("info_topic");
	nodePrivate_.deleteParam("color_topic");
	nodePrivate_.deleteParam("depth_topic");
	nodePrivate_.deleteParam("detection_image_topic");
	nodePrivate_.deleteParam("detection_info_topic");
	nodePrivate_.deleteParam("detections_topic");

	nodePrivate_.deleteParam("show_fps");
	nodePrivate_.deleteParam("output_image");
}

/* Update parameters of the node */
void ObjectDetectionOpenvino::getParams(){
	ROS_INFO("[Object detection Openvino]: Reading ROS parameters");

	nodePrivate_.param<float>("model_thresh", thresh_, 0.3);
	nodePrivate_.param<float>("model_iou_thresh", iouThresh_, 0.4);

	nodePrivate_.param<std::string>("model_xml", modelFileName_, "");
	nodePrivate_.param<std::string>("model_bin", binFileName_, "");
	nodePrivate_.param<std::string>("model_labels", labelFileName_, "");
	nodePrivate_.param<std::string>("model_type", networkType_, "");
	nodePrivate_.param<std::string>("device_target", deviceTarget_, "CPU");

	nodePrivate_.param<std::string>("info_topic", infoTopic_, "/camera/info");
	nodePrivate_.param<std::string>("color_topic", colorTopic_, "/camera/color/image_raw");
	nodePrivate_.param<std::string>("depth_topic", depthTopic_, "");
	nodePrivate_.param<std::string>("detection_image_topic", detectionImageTopic_, "image_raw");
	nodePrivate_.param<std::string>("detection_info_topic", detectionInfoTopic_, "detection_info");
	nodePrivate_.param<std::string>("detections_topic", detectionsTopic_, "detections");

	nodePrivate_.param<bool>("show_fps", showFPS_, false);
	nodePrivate_.param<bool>("output_image", outputImage_, true);
}

/* Camera info Callback */
void ObjectDetectionOpenvino::infoCallback(const sensor_msgs::CameraInfo::ConstPtr& infoMsg){
	ROS_INFO_ONCE("[Object detection Openvino]: Subscribed to camera info topic: %s", infoTopic_.c_str());

	// Read parameters of the camera
	fx_ = infoMsg->K[0];
	fy_ = infoMsg->K[4];
	cx_ = infoMsg->K[2];
	cy_ = infoMsg->K[5];

	// Create info
	vision_msgs::VisionInfo detectionInfo;
	detectionInfo.header = infoMsg->header;
	detectionInfo.method = networkType_ + " detection with COCO database";
	detectionInfo.database_location = labelFileName_;
	detectionInfo.database_version = 0;

	// Publish info
	detectionInfoPub_.publish(detectionInfo);
}

/* Callback function for color image */
void ObjectDetectionOpenvino::oneImageCallback(sensor_msgs::Image::ConstPtr colorImageMsg){
	std::vector<sensor_msgs::Image::ConstPtr> imageVec;
	imageVec.push_back(colorImageMsg);
	cameraCallback(imageVec);
}

/* Callback function for color and depth */
void ObjectDetectionOpenvino::twoImageCallback(sensor_msgs::Image::ConstPtr colorImageMsg, sensor_msgs::Image::ConstPtr depthImageMsg){
	std::vector<sensor_msgs::Image::ConstPtr> imageVec;
	imageVec.push_back(colorImageMsg);
	imageVec.push_back(depthImageMsg);
	cameraCallback(imageVec);
}

/* Camera Callback */
void ObjectDetectionOpenvino::cameraCallback(const std::vector<sensor_msgs::Image::ConstPtr>& imageMsg){
	sensor_msgs::Image::ConstPtr colorImageMsg, depthImageMsg;
	cv_bridge::CvImagePtr colorImageCv, depthImageCv;

	ROS_INFO_ONCE("[Object detection Openvino]: Subscribed to color image topic: %s", colorTopic_.c_str());

	// Note: Only infer object if there's any subscriber
	if(detectionInfoPub_.getNumSubscribers() == 0 && detectionColorPub_.getNumSubscribers() == 0 && detectionsPub_.getNumSubscribers() == 0) return;

	// Read header
	colorImageMsg = imageMsg[0];
	colorFrameId_ = colorImageMsg->header.frame_id;

	// Create arrays to publish and format headers
	vision_msgs::Detection2DArray detections2D;
	detections2D.header.frame_id = colorFrameId_;
	detections2D.header.stamp = colorImageMsg->header.stamp;

	vision_msgs::Detection3DArray detections3D;
	detections3D.header.frame_id = colorFrameId_;
	detections3D.header.stamp = colorImageMsg->header.stamp;

	visualization_msgs::MarkerArray markerArray;

	auto wallclock = std::chrono::high_resolution_clock::now();

	// Convert from ROS to CV image
	try{
		colorImageCv = cv_bridge::toCvCopy(colorImageMsg, sensor_msgs::image_encodings::BGR8);
	}catch(cv_bridge::Exception& e){
		ROS_ERROR("[Object detection Openvino]: cv_bridge exception: %s", e.what());
		return;
	}
	const size_t colorHeight = (size_t) colorImageCv->image.size().height;
	const size_t colorWidth  = (size_t) colorImageCv->image.size().width;

	// Copy data from image to input blob
	nextFrame_ = colorImageCv->image.clone();
	frameToBlob(nextFrame_, asyncInferRequestNext_, inputName_, false);

	/* Perform depth analysis */
	if(useDepth_){
		ROS_INFO_ONCE("[Object detection Openvino]: Subscribed to depth image topic: %s", depthTopic_.c_str());

		// Read header
		depthImageMsg = imageMsg[1];

		// Convert from ROS to CV image
		try{
			depthImageCv = cv_bridge::toCvCopy(depthImageMsg, sensor_msgs::image_encodings::TYPE_16UC1);
		}catch(cv_bridge::Exception& e){
			ROS_ERROR("[Object detection Openvino]: cv_bridge exception: %s", e.what());
			return;
		}
	}

	// Load network
	// In the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
	auto t0 = std::chrono::high_resolution_clock::now();
	asyncInferRequestNext_->StartAsync();

	if(InferenceEngine::OK == asyncInferRequestCurr_->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY)){
		// Show FPS
		if(showFPS_ && outputImage_){
			auto t1 = std::chrono::high_resolution_clock::now();
			ms detection = std::chrono::duration_cast<ms>(t1 - t0);

			t0 = std::chrono::high_resolution_clock::now();
			ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
			wallclock = t0;

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

		// Processing output blobs of the CURRENT request
		const InferenceEngine::TensorDesc& inputDesc = inputInfo_.begin()->second.get()->getTensorDesc();
		unsigned long resizedImgH = getTensorHeight(inputDesc);
		unsigned long resizedImgW = getTensorWidth(inputDesc);

		// Parsing outputs
		std::vector<DetectionObject> objects;
		for(auto &output: outputInfo_){
			auto outputName = output.first;
			InferenceEngine::Blob::Ptr blob = asyncInferRequestCurr_->GetBlob(outputName);

			if(networkType_ == "YOLO") parseYOLOV3Output(cnnNetwork_, outputName, blob, resizedImgH, resizedImgW, colorHeight, colorWidth, thresh_, objects);
			else if(networkType_ == "SSD") parseSSDOutput(blob, colorHeight, colorWidth, thresh_, objects);
		}

		// Filtering overlapping boxes
		std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
		for(int i = 0; i < objects.size(); ++i){
			if (objects[i].confidence == 0)
				continue;
			for (int j = i + 1; j < objects.size(); ++j) {
				if (intersectionOverUnion(objects[i], objects[j]) >= iouThresh_) {
					objects[j].confidence = 0;
				}
			}
		}

		/* Process objects */
		for(auto &object: objects){
			// Skip if confidence is less than the threshold
			if(object.confidence < thresh_) continue;

			auto label = object.classId;
			float confidence = object.confidence;

			ROS_DEBUG("[Object detection Openvino]: %s tag (%.2f%%)", this->labels_[label].c_str(), confidence*100);

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

			// Publish 2D or 3D messages
			if(!useDepth_){
				// Create detection2D and push to array
				vision_msgs::Detection2D detection2D = createDetection2DMsg(object, detections2D.header);
				detections2D.detections.push_back(detection2D);
			}else{
				int detectionId = 0;
				// Create detection3D and push to array
				vision_msgs::Detection3D detection3D = createDetection3DMsg(depthImageCv, object, detections2D.header);
				detections3D.detections.push_back(detection3D);

				// Create markers
				visualization_msgs::Marker vizMarker = createBBox3dMarker(detectionId, detection3D.bbox.center, detection3D.bbox.size, colorRGB, detections2D.header);
				visualization_msgs::Marker labelMarker = createLabel3dMarker(detectionId*10, this->labels_[label].c_str(), detection3D.bbox.center, colorRGB, detections2D.header);
				markerArray.markers.push_back(vizMarker);
				markerArray.markers.push_back(labelMarker);
			}

			/* Image */
			if(outputImage_){
				// Text label
				std::ostringstream conf;
				conf << ":" << std::fixed << std::setprecision(3) << confidence;
				std::string labelText = (label < this->labels_.size() ? this->labels_[label] : std::string("label #") + std::to_string(label)) + conf.str();
				// Rectangles for class
				cv::rectangle(currFrame_, cv::Point2f(object.xmin-1, object.ymin), cv::Point2f(object.xmin + 180, object.ymin - 22), cv::Scalar(colorRGB[2], colorRGB[1], colorRGB[0]), cv::FILLED, cv::LINE_AA);
				cv::putText(currFrame_, labelText, cv::Point2f(object.xmin, object.ymin - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0), 1.5, cv::LINE_AA);
				cv::rectangle(currFrame_, cv::Point2f(object.xmin, object.ymin), cv::Point2f(object.xmax, object.ymax), cv::Scalar(colorRGB[2], colorRGB[1], colorRGB[0]), 4, cv::LINE_AA);
			}
		}
	}

	// Publish
	if(outputImage_) publishImage(currFrame_);
	if(!useDepth_){
		detectionsPub_.publish(detections2D);
	}else{
		detectionsPub_.publish(detections3D);
		markersPub_.publish(markerArray);
		markerArray.markers.clear();
	}

	// In the truly Async mode we swap the NEXT and CURRENT requests for the next iteration
	currFrame_ = nextFrame_;
	nextFrame_ = cv::Mat();
	asyncInferRequestCurr_.swap(asyncInferRequestNext_);
}

/* Show histogram of the image */
void ObjectDetectionOpenvino::showHistogram(cv::Mat image, cv::Scalar mean){
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

/* Create detection 2D message */
vision_msgs::Detection2D ObjectDetectionOpenvino::createDetection2DMsg(DetectionObject object, std_msgs::Header header){
	vision_msgs::Detection2D detection2D;
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

	detection2D.source_img.header = detection2D.header;
	detection2D.source_img.height = croppedImage.rows;
	detection2D.source_img.width = croppedImage.cols;
	detection2D.source_img.encoding = "bgr8";
	detection2D.source_img.is_bigendian = false;
	detection2D.source_img.step = croppedImage.cols * 3;
	size_t size = detection2D.source_img.step * croppedImage.rows;
	detection2D.source_img.data.resize(size);
	memcpy((char*)(&detection2D.source_img.data[0]), croppedImage.data, size);

	return detection2D;
}

/* Create detection 3D message */
vision_msgs::Detection3D ObjectDetectionOpenvino::createDetection3DMsg(cv_bridge::CvImagePtr depthImage, DetectionObject object, std_msgs::Header header){
	vision_msgs::Detection3D detection3D;
	detection3D.header = header;

	// Class probabilities
	vision_msgs::ObjectHypothesisWithPose hypo;
	hypo.id = object.classId;
	hypo.score = object.confidence;
	detection3D.results.push_back(hypo);

	/* Perform 3d analysis */
	// TODO: Use grabCut to improve segmentation
	// Extract bounding box of the object in the depth image
	cv::Mat depthFrame = depthImage->image.clone();
	cv::Mat subDepth = depthFrame(cv::Rect(object.xmin, object.ymin, object.xmax - object.xmin, object.ymax - object.ymin));
	// Change dataType
	cv::Mat subDepthSc;
	subDepth.convertTo(subDepthSc, CV_8UC1, 0.0390625);
	// Threshold to Zero Inverted by mean (every value above mean, will be set to zero)
	cv::Mat subDepthThres;
	cv::threshold(subDepthSc, subDepthThres, cv::mean(subDepthSc)[0], 255, cv::THRESH_TOZERO_INV);
	// Find contours
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(subDepthThres, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0,0));
	for (int i = 0; i < contours.size(); i++){
		cv::drawContours(subDepthThres, contours, i, 0xffff, cv::FILLED, cv::LINE_8, hierarchy, 0, cv::Point());
	}
	// Change dataType and rescale
	subDepthThres.convertTo(subDepthThres, CV_16UC1);
	subDepthThres += subDepthThres * 256;
	// Apply mask
	cv::bitwise_and(subDepthThres, subDepth, subDepth);
	cv::Mat mask = cv::Mat(subDepth != 0);
	// Extract average and standard deviation
	cv::Scalar avg, dstd;
	cv::meanStdDev(subDepth, avg, dstd, mask);

	// If you want to see the histogram of the depth image uncomment next line
	//showHistogram(subDepthSc, cv::mean(subDepthSc));

	ROS_DEBUG("The depth of the bounding box center is %f", 0.001 * (float)depthImage->image.at<u_int16_t>( cv::Point((object.ymax + object.ymin) / 2.0, 
															(object.xmax + object.xmin) / 2.0)));

	// Extract 3d coordinates of the average value
	float avgPixelDepthX, avgPixelDepthY, avgPixelDepthZ;
	avgPixelDepthZ = avg[0]/1000.0;
	avgPixelDepthX = ((object.xmax + object.xmin) / 2.0 - cx_) * avgPixelDepthZ / fx_;
	avgPixelDepthY = ((object.ymax + object.ymin) / 2.0 - cy_) * avgPixelDepthZ / fy_;

	// 3D bounding box surrounding the object
	detection3D.bbox.center.position.x = avgPixelDepthX;
	detection3D.bbox.center.position.y = avgPixelDepthY;
	detection3D.bbox.center.position.z = avgPixelDepthZ;
	detection3D.bbox.center.orientation.x = 0.0;
	detection3D.bbox.center.orientation.y = 0.0;
	detection3D.bbox.center.orientation.z = 0.0;
	detection3D.bbox.center.orientation.w = 1.0;
	detection3D.bbox.size.x = (object.xmax - object.xmin) * avgPixelDepthZ / fx_;
	detection3D.bbox.size.y = (object.ymax - object.ymin) * avgPixelDepthZ / fy_;
	detection3D.bbox.size.z = dstd[0] * 2.0 / 1000.0;

	// TODO: The 3D data that generated these results

	return detection3D;
}

/* Create 3d Bounding Box for the object */
visualization_msgs::Marker ObjectDetectionOpenvino::createBBox3dMarker(int id, geometry_msgs::Pose center, geometry_msgs::Vector3 size, float colorRGB[3], std_msgs::Header header){
	visualization_msgs::Marker marker;
	marker.header = header;
	marker.ns = "boundingBox3d";
	marker.id = id;
	marker.type = visualization_msgs::Marker::CUBE;
	marker.action = visualization_msgs::Marker::ADD;
	marker.lifetime = ros::Duration(0.15);
	marker.pose = center;
	marker.scale = size;
	marker.color.r = colorRGB[0] / 255.0;
	marker.color.g = colorRGB[1] / 255.0;
	marker.color.b = colorRGB[2] / 255.0;
	marker.color.a = 0.2f;

	return marker;
}

/* Create 3d label for the object */
visualization_msgs::Marker ObjectDetectionOpenvino::createLabel3dMarker(int id, std::string label, geometry_msgs::Pose pose, float colorRGB[3], std_msgs::Header header){
	visualization_msgs::Marker marker;
	marker.header = header;
	marker.ns = "label3d";
	marker.id = id;
	marker.text = label;
	marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
	marker.action = visualization_msgs::Marker::ADD;
	marker.lifetime = ros::Duration(0.15);
	marker.pose.position.x = pose.position.x;
	marker.pose.position.y = pose.position.y - 0.3;
	marker.pose.position.z = pose.position.z + 0.05;
	marker.pose.orientation.x = 0.0;
	marker.pose.orientation.y = 0.0;
	marker.pose.orientation.z = 0.0;
	marker.pose.orientation.w = 1.0;
	marker.scale.z = 0.3;
	marker.color.r = colorRGB[0] / 255.0;
	marker.color.g = colorRGB[1] / 255.0;
	marker.color.b = colorRGB[2] / 255.0;
	marker.color.a = 0.8f;

	return marker;
}

/* Publish image */
void ObjectDetectionOpenvino::publishImage(cv::Mat image){
	cv_bridge::CvImage outputImageMsg;
	outputImageMsg.header.stamp = ros::Time::now();
	outputImageMsg.header.frame_id = colorFrameId_;
	outputImageMsg.encoding = sensor_msgs::image_encodings::BGR8;
	outputImageMsg.image = image;

	detectionColorPub_.publish(outputImageMsg.toImageMsg());
}

//----- OpenVino related -----------------------

/* Get color of the class */
int ObjectDetectionOpenvino::getColor(int c, int x, int max){
	float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

	float ratio = ((float)x/max)*5;
	int i = floor(ratio);
	int j = ceil(ratio);
	ratio -= i;
	float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];

	return floor(r*255);
}

/* Index for the entry */
int ObjectDetectionOpenvino::entryIndex(int side, int lcoords, int lclasses, int location, int entry) {
	int n = location / (side * side);
	int loc = location % (side * side);
	return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

/* Intersection of bounding boxes */
double ObjectDetectionOpenvino::intersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2) {
	double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
	double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
	double area_of_overlap;

	if (width_of_overlap_area < 0 || height_of_overlap_area < 0){
		area_of_overlap = 0;
	}else{
		area_of_overlap = width_of_overlap_area * height_of_overlap_area;
	}

	double box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin);
	double box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin);
	double area_of_union = box_1_area + box_2_area - area_of_overlap;

	return area_of_overlap / area_of_union;
}

/* Convert frame to blob */
void ObjectDetectionOpenvino::frameToBlob(const cv::Mat &frame, InferenceEngine::InferRequest::Ptr &inferRequest, const std::string &inputName, bool autoResize){
	if(autoResize){
		// Just set input blob containing read image. Resize and layout conversion will be done automatically 
		inferRequest->SetBlob(inputName, wrapMat2Blob(frame));
	}else{
		// Resize and copy data from the image to the input blob
		InferenceEngine::Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
		matU8ToBlob<uint8_t>(frame, frameBlob);
	}
}

/* Parse Mobiletnet SSD output */
void ObjectDetectionOpenvino::parseSSDOutput(const InferenceEngine::Blob::Ptr &blob, const unsigned long height, const unsigned long width, const float threshold,  std::vector<DetectionObject> &objects){
	// Validating output parameters
	InferenceEngine::SizeVector outputDims = blob->getTensorDesc().getDims();
	int maxProposalCount = static_cast<int>(blob->getTensorDesc().getDims()[2]);
	const int objectSize = static_cast<int>(blob->getTensorDesc().getDims()[3]);

	if(objectSize != 7){
		throw std::logic_error("[Object detection Openvino]: Output should have 7 as a last dimension");
	}
	if(outputDims.size() != 4){
		throw std::logic_error("[Object detection Openvino]: Incorrect output dimensions for SSD");
	}

	const float *outputBlob = blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();
	for(int i = 0; i < maxProposalCount; i++){
		float id = outputBlob[i * objectSize + 0];

		if(id < 0) break;

		auto label = static_cast<int>(outputBlob[i * objectSize + 1]);
		float prob = outputBlob[i * objectSize + 2];
		float xmin = outputBlob[i * objectSize + 3] * width;
		float ymin = outputBlob[i * objectSize + 4] * height;
		float xmax = outputBlob[i * objectSize + 5] * width;
		float ymax = outputBlob[i * objectSize + 6] * height;

		double newWidth = xmax - xmin;
		double newHeight = ymax - ymin;

		if(prob < threshold) continue;

		DetectionObject obj(xmin, ymin, newHeight, newWidth, label, this->labels_[label], prob);
		objects.push_back(obj);
	}
}

/* Parse Yolo v3 output*/
void ObjectDetectionOpenvino::parseYOLOV3Output(const InferenceEngine::CNNNetwork &cnnNetwork, const std::string &outputName, const InferenceEngine::Blob::Ptr &blob, const unsigned long resizedImgH, const unsigned long resizedImgW, const unsigned long originalImgH, const unsigned long originalImgW, const float threshold,  std::vector<DetectionObject> &objects) {
	// Validating output parameters 
	const int outBlobH = static_cast<int>(blob->getTensorDesc().getDims()[2]);
	const int outBlobW = static_cast<int>(blob->getTensorDesc().getDims()[3]);
	if(outBlobH != outBlobW){
		throw std::runtime_error("[Object detection Openvino]: Invalid size of output " + outputName +
		" It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(outBlobH) +
		", current W = " + std::to_string(outBlobW));
	}

	// Extracting layer parameters
	YoloParams params;
	if(auto ngraphFunction = cnnNetwork.getFunction()){
		for(const auto op: ngraphFunction->get_ops()){
			if(op->get_friendly_name() == outputName){
				auto regionYolo = std::dynamic_pointer_cast<ngraph::op::RegionYolo>(op);
				if(!regionYolo){
					throw std::runtime_error("Invalid output type: " +
					std::string(regionYolo->get_type_info().name) + ". RegionYolo expected");
				}

				params = regionYolo;
				break;
			}
		}
	}else{
		throw std::runtime_error("Can't get ngraph::Function. Make sure the provided model is in IR version 10 or greater.");
	}

	auto side = outBlobH;
	auto sideSquare = side * side;
	const float *outputBlob = blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();

	// Parsing YOLO Region output 
	for(int i = 0; i < sideSquare; ++i){
		int row = i / side;
		int col = i % side;
		for(int n = 0; n < params.num; ++n){
			int objIdx = entryIndex(side, params.coords, params.classes, n * side * side + i, params.coords);
			int boxIdx = entryIndex(side, params.coords, params.classes, n * side * side + i, 0);
			float scale = outputBlob[objIdx];

			if(scale < threshold) continue;

			double x = (col + outputBlob[boxIdx + 0 * sideSquare]) / side * resizedImgW;
			double y = (row + outputBlob[boxIdx + 1 * sideSquare]) / side * resizedImgH;
			double height = std::exp(outputBlob[boxIdx + 3 * sideSquare]) * params.anchors[2 * n + 1];
			double width = std::exp(outputBlob[boxIdx + 2 * sideSquare]) * params.anchors[2 * n];
			for(int j = 0; j < params.classes; ++j) {
				int classIdx = entryIndex(side, params.coords, params.classes, n * sideSquare + i, params.coords + 1 + j);
				float prob = scale * outputBlob[classIdx];

				if(prob < threshold) continue;

				DetectionObject obj(x, y, height, width, j, this->labels_[j], prob,
					static_cast<float>(originalImgH) / static_cast<float>(resizedImgH),
					static_cast<float>(originalImgW) / static_cast<float>(resizedImgW));
				objects.push_back(obj);
			}
		}
	}
}

//----------------------------------------------

/* Main */
int main(int argc, char** argv){
	ros::init(argc, argv, "object_detection_openvino");
	ros::NodeHandle node("");
	ros::NodeHandle node_private("~");

	try{
		ROS_INFO("[Object detection Openvino]: Initializing node");
		ObjectDetectionOpenvino detector(node, node_private);
		ros::spin();
	}catch(const char* s){
		ROS_FATAL_STREAM("[Object detection Openvino]: " << s);
	}catch(...){
		ROS_FATAL_STREAM("[Object detection Openvino]: Unexpected error");
	}
}
