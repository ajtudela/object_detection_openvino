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
#include <vision_msgs/VisionInfo.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/ObjectHypothesisWithPose.h>

#include "object_detection_openvino/objectDetectionOpenvino.hpp"

/* Initialize the subscribers, the publishers and the inference engine */
ObjectDetectionOpenvino::ObjectDetectionOpenvino(ros::NodeHandle& node, ros::NodeHandle& node_private): node_(node), nodePrivate_(node_private), imageTransport_(nodePrivate_){
	// Initialize ROS parameters
	ROS_INFO("[Object detection Openvino]: Reading ROS parameters");
	paramsSrv_ = nodePrivate_.advertiseService("params", &ObjectDetectionOpenvino::updateParams, this);

	initialize();

	ROS_INFO_STREAM("[Object detection Openvino]: Loading Inference Engine" << InferenceEngine::GetInferenceEngineVersion());
	ROS_INFO_STREAM("[Object detection Openvino]: Device info: " << core_.GetVersions(deviceTarget_));

	// Load extensions for the plugin 
#ifdef WITH_EXTENSIONS
	if (deviceTarget_.find("CPU") != std::string::npos) {
		/**
		 * cpu_extensions library is compiled from "extension" folder containing
		 * custom MKLDNNPlugin layer implementations. These layers are not supported
		 * by mkldnn, but they can be useful for inferring custom topologies.
		**/
		core_.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
	}
#endif

	// Initialize subscribers
	infoSub_ = node_.subscribe<sensor_msgs::CameraInfo>(infoTopic_, 1, &ObjectDetectionOpenvino::infoCallback, this);
	colorSub_ = imageTransport_.subscribe(colorTopic_, 1, &ObjectDetectionOpenvino::cameraCallback, this);

	// Initialize publishers
	detectionColorPub_ = imageTransport_.advertise(detectionImageTopic_, 1);
	detectionInfoPub_ = nodePrivate_.advertise<vision_msgs::VisionInfo>(detectionInfoTopic_, 1);
	detection2DPub_ = nodePrivate_.advertise<vision_msgs::Detection2DArray>(detection2DTopic_, 1);

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

		if (auto ngraphFunction = cnnNetwork_.getFunction()){
			for (const auto op : ngraphFunction->get_ops()) {
				auto outputLayer = outputInfo_.find(op->get_friendly_name());
				if (outputLayer != outputInfo_.end()) {
					auto regionYolo = std::dynamic_pointer_cast<ngraph::op::RegionYolo>(op);
					if (!regionYolo) {
						throw std::runtime_error("[Object detection Openvino]: Invalid output type: " +
							std::string(regionYolo->get_type_info().name) + ". RegionYolo expected");
					}
					yoloParams_[outputLayer->first] = YoloParams(regionYolo);
				}
			}
		}
		else {
			ROS_FATAL("[Object detection Openvino]: Can't get ngraph::Function. Make sure the provided model is in IR version 10 or greater.");
			ros::shutdown();
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
	nodePrivate_.deleteParam("detection_image_topic");
	nodePrivate_.deleteParam("detection_info_topic");
	nodePrivate_.deleteParam("detection_2d_topic");

	nodePrivate_.deleteParam("show_fps");
	nodePrivate_.deleteParam("output_image");
}

/* Update parameters of the node */
bool ObjectDetectionOpenvino::updateParams(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res){
	nodePrivate_.param<float>("model_thresh", thresh_, 0.3);
	nodePrivate_.param<float>("model_iou_thresh", iouThresh_, 0.4);

	nodePrivate_.param<std::string>("model_xml", modelFileName_, "-");
	nodePrivate_.param<std::string>("model_bin", binFileName_, "-");
	nodePrivate_.param<std::string>("model_labels", labelFileName_, "-");
	nodePrivate_.param<std::string>("model_type", networkType_, "-");
	nodePrivate_.param<std::string>("device_target", deviceTarget_, "CPU");

	nodePrivate_.param<std::string>("info_topic", infoTopic_, "/camera/info");
	nodePrivate_.param<std::string>("color_topic", colorTopic_, "/camera/color/image_raw");
	nodePrivate_.param<std::string>("detection_image_topic", detectionImageTopic_, "image_raw");
	nodePrivate_.param<std::string>("detection_info_topic", detectionInfoTopic_, "detection_info");
	nodePrivate_.param<std::string>("detection_2d_topic", detection2DTopic_, "detections_2d");

	nodePrivate_.param<bool>("show_fps", showFPS_, false);
	nodePrivate_.param<bool>("output_image", outputImage_, true);

	return true;
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

/* Camera Callback */
void ObjectDetectionOpenvino::cameraCallback(const sensor_msgs::Image::ConstPtr& colorImageMsg){
	ROS_INFO_ONCE("[Object detection Openvino]: Subscribed to color image topic: %s", colorTopic_.c_str());

	// Note: Only infer object if there's any subscriber
	if(detectionInfoPub_.getNumSubscribers() == 0 && !detectionColorPub_.getNumSubscribers() == 0 && !detection2DPub_.getNumSubscribers() == 0) return;

	// Read header
	colorFrameId_ = colorImageMsg->header.frame_id;

	// Create array to publish
	vision_msgs::Detection2DArray detections2D;
	detections2D.header.frame_id = colorFrameId_;
	detections2D.header.stamp = colorImageMsg->header.stamp;

	auto wallclock = std::chrono::high_resolution_clock::now();

	// Convert from ROS to CV image
	cv_bridge::CvImagePtr colorFrame;
	try{
		colorFrame = cv_bridge::toCvCopy(colorImageMsg, sensor_msgs::image_encodings::BGR8);
	}catch(cv_bridge::Exception& e){
		ROS_ERROR("[Object detection Openvino]: cv_bridge exception: %s", e.what());
		return;
	}
	const size_t width  = (size_t) colorFrame->image.size().width;
	const size_t height = (size_t) colorFrame->image.size().height;

	// Copy data from image to input blob
	nextFrame_ = colorFrame->image.clone();
	frameToBlob(nextFrame_, asyncInferRequestNext_, inputName_, false);

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
			InferenceEngine::CNNLayerPtr layer = cnnNetwork_.getLayerByName(outputName.c_str());
			InferenceEngine::Blob::Ptr blob = asyncInferRequestCurr_->GetBlob(outputName);

			if(networkType_ == "YOLO") parseYOLOV3Output(yoloParams_[outputName], outputName, blob, resizedImgH, resizedImgW, height, width, thresh_, objects);
			else if(networkType_ == "SSD") parseSSDOutput(layer, blob, height, width, thresh_, objects);
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
			object.xmax = object.xmax > width ? width : object.xmax;
			object.ymax = object.ymax > height ? height : object.ymax;

			/* Create detection2D */
			vision_msgs::Detection2D detection2D;
			detection2D.header.frame_id = colorFrameId_;
			detection2D.header.stamp = colorImageMsg->header.stamp;

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
			cv::Rect rectCrop(object.xmin, object.ymin, object.xmax - object.xmin, object.ymax - object.ymin);
			cv::Mat croppedImage = currFrame_(rectCrop);
			detection2D.source_img.header = detection2D.header;
			detection2D.source_img.height = croppedImage.rows;
			detection2D.source_img.width = croppedImage.cols;
			detection2D.source_img.encoding = "bgr8";
			detection2D.source_img.is_bigendian = false;
			detection2D.source_img.step = croppedImage.cols * 3;
			size_t size = detection2D.source_img.step * croppedImage.rows;
			detection2D.source_img.data.resize(size);
			memcpy((char*)(&detection2D.source_img.data[0]), croppedImage.data, size);

			// And push to array
			detections2D.detections.push_back(detection2D);

			/* Image */
			if(outputImage_){
				// Color of the class
				int offset = object.classId * 123457 % COCO_CLASSES;
				float colorRGB[3];
				colorRGB[0] = getColor(2,offset,COCO_CLASSES);
				colorRGB[1] = getColor(1,offset,COCO_CLASSES);
				colorRGB[2] = getColor(0,offset,COCO_CLASSES);
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
	detection2DPub_.publish(detections2D);
	if(outputImage_) publishImage(currFrame_);

	// In the truly Async mode we swap the NEXT and CURRENT requests for the next iteration
	currFrame_ = nextFrame_;
	nextFrame_ = cv::Mat();
	asyncInferRequestCurr_.swap(asyncInferRequestNext_);
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
void ObjectDetectionOpenvino::parseSSDOutput(const InferenceEngine::CNNLayerPtr &layer, const InferenceEngine::Blob::Ptr &blob, const unsigned long height, const unsigned long width, const float threshold,  std::vector<DetectionObject> &objects){
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
void ObjectDetectionOpenvino::parseYOLOV3Output(const YoloParams &params, const std::string &outputName, const InferenceEngine::Blob::Ptr &blob, const unsigned long resizedImgH, const unsigned long resizedImgW, const unsigned long originalImgH, const unsigned long originalImgW, const float threshold,  std::vector<DetectionObject> &objects) {
	// Validating output parameters 
	const int outBlobH = static_cast<int>(blob->getTensorDesc().getDims()[2]);
	const int outBlobW = static_cast<int>(blob->getTensorDesc().getDims()[3]);
	if(outBlobH != outBlobW){
		throw std::runtime_error("[Object detection Openvino]: Invalid size of output " + outputName +
		" It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(outBlobH) +
		", current W = " + std::to_string(outBlobW));
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
