/*
 * OPENVINO CLASS
 *
 * Copyright (c) 2020-2021 Alberto José Tudela Roldán <ajtudela@gmail.com>
 * 
 * This file is part of object_detection_openvino project.
 * 
 * All rights reserved.
 *
 */

// C++
#include <iostream>
#include <iterator>
#include <boost/filesystem.hpp>

// Openvino
#include <ngraph/ngraph.hpp>

// ROS
#include "rclcpp/rclcpp.hpp"

#include "object_detection_openvino/openvino.hpp"

Openvino::Openvino(std::string node_name) : node_name_(node_name){
}

Openvino::~Openvino(){
}

void Openvino::set_target_device(std::string device){
	RCLCPP_INFO_STREAM(rclcpp::get_logger(node_name_), 
		"Loading Inference Engine" << InferenceEngine::GetInferenceEngineVersion());
	RCLCPP_INFO_STREAM(rclcpp::get_logger(node_name_), 
		"Device info: " << core_.GetVersions(device));

	// Load extensions for the plugin 
#ifdef WITH_EXTENSIONS
	if (device.find("CPU") != std::string::npos){
		/**
		 * cpu_extensions library is compiled from "extension" folder containing
		 * custom MKLDNNPlugin layer implementations. These layers are not supported
		 * by mkldnn, but they can be useful for inferring custom topologies.
		**/
		core_.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
	}else if (device.find("GPU") != std::string::npos){
		core_.SetConfig({{PluginConfigParams::KEY_DUMP_KERNELS, PluginConfigParams::YES}}, "GPU");
	}
#endif
}

void Openvino::set_network_model(std::string model_filename, 
								std::string bin_filename, 
								std::string label_filename){
	// Read network model
	RCLCPP_INFO(rclcpp::get_logger(node_name_), "Loading network files");
	if (!boost::filesystem::exists(model_filename)){
		RCLCPP_FATAL(rclcpp::get_logger(node_name_), "Network file doesn't exist.");
		rclcpp::shutdown();
	}
	cnn_network_ = core_.ReadNetwork(model_filename, bin_filename);
	// Read labels (if any)
	std::ifstream input_file(label_filename);
	std::copy(std::istream_iterator<std::string>(input_file), 
			std::istream_iterator<std::string>(), 
			std::back_inserter(this->labels_));
}

void Openvino::configure_network(std::string network_type){
	network_type_ = network_type;

	// Prepare input blobs
	RCLCPP_INFO(rclcpp::get_logger(node_name_), "Checking that the inputs are as expected");

	input_info_ = InferenceEngine::InputsDataMap(cnn_network_.getInputsInfo());
	if (network_type_ == "YOLO"){
		if (input_info_.size() != 1){
			RCLCPP_FATAL(rclcpp::get_logger(node_name_), 
						"Only accepts networks that have only one input");
			rclcpp::shutdown();
		}
		InferenceEngine::InputInfo::Ptr& input = input_info_.begin()->second;
		input_name_ = input_info_.begin()->first;
		input->setPrecision(InferenceEngine::Precision::U8);
		input->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
	}else if (network_type_ == "SSD"){
		if (input_info_.size() != 1 && input_info_.size() != 2 ){
			RCLCPP_FATAL(rclcpp::get_logger(node_name_), 
						"Only accepts networks with 1 or 2 inputs");
			rclcpp::shutdown();
		}
		for (auto &input : input_info_){
			// First input contains images
			if (input.second->getTensorDesc().getDims().size() == 4){
				input_name_ = input.first;
				input.second->setPrecision(InferenceEngine::Precision::U8);
				input.second->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
			// Second input contains image info
			}else if (input.second->getTensorDesc().getDims().size() == 2){
				input_name_ = input.first;
				input.second->setPrecision(InferenceEngine::Precision::FP32);
			}else{
				throw std::logic_error("OpenVino: Unsupported "
					+ std::to_string(input.second->getTensorDesc().getDims().size())
					+ "D input layer '" + input.first
					+ "'. Only 2D and 4D input layers are supported");
				rclcpp::shutdown();
			}
		}
	}

	// Set batch size to 1
	RCLCPP_INFO(rclcpp::get_logger(node_name_), "Batch size is forced to  1");
	InferenceEngine::ICNNNetwork::InputShapes inputShapes = cnn_network_.getInputShapes();
	InferenceEngine::SizeVector& inSizeVector = inputShapes.begin()->second;
	inSizeVector[0] = 1; 
	cnn_network_.reshape(inputShapes);

	// Prepare output blobs
	RCLCPP_INFO(rclcpp::get_logger(node_name_), "Checking that the outputs are as expected");
	output_info_ = InferenceEngine::OutputsDataMap(cnn_network_.getOutputsInfo());
	if (network_type_ == "YOLO"){
		if (output_info_.size() != 3 && output_info_.size() != 2){
			RCLCPP_FATAL(rclcpp::get_logger(node_name_), 
				"Only accepts networks with three (YOLO) or two (tiny-YOLO) outputs");
			rclcpp::shutdown();
		}

		for (auto &output : output_info_){
			output.second->setPrecision(InferenceEngine::Precision::FP32);
			output.second->setLayout(InferenceEngine::Layout::NCHW);
		}
	}else if (network_type_ == "SSD"){
		if (output_info_.size() != 1){
			throw std::logic_error("Openvino: Only accepts networks with one output");
		}

		for (auto &output : output_info_){
			output.second->setPrecision(InferenceEngine::Precision::FP32);
			output.second->setLayout(InferenceEngine::Layout::NCHW);
		}
	}
}

void Openvino::load_model_to_device(std::string device){
	RCLCPP_INFO(rclcpp::get_logger(node_name_), "Loading model to the device");
	inf_network_ = core_.LoadNetwork(cnn_network_, device);
}

void Openvino::create_async_infer_request(){
	RCLCPP_INFO(rclcpp::get_logger(node_name_), "Create infer request");
	async_infer_request_current_ = inf_network_.CreateInferRequestPtr();
	async_infer_request_next_ = inf_network_.CreateInferRequestPtr();
}

void Openvino::start_next_async_infer_request(){
	async_infer_request_next_->StartAsync();
}

void Openvino::swap_async_infer_request(){
	async_infer_request_current_.swap(async_infer_request_next_);
}

bool Openvino::is_device_ready(){
	return InferenceEngine::OK == async_infer_request_current_->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
}

std::vector<std::string> Openvino::get_labels(){
	return labels_;
}

std::vector<DetectionObject> Openvino::get_detection_objects(size_t height, 
															size_t width, 
															float threshold, 
															float iou_threshold){
	// Processing output blobs of the CURRENT request
	const InferenceEngine::TensorDesc& inputDesc = input_info_.begin()->second.get()->getTensorDesc();
	unsigned long resized_img_h = getTensorHeight(inputDesc);
	unsigned long resized_img_w = getTensorWidth(inputDesc);

	// Parsing outputs
	std::vector<DetectionObject> objects;
	for (auto &output: output_info_){
		auto output_name = output.first;
		InferenceEngine::Blob::Ptr blob = async_infer_request_current_->GetBlob(output_name);

		if (network_type_ == "YOLO"){
			parse_yolov3_output(cnn_network_, 
								output_name, 
								blob, 
								resized_img_h, 
								resized_img_w, 
								height, 
								width, 
								threshold, 
								objects);
		}else if (network_type_ == "SSD"){
			parse_ssd_output(blob, height, width, threshold, objects);
		}
	}

	// Filtering overlapping boxes
	std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
	for (int i = 0; i < objects.size(); ++i){
		if (objects[i].confidence == 0)
			continue;
		for (int j = i + 1; j < objects.size(); ++j) {
			if (intersection_over_union(objects[i], objects[j]) >= iou_threshold) {
				objects[j].confidence = 0;
			}
		}
	}

	return objects;
}

void Openvino::frame_to_next_infer(const cv::Mat &frame, bool auto_resize){
	frame_to_blob(frame, async_infer_request_next_, input_name_, auto_resize);
}

int Openvino::entry_index(int side, int lcoords, int lclasses, int location, int entry) {
	int n = location / (side * side);
	int loc = location % (side * side);
	return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

double Openvino::intersection_over_union(const DetectionObject &box_1, const DetectionObject &box_2) {
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

void Openvino::frame_to_blob(const cv::Mat &frame, 
							InferenceEngine::InferRequest::Ptr &infer_request, 
							const std::string &input_name, bool auto_resize){
	if (auto_resize){
		// Just set input blob containing read image. Resize and layout conversion will be done automatically 
		infer_request->SetBlob(input_name, wrapMat2Blob(frame));
	}else{
		// Resize and copy data from the image to the input blob
		InferenceEngine::Blob::Ptr frame_blob = infer_request->GetBlob(input_name);
		matU8ToBlob<uint8_t>(frame, frame_blob);
	}
}

void Openvino::parse_ssd_output(const InferenceEngine::Blob::Ptr &blob, 
								const unsigned long height, 
								const unsigned long width, 
								const float threshold, 
								std::vector<DetectionObject> &objects){
	// Validating output parameters
	InferenceEngine::SizeVector output_dims = blob->getTensorDesc().getDims();
	int max_proposal_count = static_cast<int>(blob->getTensorDesc().getDims()[2]);
	const int object_size = static_cast<int>(blob->getTensorDesc().getDims()[3]);

	if (object_size != 7){
		throw std::logic_error("Openvino: Output should have 7 as a last dimension");
	}
	if (output_dims.size() != 4){
		throw std::logic_error("Openvino: Incorrect output dimensions for SSD");
	}

	const float *output_blob = blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();
	for (int i = 0; i < max_proposal_count; i++){
		float id = output_blob[i * object_size + 0];

		if (id < 0) break;

		auto label = static_cast<int>(output_blob[i * object_size + 1]);
		float prob = output_blob[i * object_size + 2];
		float xmin = output_blob[i * object_size + 3] * width;
		float ymin = output_blob[i * object_size + 4] * height;
		float xmax = output_blob[i * object_size + 5] * width;
		float ymax = output_blob[i * object_size + 6] * height;

		double new_height = ymax - ymin;
		double new_width = xmax - xmin;

		if (prob < threshold) continue;

		DetectionObject obj(xmin, ymin, new_height, new_width, label, this->labels_[label], prob);
		objects.push_back(obj);
	}
}

void Openvino::parse_yolov3_output(const InferenceEngine::CNNNetwork &cnn_network, 
									const std::string &output_name, 
									const InferenceEngine::Blob::Ptr &blob, 
									const unsigned long resized_img_h, 
									const unsigned long resized_img_w, 
									const unsigned long original_img_h, 
									const unsigned long original_img_w, 
									const float threshold, 
									std::vector<DetectionObject> &objects) {
	// Validating output parameters 
	const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
	const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
	if (out_blob_h != out_blob_w){
		throw std::runtime_error("Openvino: Invalid size of output " + output_name
			+ " It should be in NCHW layout and H should be equal to W. Current H = "
			+ std::to_string(out_blob_h)
			+ ", current W = " + std::to_string(out_blob_w));
	}

	// Extracting layer parameters
	YoloParams params;
	if (auto ngraph_function = cnn_network.getFunction()){
		for (const auto op: ngraph_function->get_ops()){
			if (op->get_friendly_name() == output_name){
				auto region_yolo = std::dynamic_pointer_cast<ngraph::op::RegionYolo>(op);
				if (!region_yolo){
					throw std::runtime_error("Openvino: Invalid output type: "
						+ std::string(region_yolo->get_type_info().name)
						+ ". region_yolo expected");
				}

				params = region_yolo;
				break;
			}
		}
	}else{
		throw std::runtime_error("Openvino: Can't get ngraph::Function."
			"Make sure the provided model is in IR version 10 or greater.");
	}

	auto side = out_blob_h;
	auto side_square = side * side;
	const float *output_blob = blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();

	// Parsing YOLO Region output 
	for (int i = 0; i < side_square; ++i){
		int row = i / side;
		int col = i % side;
		for (int n = 0; n < params.num; ++n){
			int obj_idx = entry_index(side, params.coords, params.classes, n * side * side + i, params.coords);
			int box_idx = entry_index(side, params.coords, params.classes, n * side * side + i, 0);
			float scale = output_blob[obj_idx];

			if (scale < threshold) continue;

			double x = (col + output_blob[box_idx + 0 * side_square]) / side * resized_img_w;
			double y = (row + output_blob[box_idx + 1 * side_square]) / side * resized_img_h;
			double height = std::exp(output_blob[box_idx + 3 * side_square]) * params.anchors[2 * n + 1];
			double width = std::exp(output_blob[box_idx + 2 * side_square]) * params.anchors[2 * n];
			for (int j = 0; j < params.classes; ++j) {
				int class_idx = entry_index(side, params.coords, params.classes, n * side_square + i, params.coords + 1 + j);
				float prob = scale * output_blob[class_idx];

				if (prob < threshold) continue;

				DetectionObject obj(x, y, height, width, j, this->labels_[j], prob,
					static_cast<float>(original_img_h) / static_cast<float>(resized_img_h),
					static_cast<float>(original_img_w) / static_cast<float>(resized_img_w));
				objects.push_back(obj);
			}
		}
	}
}
