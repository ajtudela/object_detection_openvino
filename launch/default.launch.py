#!/usr/bin/env python3

'''
    Launches an object detection node with default parameters.
'''
import os

from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from nav2_common.launch import RewrittenYaml

def generate_launch_description():
    # Getting directories and launch-files
    object_detection_dir = get_package_share_directory('object_detection_openvino')
    default_params_file = os.path.join(object_detection_dir, 'params', 'default_params.yaml')
    model_xml_file = os.path.join(object_detection_dir, 'model', 'frozen_inference_graph.xml')
    model_bin_file = os.path.join(object_detection_dir, 'model', 'frozen_inference_graph.bin')
    labels_file = os.path.join(object_detection_dir, 'model', 'labels.txt')

    # Input parameters declaration
    params_file = LaunchConfiguration('params_file')
    network_type = LaunchConfiguration('network_type', default='yolo')

    declare_params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Full path to the ROS2 parameters file with detection configuration'
    )

    declare_network_type_arg = DeclareLaunchArgument(
        'network_type',
        default_value=network_type,
        description='Network type: yolo or ssd'
    )

    # Create our own temporary YAML files that include substitutions
    param_substitutions = {
        'model_xml': model_xml_file,
        'model_bin': model_bin_file,
        'labels': labels_file,
    }

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key='',
        param_rewrites=param_substitutions,
        convert_types=True
    )

    # Prepare the laser segmentation node.
    detection_node = Node(
        package = 'object_detection_openvino',
        namespace = '',
        executable = 'object_detection_openvino',
        name = 'object_detection',
        parameters=[configured_params],
        emulate_tty = True
    )

    return LaunchDescription([
        declare_params_file_arg,
        declare_network_type_arg,
        detection_node
    ])