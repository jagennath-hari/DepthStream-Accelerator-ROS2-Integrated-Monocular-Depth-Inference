from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('trt_path',           default_value='./zoe.trt',       description='directory of TensorRT engine(trt) file'),
        DeclareLaunchArgument('image_topic',           default_value='/image_rect_color',       description='image topic'),
        DeclareLaunchArgument('gui',           default_value='false',       description='show depth map in a GUI'),
        Node(
            package = 'monocular_depth',
            executable = 'depth_map',
            name = 'depth_map_node',
            parameters=[
                {"trt_path": LaunchConfiguration('trt_path')},
                {"image_topic": LaunchConfiguration('image_topic')},
                {"gui": LaunchConfiguration('gui')}
            ]
        )
    ])
