import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess


def generate_launch_description():
  
   # 1. System command to lock focus and exposure BEFORE the node starts
   # We set focus to 10 (infinity) and disable auto-focus.
   # We also disable auto-exposure (1=manual) and set a fixed value (e.g. 150)
   # Adjust '/dev/video0' if your camera is on a different port.
#    config_cam_cmd = ExecuteProcess(
#        cmd=[
#            'v4l2-ctl', '-d', '/dev/video0', 
#            # ls -la /dev/video*
#            # v4l2-ctl -d /dev/video0 --set-ctrl=focus_automatic_continuous=0 --set-ctrl=focus_absolute=10 
#            '--set-ctrl=focus_automatic_continuous=0',
#         #    '--set-ctrl=focus_absolute=10',
#            # Optional: Lock exposure to prevent brightness flickering during calibration
#            '--set-ctrl=exposure_auto=3',
#            '--set-ctrl=exposure_time_absolute=250'           
#        ],
#        output='screen'
#    )

#    config_cam_cmd = ExecuteProcess(
#        cmd=[
#            'v4l2-ctl', '-d', '/dev/video0', 
#            '--set-ctrl=focus_automatic_continuous=0',
#            '--set-ctrl=focus_absolute=10',
#            # Optional: Lock exposure to prevent brightness flickering during calibration
#            '--set-ctrl=exposure_auto=3',
#            '--set-ctrl=exposure_time_absolute=250'           
#        ],
#        output='screen'
#    )


   # 2. Launch the usb_cam node at 1080p
   usb_cam_node = Node(
       package='usb_cam',
       executable='usb_cam_node_exe',
       name='usb_cam',
       namespace='usb_cam',
       output='screen',
       remappings=[
           ('image_raw', '/image_raw'),
           ('camera_info', '/camera_info')
       ],
       parameters=[{
           'video_device': '/dev/video0',
           'framerate': 30.0,
           'image_width': 1920,
           'image_height': 1080,
           'pixel_format': 'mjpeg2rgb', # Critical for 1080p bandwidth on USB 2.0
           'camera_name': 'c922_1080p',
           'frame_id': 'camera_link',
           'brightness': 128,
        #    'autofocus': False,
        #    'focus': 10,
           'autoexposure': False,
           'exposure': 250,
           'white_balance_temperature_auto': 4000,
           'io_method': 'mmap',
           'camera_info_url': 'file:///home/cc/ee106a/fa25/class/ee106a-aic/ros_workspaces/final/calibration/ost.yaml' # Uncomment once you have it
       }]
   )


   return LaunchDescription([
    #    config_cam_cmd,
       usb_cam_node
   ])


