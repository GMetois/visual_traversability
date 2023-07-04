# ROS - Python librairies
import rospy
import cv_bridge

# Import useful ROS types
from sensor_msgs.msg import Image

# Python librairies
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import PIL
import sys

# Importing custom made parameters
from params import robot
from params import learning
from params import dataset
from depth import utils as depth
import utilities.frames as frames
from exportedmodels import ResNet18Velocity

# Importing parameters relative to the robot size and configuration
ALPHA = robot.alpha
ROBOT_TO_CAM = robot.ROBOT_TO_CAM
CAM_TO_ROBOT = robot.CAM_TO_ROBOT
K = robot.K
WORLD_TO_ROBOT = np.eye(4)
ROBOT_TO_WORLD = frames.inverse_transform_matrix(WORLD_TO_ROBOT)
L_ROBOT = robot.L

# Parameters relative to the costmap configuration
X = 20
Y = 20
RESOLUTION = 0.20

# Parameters relative to the rosbag input
IMAGE_H, IMAGE_W = 720, 1080
IMAGE_TOPIC = robot.IMAGE_TOPIC
IMAGE_RATE = robot.CAMERA_SAMPLE_RATE
ODOM_TOPIC = robot.ODOM_TOPIC
ODOM_RATE = robot.ODOM_SAMPLE_RATE
DEPTH_TOPIC = robot.DEPTH_TOPIC
DEPTH_RATE = robot.DEPTH_SAMPLE_RATE
NB_MESSAGES_THR = dataset.NB_MESSAGES_THR
TIME_DELTA = dataset.TIME_DELTA
INPUT_DIR = "/home/gabriel/PRE/bagfiles/raw_bagfiles/ENSTA_Campus/tom_2023-05-30-13-25-39_0.bag"

# Parameters relative to the video recording
OUTPUT_DIR = "/home/gabriel/output.avi"
VISUALIZE = False
RECORD = True
LIVE = False

# Parameters relative to the model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = ResNet18Velocity.ResNet18Velocity(**learning.NET_PARAMS).to(device=DEVICE)
WEIGHTS = "/home/gabriel/catkin_ws/src/visual_traversability/Parameters/ResNet18Velocity_Augmented/network.params"

CROP_WIDTH = 210
CROP_HEIGHT = 70
NORMALIZE_PARAMS = learning.NORMALIZE_PARAMS
TRANSFORM = ResNet18Velocity.test_transform
TRANSFORM_DEPTH = ResNet18Velocity.transform_depth
TRANSFORM_NORMAL = ResNet18Velocity.transform_normal
MIDPOINTS = np.array([[0.43156512],[0.98983318],[1.19973744],[1.35943443],[1.51740755],[1.67225206],[1.80821536],[1.94262708],[2.12798895],[2.6080252]])
VELOCITY = 0.2

# Paremeters relative to the video treatment
THRESHOLD_INTERSECT = 0.1
THRESHOLD_AREA = 25