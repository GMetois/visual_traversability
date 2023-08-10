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
from exportedmodels import ResNet18Velocity_Regression_V2

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
INPUT_DIR = "/home/gabriel/PRE/bagfiles/raw_bagfiles/ENSTA_Campus/tom_2023-05-30-13-28-39_1.bag"

# Parameters relative to the video recording
OUTPUT_DIR = "/home/gabriel/output.avi"
VISUALIZE = True
RECORD = False
LIVE = False

# Parameters relative to the model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REGRESSION = True
if REGRESSION == True :
    MODEL = ResNet18Velocity_Regression_V2.ResNet18Velocity_Regression_V2(nb_input_channels=learning.NET_PARAMS["nb_input_channels"],
                                                                    nb_input_features=learning.NET_PARAMS["nb_input_features"],
                                                                    nb_classes=1).to(device=DEVICE)
else :
    MODEL = ResNet18Velocity.ResNet18Velocity(nb_input_channels=learning.NET_PARAMS["nb_input_channels"],
                                              nb_input_features=learning.NET_PARAMS["nb_input_features"],
                                              nb_classes=learning.NET_PARAMS["nb_classes"]).to(device=DEVICE)
WEIGHTS = "/home/gabriel/catkin_ws/src/visual_traversability/Parameters/ResNet18Velocity_Regression_V2/total_filtered_hard.params"

CROP_WIDTH = 210
CROP_HEIGHT = 70
NORMALIZE_PARAMS = learning.NORMALIZE_PARAMS
TRANSFORM = ResNet18Velocity.test_transform
TRANSFORM_DEPTH = ResNet18Velocity.transform_depth
TRANSFORM_NORMAL = ResNet18Velocity.transform_normal

DATASET = "/home/gabriel/PRE/datasets/dataset_multimodal_siamese_png_filtered_hard"
if REGRESSION == False :
    MIDPOINTS = np.load(DATASET + "/bins_midpoints.npy")
else :
    MIDPOINTS = None

VELOCITY = 0.2

# Paremeters relative to the video treatment
THRESHOLD_INTERSECT = 0.1
THRESHOLD_AREA = 25