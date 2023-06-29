# ROS - Python librairies
import rospy
import cv_bridge

# Import useful ROS types
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

# Python librairies
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models
import PIL
import time
import sys
import os

# Importing custom made parameters
from params import robot
from params import learning
from params import dataset
from depth import utils as depth
import utilities.frames as frames
from models import ResNet18Velocity

# Setting some custom made parameters
VISUALIZE = False
RECORD = True
    
# (Constant) Transform matrix from the IMU frame to the camera frame
ALPHA = -0.197  # Camera tilt (approx -11.3 degrees)
ROBOT_TO_CAM = np.array([[0, np.sin(ALPHA), np.cos(ALPHA), 0.084],
                         [-1, 0, 0, 0.060],
                         [0, -np.cos(ALPHA), np.sin(ALPHA), 0.774],
                         [0, 0, 0, 1]])
# Inverse the transform
CAM_TO_ROBOT = frames.inverse_transform_matrix(ROBOT_TO_CAM)
# (Constant) Internal calibration matrix (approx focal length)
K = np.array([[1067, 0, 943],
            [0, 1067, 521],
            [0, 0, 1]])

# Compute the transform matrix between the world and the robot
WORLD_TO_ROBOT = np.eye(4)  # The trajectory is directly generated in the robot frame
# Compute the inverse transform
ROBOT_TO_WORLD = frames.inverse_transform_matrix(WORLD_TO_ROBOT)

X = 10
Y = 15
L_ROBOT = robot.L
RESOLUTION = 0.20
VID_DIR = "/home/gabriel/output.avi"
CROP_WIDTH = 210
CROP_HEIGHT = 70
IMAGE_H, IMAGE_W = 720, 1080
THRESHOLD_INTERSECT = 0.1
THRESHOLD_AREA = 25
NORMALIZE_PARAMS = learning.NORMALIZE_PARAMS
VELOCITY = 0.2

class Projection :
    
    # Initializing some ROS related parameters
    bridge = cv_bridge.CvBridge()
    IMAGE_TOPIC = robot.IMAGE_TOPIC
    ODOM_TOPIC = robot.ODOM_TOPIC
    DEPTH_TOPIC = robot.DEPTH_TOPIC
    imgh = IMAGE_H
    imgw = IMAGE_W
    wait_for_depth = True
    
    # Initializing some PyTorch related parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device, "\n")
    
    # Initializing some parameters for the model
    transform = transforms.Compose([ 
        # transforms.Resize(100),
        transforms.Resize((70, 210)),
        # transforms.Grayscale(),
        # transforms.CenterCrop(100),
        # transforms.RandomCrop(100),
        transforms.ToTensor(),
        # Mean and standard deviation were pre-computed on the training data
        # (on the ImageNet dataset)
        transforms.Normalize(
            mean=NORMALIZE_PARAMS["rbg"]["mean"],
            std=NORMALIZE_PARAMS["rbg"]["std"]
        ),
    ])
    model = ResNet18Velocity.ResNet18Velocity(**learning.NET_PARAMS).to(device=device)
    
    model.load_state_dict(torch.load("/home/gabriel/catkin_ws/src/visual_traversability/Parameters/ResNet18Velocity/2023-06-28-16-39-21.params"))
    model.eval()
    
    midpoints = np.array([[0.43156512],[0.98983318],[1.19973744],[1.35943443],[1.51740755],[1.67225206],[1.80821536],[1.94262708],[2.12798895],[2.6080252]])
    
    #Buffer for the image and grids
    img = np.zeros((imgh, imgw, 3))
    img_depth = np.zeros((imgh, imgw, 1))
    img_normals = np.zeros((imgh, imgw, 3))
    rectangle_list = np.zeros((Y,X,2,2), np.int32)
    grid_list = np.zeros((Y,X,4,2), np.int32)
    initialized = False
    time_wait = 250
    min_cost_global = sys.float_info.max
    max_cost_global = sys.float_info.min

    ###### TO INSERT


    def __init__(self) :
         #INITIALIZATION
        #img = cv2.imread(IMG_DIR, cv2.IMREAD_COLOR)
        #self.rectangle_list, self.grid_list = self.get_lists()
        self.sub_image = rospy.Subscriber(self.IMAGE_TOPIC, Image, self.callback_image, queue_size=1)
        self.sub_depth = rospy.Subscriber(self.DEPTH_TOPIC, Image, self.callback_depth, queue_size=1)
        if RECORD :
            self.writer = cv2.VideoWriter(VID_DIR, cv2.VideoWriter_fourcc(*'XVID'), 24, (1920,1080))

    def get_corners(self, x, y) :
        """
        Function that gives the corners of a cell in the costmap
    
        args :
            x , y = coordinates of the cell with the altitude set to 0
        returns :
            points = a list a 4 points (x,y)
        """
    
        points = np.array([[x, y, 0], [x+1, y, 0], [x+1, y+1, 0], [x, y+1, 0]])
    
        return(points)
    
    def correct_points(self, points_image):
            """Remove the points which are outside the image
    
            Args:
                points_image (ndarray (n, 2)): Points coordinates in the image plan
    
            Returns:
                points_image but the points outside the image are now on the edge of the image
            """
            # Keep only points which are on the image
            result = np.copy(points_image)
            for i in range(len(result)) :
                if result[i, 0] < 0 :
                    result[i, 0] = 0
                if result[i, 0] > self.imgw :
                    result[i, 0] = self.imgw
                if result[i, 1] < 0 :
                    result[i, 1] = 0
                if result[i, 1] > self.imgh :
                    result[i, 1] = self.imgh
            
            return result
    
    def get_lists(self) :
    
        rectangle_list = np.zeros((Y,X,2,2), np.int32)
        grid_list = np.zeros((Y,X,4,2), np.int32)
    
        if X % 2 == 1 :
            offset = X // 2 + 0.5
        else :
            offset = (X // 2)
    
        for x in range(X):
            for y in range(Y):
                points_costmap = self.get_corners(x, y)
                points_robot = points_costmap - np.array([(offset, 0, 0)])
                # Strange computation because the robot frame has the x axis toward the back of the robot
                # and the y axis to ward the left.
                points_robot = points_robot[:, [1,0,2]]
                points_robot = points_robot * np.array([1,-1,1])
    
                # Switching from the costmap coordinates to the world coordinates using the resolution of the costmap.
                points_robot = points_robot * RESOLUTION
    
                # Compute the points coordinates in the camera frame
                points_camera = frames.apply_rigid_motion(points_robot, CAM_TO_ROBOT)
    
                # Compute the points coordinates in the image plan
                points_image = frames.camera_frame_to_image(points_camera, K)
                grid_list[y,x] = points_image
    
                # Get the Area of the cell that is in the image
                intern_points = self.correct_points(points_image)
                intern_area = (intern_points[0,1] - intern_points[2,1]) * ((intern_points[1,0]-intern_points[0,0])+(intern_points[2,0]-intern_points[3,0]))/2
    
                # Get the Area of the cell in total
                area = (points_image[0,1] - points_image[2,1]) * ((points_image[1,0]-points_image[0,0])+(points_image[2,0]-points_image[3,0]))/2
    
                # If the area in squared pixels of the costmap cell is big enough, then relevant data can be extracted
                if intern_area/area >= THRESHOLD_INTERSECT and area >= THRESHOLD_AREA :

                    # Get the rectangle inside the points for the NN
                    #centroid = np.mean(points_image, axis=0)
                    #crop_width = np.int32(np.min([self.imgw,np.max(points_image[:,0])])-np.max([0,np.min(points_image[:,0])]))
                    #crop_height = np.int32(crop_width//3)
                    #tl_x = np.clip(centroid[0] - crop_width//2, 0, self.imgw-crop_width)
                    #tl_y = np.clip(centroid[1]-crop_height//2, 0, self.imgh-crop_height)
                    #rect_tl = np.int32([tl_x, tl_y])
                    #rect_br = rect_tl + [crop_width, crop_height]

                    centroid = np.mean(points_robot, axis=0)
                    point_tl = centroid + [0, 0.5*L_ROBOT, 0]
                    point_br = centroid - [0, 0.5*L_ROBOT, 0]
                    point_tl = frames.apply_rigid_motion(point_tl, CAM_TO_ROBOT)
                    point_br = frames.apply_rigid_motion(point_br, CAM_TO_ROBOT)
                    point_tl = frames.camera_frame_to_image(point_tl, K)
                    point_br = frames.camera_frame_to_image(point_br, K)
                    point_tl = point_tl[0]
                    point_br = point_br[0]
                    crop_width = np.int32(np.min([self.imgw,point_br[0]])-np.max([0,point_tl[0]]))
                    crop_height = np.int32(crop_width//3)
                    
                    centroid = np.mean(points_image, axis=0)
                    tl_x = np.clip(centroid[0] - crop_width//2, 0, self.imgw-crop_width)
                    tl_y = np.clip(centroid[1]-crop_height//2, 0, self.imgh-crop_height)
                    rect_tl = np.int32([tl_x, tl_y])
                    rect_br = rect_tl + [crop_width, crop_height]

                    rectangle_list[y,x] = np.array([rect_tl, rect_br])
    
        return(rectangle_list, grid_list)
    
    def predict_costs(self, img, img_depth, img_normals, rectangle_list, model, transform):
    
        costmap = np.zeros((Y,X))
        min_cost = sys.float_info.max
        max_cost = sys.float_info.min
    
    
        # Turn off gradients computation
        with torch.no_grad():
            for x in range(X):
                for y in range(Y) :
                    # Convert the image from BGR to RGB
                    rectangle = rectangle_list[y,x]
                    crop = img[rectangle[0,1]:rectangle[1,1], rectangle[0,0]:rectangle[1,0]]
                    depth_crop = img_depth[rectangle[0,1]:rectangle[1,1], rectangle[0,0]:rectangle[1,0]]
                    normals_crop = img_normals[rectangle[0,1]:rectangle[1,1], rectangle[0,0]:rectangle[1,0]]
    
                    if not np.array_equal(rectangle, np.zeros(rectangle.shape)) :
                        
                        #cv2.imshow("Result", crop)
                        #cv2.waitKey(250)
                        #cv2.destroyAllWindows()
    
                        crop = cv2.cvtColor(np.uint8(crop), cv2.COLOR_BGR2RGB)
                        # Make a PIL image
                        crop = PIL.Image.fromarray(crop)
                        # Apply transforms to the image
                        crop = transform(crop)
                        # Add a dimension of size one (to create a batch of size one)
                        crop = torch.unsqueeze(crop, dim=0)
                        crop = crop.to(self.device)
    
                        # Computing the cost from the classification problem with the help of midpoints
                        output = model(crop, VELOCITY)
                        softmax = nn.Softmax(dim=1)
                        output = softmax(output)
                        output = output.cpu()[0]
                        probs = output.numpy()
                        cost = np.dot(probs,self.midpoints)[0]
                        costmap[y,x] = cost
                        if cost < min_cost :
                            min_cost = cost
                        elif cost > max_cost :
                            max_cost = cost
        
        return(costmap, min_cost, max_cost)
    
    def visualize(self, img, costmap, rectangle_list, grid_list, max_cost, min_cost) :
        imgviz = img.copy()
        costmapviz = np.zeros((Y,X,3), np.uint8)
    
        for x in range(X):
            for y in range(Y):
                rectangle = rectangle_list[y,x]
                if not np.array_equal(rectangle, np.zeros(rectangle.shape)) :
                    rect_tl, rect_br = rectangle[0], rectangle[1]
                    points_image = grid_list[y,x]
    
                    # Displaying the results for validation on the main image
                    centroid = np.mean(points_image, axis=0)
                    cv2.circle(imgviz, tuple(np.int32(centroid)) , radius=4, color=(255,0,0), thickness=-1)
                    rect_tl = tuple(rect_tl)
                    rect_br = tuple(rect_br)
                    cv2.rectangle(imgviz, rect_tl, rect_br, (255,0,0), 1)
                    points_image_reshape = points_image.reshape((-1,1,2))
                    cv2.polylines(imgviz,np.int32([points_image_reshape]),True,(0,255,255))
    
        for x in range(X) :
            for y in range(Y) :
                if costmap[y,x]!= 0 :
                    value = np.uint8(((costmap[y,x]-min_cost)/(max_cost-min_cost))*255)
                    costmapviz[y,x] = (value, value, value)
        costmapviz = cv2.applyColorMap(src=costmapviz, colormap=cv2.COLORMAP_JET)
        for x in range(X) :
            for y in range(Y) :
                if costmap[y,x]== 0 :
                    costmapviz[y,x] = (0, 0, 0)
        
        cv2.imshow("Result", imgviz)
        cv2.imshow("Costmap", cv2.resize(cv2.flip(costmapviz, 0),(X*20,Y*20)))
        cv2.waitKey(16)   

    def record(self, img, costmap, rectangle_list, grid_list, max_cost, min_cost) :
        imgviz = img.copy()
        costmapviz = np.zeros((Y,X,3), np.uint8)
    
        for x in range(X):
            for y in range(Y):
                rectangle = rectangle_list[y,x]
                if not np.array_equal(rectangle, np.zeros(rectangle.shape)) :
                    rect_tl, rect_br = rectangle[0], rectangle[1]
                    points_image = grid_list[y,x]
    
                    # Displaying the results for validation on the main image
                    centroid = np.mean(points_image, axis=0)
                    cv2.circle(imgviz, tuple(np.int32(centroid)) , radius=4, color=(255,0,0), thickness=-1)
                    rect_tl = tuple(rect_tl)
                    rect_br = tuple(rect_br)
                    cv2.rectangle(imgviz, rect_tl, rect_br, (255,0,0), 1)
                    points_image_reshape = points_image.reshape((-1,1,2))
                    cv2.polylines(imgviz,np.int32([points_image_reshape]),True,(0,255,255))
    
        for x in range(X) :
            for y in range(Y) :
                if costmap[y,x]!= 0 :
                    value = np.uint8(((costmap[y,x]-min_cost)/(max_cost-min_cost))*255)
                    costmapviz[y,x] = (value, value, value)
        costmapviz = cv2.applyColorMap(src=costmapviz, colormap=cv2.COLORMAP_JET)
        for x in range(X) :
            for y in range(Y) :
                if costmap[y,x]== 0 :
                    costmapviz[y,x] = (0, 0, 0)

        imgviz_resized = cv2.resize(imgviz[:,:,:3], (np.int32(self.imgw/2), np.int32(self.imgh/2)))
        costmapviz = cv2.resize(cv2.flip(costmapviz, 0),(np.int32(self.imgw/2), np.int32(self.imgh/2)))
        result = np.vstack((imgviz_resized, costmapviz))
        result_borders = cv2.copyMakeBorder(result, 0, 0, np.int32(self.imgw/4), np.int32(self.imgw/4), cv2.BORDER_CONSTANT, value=(0,0,0))
        self.writer.write(result_borders)

    def callback_image(self, msg) :
        if self.wait_for_depth == False :
            self.img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            if self.initialized == False :
                self.imgh, self.imgw, _ = self.img.shape
                self.rectangle_list, self.grid_list = self.get_lists()
                self.initialized = True

            costmap, min_cost, max_cost = self.predict_costs(self.img, self.img_depth, self.img_normals, self.rectangle_list, self.model, self.transform)
            if self.min_cost_global > min_cost :
                self.min_cost_global = min_cost
            if self.max_cost_global < max_cost :
                self.max_cost_global = max_cost

            if VISUALIZE :
                self.visualize(self.img, costmap, self.rectangle_list, self.grid_list, max_cost, min_cost)

            if RECORD :
                self.record(self.img, costmap, self.rectangle_list, self.grid_list, max_cost, min_cost)

        self.wait_for_depth = True

    def callback_depth(self, msg_depth) :
        if self.wait_for_depth == True :
            self.img_depth = self.bridge.imgmsg_to_cv2(msg_depth, desired_encoding="passthrough")
            depthclass = depth.Depth(self.img_depth, dataset.DEPTH_RANGE)
            depthclass.compute_normal(K = robot.K, bilateral_filter = dataset.BILATERAL_FILTER, gradient_threshold = dataset.GRADIENT_THR)

            self.img_depth = depthclass.get_depth(fill = True, default_depth=dataset.DEPTH_RANGE[0], convert_range=True)

            self.img_normals = depthclass.get_normal(fill = True, default_normal = dataset.DEFAULT_NORMAL, convert_range = True)
            self.img_normals = cv2.cvtColor(self.img_normals, cv2.COLOR_BGR2RGB)
            self.wait_for_depth = False


# Main Program for testing

if __name__ == "__main__" :
    rospy.init_node("ProjectionV2")

    projection = Projection()

    rospy.spin()
    if RECORD :
        projection.writer.release()
    cv2.destroyAllWindows()
