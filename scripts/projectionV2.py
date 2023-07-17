# ROS - Python librairies
import rospy
import cv_bridge
import rosbag

# Import useful ROS types
from sensor_msgs.msg import Image

# Python librairies
import numpy as np
import cv2
import torch
import torch.nn as nn
import PIL
import sys
from tqdm import tqdm

# Importing custom made parameters
from depth import utils as depth
from params import dataset
import utilities.frames as frames
import visualparams as viz

class Projection :
    
    # Initializing some ROS related parameters
    bridge = cv_bridge.CvBridge()
    IMAGE_TOPIC = viz.IMAGE_TOPIC
    ODOM_TOPIC = viz.ODOM_TOPIC
    DEPTH_TOPIC = viz.DEPTH_TOPIC
    imgh = viz.IMAGE_H
    imgw = viz.IMAGE_W
    live = False
    bag = None

    # Costmap parameters
    X = viz.X
    Y = viz.Y
    
    # Initializing some parameters for the model
    transform = viz.TRANSFORM
    transform_depth = viz.TRANSFORM_DEPTH
    transform_normal = viz.TRANSFORM_NORMAL
    device = viz.DEVICE
    
    model = viz.MODEL
    model.load_state_dict(torch.load(viz.WEIGHTS))
    model.eval()
    
    midpoints = viz.MIDPOINTS
    velocity = viz.VELOCITY
    
    #Buffer for the image and grids
    img = np.zeros((imgh, imgw, 3))
    img_depth = np.zeros((imgh, imgw, 1))
    img_normals = np.zeros((imgh, imgw, 3))
    
    #Buffer for the grids
    rectangle_list = np.zeros((Y,X,2,2), np.int32)
    grid_list = np.zeros((Y,X,4,2), np.int32)
    
    #Control variables
    record = False
    visualize = True
    initialized = False
    time_wait = 250
    min_cost_global = sys.float_info.max
    max_cost_global = sys.float_info.min
    wait_for_depth = True


    def __init__(self) :
        
        #INITIALIZATION
        if viz.LIVE == True :
            self.sub_image = rospy.Subscriber(self.IMAGE_TOPIC, Image, self.callback_image, queue_size=1)
            self.sub_depth = rospy.Subscriber(self.DEPTH_TOPIC, Image, self.callback_depth, queue_size=1)
        else :
            self.bag = rosbag.Bag(viz.INPUT_DIR)
            if not self.is_bag_healthy() :
                print("Rosbag not healthy, cannot open")
                self.bag = None
        
        if self.record :
            self.writer = cv2.VideoWriter(viz.OUTPUT_DIR, cv2.VideoWriter_fourcc(*'XVID'), 24, (1920,1080))

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
        """
        Setup the list of rectangles and lines that will be later used of inference

        Args:
            Nothing : O_o

        Returns:
            rectangle_list : a list of X*Y coordinates of the rectangle position indicating where to crop
            grid_list : a list of X*Y coordinates to indicate the visual projection of the costmap on the display
        """
    
        rectangle_list = np.zeros((self.Y,self.X,2,2), np.int32)
        grid_list = np.zeros((self.Y,self.X,4,2), np.int32)
    
        if self.X % 2 == 1 :
            offset = self.X // 2 + 0.5
        else :
            offset = (self.X // 2)
    
        for x in range(self.X):
            for y in range(self.Y):
                #Get the list of coordinates of the corners in the costmap frame
                points_costmap = self.get_corners(x, y)
                points_robot = points_costmap - np.array([(offset, 0, 0)])
                
                # Strange computation because the robot frame has the x axis toward the back of the robot
                # and the y axis to ward the left.
                points_robot = points_robot[:, [1,0,2]]
                points_robot = points_robot * np.array([1,-1,1])
    
                # Switching from the costmap coordinates to the world coordinates using the resolution of the costmap.
                points_robot = points_robot * viz.RESOLUTION
    
                # Compute the points coordinates in the camera frame
                points_camera = frames.apply_rigid_motion(points_robot, viz.CAM_TO_ROBOT)
    
                # Compute the points coordinates in the image plan
                points_image = frames.camera_frame_to_image(points_camera, viz.K)
                grid_list[y,x] = points_image
    
                # Get the Area of the cell that is in the image
                intern_points = self.correct_points(points_image)
                intern_area = (intern_points[0,1] - intern_points[2,1]) * ((intern_points[1,0]-intern_points[0,0])+(intern_points[2,0]-intern_points[3,0]))/2
    
                # Get the Area of the cell in total
                area = (points_image[0,1] - points_image[2,1]) * ((points_image[1,0]-points_image[0,0])+(points_image[2,0]-points_image[3,0]))/2
    
                # If the area in squared pixels of the costmap cell is big enough, then relevant data can be extracted
                #
                # IMPORTANT : If there's nothing to extract because the rectangle is too small on the image,
                # KEEP THE COORDINATES TO ZERO, this is the way we're going to check later for the pertinence of the coordinates.
                if intern_area/area >= viz.THRESHOLD_INTERSECT and area >= viz.THRESHOLD_AREA :

                    #We project the footprint of the robot as if it was centered on the cell
                    #We then get the smallest bounding rectangle of the footprint to keep the terrain on wich
                    #it would step over if it was on the costmap's cell
                    
                    #Getting the footprint coordinates
                    centroid = np.mean(points_robot, axis=0)
                    point_tl = centroid + [0, 0.5*viz.L_ROBOT, 0]
                    point_br = centroid - [0, 0.5*viz.L_ROBOT, 0]
                    
                    #Projecting the footprint in the image frame
                    point_tl = frames.apply_rigid_motion(point_tl, viz.CAM_TO_ROBOT)
                    point_br = frames.apply_rigid_motion(point_br, viz.CAM_TO_ROBOT)
                    point_tl = frames.camera_frame_to_image(point_tl, viz.K)
                    point_br = frames.camera_frame_to_image(point_br, viz.K)
                    
                    #Extracting the parameters for the rectangle
                    point_tl = point_tl[0]
                    point_br = point_br[0]
                    crop_width = np.int32(np.min([self.imgw,point_br[0]])-np.max([0,point_tl[0]]))
                    crop_height = np.int32(crop_width//3)
                    
                    #Extracting the rectangle from the centroid of the projection of the costmap's cell
                    centroid = np.mean(points_image, axis=0)
                    tl_x = np.clip(centroid[0] - crop_width//2, 0, self.imgw-crop_width)
                    tl_y = np.clip(centroid[1]-crop_height//2, 0, self.imgh-crop_height)
                    rect_tl = np.int32([tl_x, tl_y])
                    rect_br = rect_tl + [crop_width, crop_height]

                    #Appending the rectangle to the list
                    rectangle_list[y,x] = np.array([rect_tl, rect_br])
    
        return(rectangle_list, grid_list)
    
    def predict_costs(self, img, img_depth, img_normals, rectangle_list, model):
        """
        The main function of this programs, take a list of coordinates and the input image
        Put them in the NN and compute the cost for each crop
        Then reconstituate a costmap of costs

        Args:
            img : RGB input of the robot
            img_depth : depth image of the robot
            img_normals : RGB representation of the normals computed from the depth image
            rectangle_list : list of the rectangle coordinates indicating where to crop according to the costmap's projection on the image
            model : the NN
        
        Returns:
            Costmap : A numpy array of X*Y dimension with the costs
            max_cost, min_cost : the max and min cost of the costmap, useful for visualization later ;)
        """

        #Intializing buffers
        costmap = np.zeros((self.Y,self.X))
        min_cost = sys.float_info.max
        max_cost = sys.float_info.min
    
    
        # Turn off gradients computation
        with torch.no_grad():
            
            #Iteratinf on the rectangles
            for x in range(self.X):
                for y in range(self.Y) :
                    
                    #Getting the rectangle coordinates
                    rectangle = rectangle_list[y,x]

                    #Cropping the images to get the inputs we want for this perticular cell
                    crop = img[rectangle[0,1]:rectangle[1,1], rectangle[0,0]:rectangle[1,0]]
                    depth_crop = img_depth[rectangle[0,1]:rectangle[1,1], rectangle[0,0]:rectangle[1,0]]
                    normals_crop = img_normals[rectangle[0,1]:rectangle[1,1], rectangle[0,0]:rectangle[1,0]]
    
                    #If the rectangle is not empty (Check if we considered beforehand that it was useful to crop there)
                    if not np.array_equal(rectangle, np.zeros(rectangle.shape)) :
    
                        #Converting the BGR image to RGB
                        crop = cv2.cvtColor(np.uint8(crop), cv2.COLOR_BGR2RGB)
                       
                        # Make a PIL image
                        crop = PIL.Image.fromarray(crop)
                        depth_crop = PIL.Image.fromarray(depth_crop)
                        normals_crop = PIL.Image.fromarray(normals_crop)
                        
                        # Apply transforms to the image
                        crop = self.transform(crop)
                        depth_crop = self.transform_depth(depth_crop)
                        normals_crop = self.transform_normal(normals_crop)
                        
                        #Constructing the main image input to the format of the NN
                        multimodal_image = torch.cat((crop, depth_crop, normals_crop)).float()
                        multimodal_image = torch.unsqueeze(multimodal_image, dim=0)
                        multimodal_image = multimodal_image.to(self.device)

                        #Computing the fixated velocity
                        #TODO find a way to take a variable input, or an imput of more than one velocity
                        #to compute more costmaps and avoid the velocity dependance
                        velocity = torch.tensor([self.velocity])
                        velocity = velocity.to(self.device)
                        velocity.unsqueeze_(1)
    
                        # Computing the cost from the classification problem with the help of midpoints
                        output = model(multimodal_image, velocity)
                        softmax = nn.Softmax(dim=1)
                        output = softmax(output)
                        output = output.cpu()[0]
                        probs = output.numpy()
                        cost = np.dot(probs,self.midpoints)[0]
                        
                        #Filling the output array (the numeric costmap)
                        costmap[y,x] = cost
                        if cost < min_cost :
                            min_cost = cost
                        elif cost > max_cost :
                            max_cost = cost
        
        return(costmap, min_cost, max_cost)
    
    def display(self, img, costmap, rectangle_list, grid_list, max_cost, min_cost) :
        """
        A function that displays what's currently computed

        Args :
            img : the base image
            costmap : the numerical costmap
            rectangle_list : the list of the coordinates of the cropping rectangles for the NN input
            gris_list : the list of the coordinates of the projected costmap's cells
            max_cost, min_cost : the max and min cost for the color gradient
        
        Returns :
            Displays a bunch of windows with funny colors on it, but nothing else.
        """

        #Buffers initialization
        imgviz = img.copy()
        costmapviz = np.zeros((self.Y,self.X,3), np.uint8)
    
        #For each costmap element
        for x in range(self.X):
            for y in range(self.Y):

                #Getting the rectangle coordinate
                rectangle = rectangle_list[y,x]

                #Checking if we estimated beforehand that the rectangle might have something interesting to display
                if not np.array_equal(rectangle, np.zeros(rectangle.shape)) :

                    #If there's something we get the coordinates of the cell and the rectangle
                    rect_tl, rect_br = rectangle[0], rectangle[1]
                    points_image = grid_list[y,x]
    
                    #Display the center of the cell
                    centroid = np.mean(points_image, axis=0)
                    cv2.circle(imgviz, tuple(np.int32(centroid)) , radius=4, color=(255,0,0), thickness=-1)
                    
                    # Displaying the rectangle
                    rect_tl = tuple(rect_tl)
                    rect_br = tuple(rect_br)
                    cv2.rectangle(imgviz, rect_tl, rect_br, (255,0,0), 1)

                    #Display the frontiers of the cell
                    points_image_reshape = points_image.reshape((-1,1,2))
                    cv2.polylines(imgviz,np.int32([points_image_reshape]),True,(0,255,255))
    
        #Building cell per cell and array that will become our costmap visualization
        for x in range(self.X) :
            for y in range(self.Y) :
                
                #If the cell is not empty because some cost has been generated
                if costmap[y,x]!= 0 :
                    
                    #Normalizing the content
                    value = np.uint8(((costmap[y,x]-min_cost)/(max_cost-min_cost))*255)
                    costmapviz[y,x] = (value, value, value)

                else :
                    #If nothing we leave the image black
                    costmapviz[y,x] = (0, 0, 0)

        #Applying the color gradient        
        costmapviz = cv2.applyColorMap(src=costmapviz, colormap=cv2.COLORMAP_JET)
        
        #Displaying the results
        cv2.imshow("Result", imgviz)
        cv2.imshow("Costmap", cv2.resize(cv2.flip(costmapviz, 0),(self.X*20,self.Y*20)))
        cv2.waitKey(16)   

    def writeback(self, img, costmap, rectangle_list, grid_list, max_cost, min_cost) :
        """
        A function that write what's being computed frame per frame in a video

        Args :
            img : the base image
            costmap : the numerical costmap
            rectangle_list : the list of the coordinates of the cropping rectangles for the NN input
            gris_list : the list of the coordinates of the projected costmap's cells
            max_cost, min_cost : the max and min cost for the color gradient
        
        Returns :
            Creates a cute video in the OUTPUT_DIR, but nothing more.
        """
    
        #Buffers
        imgviz = img.copy()
        costmapviz = np.zeros((self.Y,self.X,3), np.uint8)
    
        #For each cell of the costmap
        for x in range(self.X):
            for y in range(self.Y):
                
                #Getting the rectangle coordinates
                rectangle = rectangle_list[y,x]

                #If there's something noticable to show
                if not np.array_equal(rectangle, np.zeros(rectangle.shape)) :
                    
                    #Getting the rectangle's and the cell's coordinates
                    rect_tl, rect_br = rectangle[0], rectangle[1]
                    points_image = grid_list[y,x]
    
                    # Displaying the centroid of the cell
                    centroid = np.mean(points_image, axis=0)
                    cv2.circle(imgviz, tuple(np.int32(centroid)) , radius=4, color=(255,0,0), thickness=-1)
                    
                    #Displaying the rectangle
                    rect_tl = tuple(rect_tl)
                    rect_br = tuple(rect_br)
                    cv2.rectangle(imgviz, rect_tl, rect_br, (255,0,0), 1)
                    
                    #Displaying the projection the cell
                    points_image_reshape = points_image.reshape((-1,1,2))
                    cv2.polylines(imgviz,np.int32([points_image_reshape]),True,(0,255,255))
    
        #Building cell per cell and array that will become our costmap visualization
        for x in range(self.X) :
            for y in range(self.Y) :
                
                #If the cell is not empty because some cost has been generated
                if costmap[y,x]!= 0 :
                    
                    #Normalizing the content
                    value = np.uint8(((costmap[y,x]-min_cost)/(max_cost-min_cost))*255)
                    costmapviz[y,x] = (value, value, value)

                else :
                    #If nothing we leave the image black
                    costmapviz[y,x] = (0, 0, 0)
        
        #Applying the color gradient        
        costmapviz = cv2.applyColorMap(src=costmapviz, colormap=cv2.COLORMAP_JET)

        #Resizing the images to fill the video frame correctly
        imgviz_resized = cv2.resize(imgviz[:,:,:3], (np.int32(self.imgw/2), np.int32(self.imgh/2)))
        costmapviz = cv2.resize(cv2.flip(costmapviz, 0),(np.int32(self.imgw/2), np.int32(self.imgh/2)))
        
        #Stacking the results and fill the borders
        result = np.vstack((imgviz_resized, costmapviz))
        result_borders = cv2.copyMakeBorder(result, 0, 0, np.int32(self.imgw/4), np.int32(self.imgw/4), cv2.BORDER_CONSTANT, value=(0,0,0))
        
        #Write the resulting frame in the video
        self.writer.write(result_borders)

    def callback_image(self, msg) :
        """The ROS BRG image callback
        """

        #If we have recieved the depth image
        if self.wait_for_depth == False :
            
            #Converting the image to OpenCV
            self.img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
           
            #If it's the first image we're getting    
            if self.initialized == False :
                #Getting the images characteristics
                self.imgh, self.imgw, _ = self.img.shape
                #Setting up the lists of useful corrdinates
                self.rectangle_list, self.grid_list = self.get_lists()
                self.initialized = True
            
            #Building the costmap
            costmap, min_cost, max_cost = self.predict_costs(self.img, self.img_depth, self.img_normals, self.rectangle_list, self.model)
            
            #Updating the cost history (for diplay purpose)
            if self.min_cost_global > min_cost :
                self.min_cost_global = min_cost
            if self.max_cost_global < max_cost :
                self.max_cost_global = max_cost
            
            #Display if necessary
            if self.visualize :
                self.display(self.img, costmap, self.rectangle_list, self.grid_list, max_cost, min_cost)
            
            #Record if necessary
            if self.record :
                self.writeback(self.img, costmap, self.rectangle_list, self.grid_list, max_cost, min_cost)
        
        self.wait_for_depth = True

    def callback_depth(self, msg_depth) :
        """The ROS Depth image callback
        """
        #If we're waiting for a depth image for the inferance
        if self.wait_for_depth == True :
            
            #Convert the image to OpenCV
            self.img_depth = self.bridge.imgmsg_to_cv2(msg_depth, desired_encoding="passthrough")
            
            self.compute_depth()
            
            #Signal that we're not waiting for a depth image
            self.wait_for_depth = False

    def compute_depth(self) :
        """Process the depth image and compute the normals

        Args :
            None
        
        Returns :
            None but fills the img_normals and the img_depth of the class   
        """

        #Copy for edit permission T_T
        depthcopy = self.img_depth.copy()
        
        #Using a Depth class to compute the normals and fill the holes in the depth image
        depthclass = depth.Depth(depthcopy, dataset.DEPTH_RANGE)
        depthclass.compute_normal(K = viz.K, bilateral_filter = dataset.BILATERAL_FILTER, gradient_threshold = dataset.GRADIENT_THR)
        
        #Setting the attribute to the resulting depth and normals
        self.img_depth = depthclass.get_depth(fill = True, default_depth=dataset.DEPTH_RANGE[0], convert_range=True)
        self.img_normals = depthclass.get_normal(fill = True, default_normal = dataset.DEFAULT_NORMAL, convert_range = True)
        self.img_normals = cv2.cvtColor(self.img_normals, cv2.COLOR_BGR2RGB)

    def is_bag_healthy(self) -> bool:
        """Check if a bag file is healthy

        Args:
            bag (str): Path to the bag file

        Returns:
            bool: True if the bag file is healthy, False otherwise
        """    
        # Get the bag file duration
        duration = self.bag.get_end_time() - self.bag.get_start_time()  # [seconds]

        for topic, frequency in [(viz.IMAGE_TOPIC,
                                  viz.IMAGE_RATE),
                                 (viz.DEPTH_TOPIC,
                                  viz.DEPTH_RATE),
                                 (viz.ODOM_TOPIC,
                                  viz.ODOM_RATE)] :

            # Get the number of messages in the bag file
            nb_messages = self.bag.get_message_count(topic)

            # Check if the number of messages is consistent with the frequency
            if np.abs(nb_messages - frequency*duration)/(frequency*duration) >\
                    viz.NB_MESSAGES_THR:
                return False

        return True

# Main Program for testing

if __name__ == "__main__" :
    rospy.init_node("ProjectionV2")

    projection = Projection()

    print("-------------------- VISUAL_TRAVERSABILITY EVALUATION NODE --------------------")
    print("Device : ", projection.device)
    print("Using a live broadcasted rosbag : ", projection.live)
    print("Visualizing the output: ", projection.visualize)
    print("Recording the output in a video: ", projection.record)
    print("-------------------------------------------------------------------------------")

    # If we work on a recorded rosbag
    if not projection.live :

        for _, msg_image, t_image in tqdm(projection.bag.read_messages(topics=[projection.IMAGE_TOPIC]), total=projection.bag.get_message_count(projection.IMAGE_TOPIC)) :

            projection.img = projection.bridge.imgmsg_to_cv2(msg_image, desired_encoding="passthrough")

            # Keep only the images that can be matched with a depth image
            if list(projection.bag.read_messages(
                topics=[viz.DEPTH_TOPIC],
                start_time=t_image - rospy.Duration(
                    viz.TIME_DELTA),
                end_time=t_image + rospy.Duration(
                    viz.TIME_DELTA))):
                
                # Find the depth image whose timestamp is closest to that
                # of the rgb image
                min_t = viz.TIME_DELTA
                
                # Go through the depth topic
                for _, msg_depth_i, t_depth in projection.bag.read_messages(
                    topics=[projection.DEPTH_TOPIC],
                    start_time=t_image - rospy.Duration(viz.TIME_DELTA),
                    end_time=t_image + rospy.Duration(viz.TIME_DELTA)):
                    
                    # Keep the depth image whose timestamp is closest to
                    # that of the rgb image
                    if np.abs(t_depth.to_sec()-t_image.to_sec()) < min_t:
                        min_t = np.abs(t_depth.to_sec() - t_image.to_sec())
                        projection.img_depth = projection.bridge.imgmsg_to_cv2(msg_depth_i, desired_encoding="passthrough")
            
            # If no depth image is found, skip the current image
            else:
                continue

            projection.compute_depth()

            #If it's the first image we're getting    
            if projection.initialized == False :
                #Getting the images characteristics
                projection.imgh, projection.imgw, _ = projection.img.shape
                #Setting up the lists of useful corrdinates
                projection.rectangle_list, projection.grid_list = projection.get_lists()
                projection.initialized = True
            
            #Building the costmap
            costmap, min_cost, max_cost = projection.predict_costs(projection.img, projection.img_depth, projection.img_normals, projection.rectangle_list, projection.model)
            
            #Updating the cost history (for diplay purpose)
            if projection.min_cost_global > min_cost :
                projection.min_cost_global = min_cost
            if projection.max_cost_global < max_cost :
                projection.max_cost_global = max_cost
            
            #Display if necessary
            if projection.visualize :
                projection.display(projection.img, costmap, projection.rectangle_list, projection.grid_list, max_cost, min_cost)
            
            #Record if necessary
            if projection.record :
                projection.writeback(projection.img, costmap, projection.rectangle_list, projection.grid_list, max_cost, min_cost)
        
        projection.bag.close()
        print("Job done")

    rospy.spin()
    
    if projection.record :
        projection.writer.release()
    cv2.destroyAllWindows()
