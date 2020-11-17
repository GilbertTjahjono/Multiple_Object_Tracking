import pyrealsense2 as rs
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
from numpy import random
import cv2
import math
from utils.utils import *
import time

#Keras for neural network models
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.regularizers import l1, l2, l1_l2

#Sklearn for classic machine learning models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import pickle

# Set printoptions
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
# matplotlib.rc('font', **{'size': 11})

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)

class collision_test:
    def __init__(self, max_obj = 2):
        self.collision_dict = {}
        for i in range(max_obj):
            self.collision_dict[i] = None

    def collision_time(self, threshold_x, x0, y0, vx, vy, ax, ay):
        tf = None
        t1 = None
        t2 = None
        #Calculate estimated time of collision
        if (vy**2 - 2*ay*y0) > 0:
            t1 = (-1*vy + math.sqrt(vy**2 - 2*ay*y0))/2
            t2 = (-1*vy - math.sqrt(vy**2 - 2*ay*y0))/2
        #Assign non-negative solution to tf
        if t1 is not None and t2 is not None:
            if t1 >= 0:
                tf = t1
            else:
                tf = t2
        #Passing by or collision?
        if tf is not None:
            x_tf = x0 + vx * tf + 0.5 * ax * (tf ** 2)
            if threshold_x >= x_tf >= (-1 * threshold_x):
                collision = True
            else:
                collision = False
        #vy equals to zero or sqrt < 0
        else:
            collision = False

        return collision, tf

    def warning(self, collision, final_time, id):
        warn = None
        if collision == True and self.collision_dict[id] == False:
            warn = "Collision with object - " + str(id) + "in " + round(final_time, 3) + " s"
            self.collision_dict[id] = collision
        elif collision == True and self.collision_dict[id] == True:
            warn = "time until collision " + round(final_time, 3) + " s"
            self.collision_dict[id] = collision
        elif collision == False and self.collision_dict[id] == True:
            warn = "Collision avoided"
            self.collision_dict[id] = collision
        elif collision == False and self.collision_dict[id] == False:
            warn = "No Collision with object - " + str(id)
            self.collision_dict[id] = collision
        return warn

# Observer
class kalman:
    def __init__(self):
        # Initial sensor variance
        self.sigma_z = 0

        # Dimension, class initialization
        self.kf = KalmanFilter(dim_x=6, dim_z=3)

        # Initial state
        self.kf.x = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])

        # Initial sensor noise covariance matrix
        # YOLOv3-TINY
        # self.R = np.array([[0.00042215, -0.00029808, -0.0039606],
        #                    [-0.00029808, 0.00090076, 0.0073543],
        #                    [-0.0039606, 0.0073543, 0.09775805]])

        # YOLOv4
        self.R = np.array([[0.00053424, -0.00029277, -0.0011946],
                           [-0.00029277, 0.0036107, 0.022894],
                           [-0.0011946, 0.022894, 0.16434]])

        # Initial measurement matrix
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0]])

        # Process Noise Covariance Matrix
        # YOLOv3-TINY - FPA
        # self.Q0 = np.array([[-188.65132426, 106.25828932, 360.90666628, -118.74800123, 21.85128471, -15.85290398],
        #                 [-67.03939271, -281.08276584, -29.11979812, -3.08340105, -331.38586768, 112.75600249],
        #                 [-877.06575002, -49.90727187, 54.53888815, -18.5791291, -47.20301402, 169.8866669],
        #                 [68.67889543, -206.54381897, 240.62310132, 188.56084134, 349.39736114, 80.6798933],
        #                 [-326.89797443, 230.61483448, -97.3211894, -285.39227559, -350.51333345, -9.61601188],
        #                 [-22.95852098, 152.10800321, -186.35774196, -880.01669654, -181.79198985, 62.55071459]])

        # YOLOv3_TINY - GA
        # self.Q0 = np.array([[489.3, 306.7, -613.2, 38.6, -428.2, -333.3],
        #                    [1558., -19.56, -195.6, -909.7, -805.5, -1449.],
        #                    [-578., 149.3, 117.9, -1033., -1223., 325.5],
        #                    [-567.2, 677.6, -821.6, -506.1, 370.7, -804.2],
        #                    [683.2, -433.5, -429.3, 45.88, 530.4, -237.8],
        #                    [-993.2, 174.4, 177.4, -63.31, 962.9, -255.8]])

        # YOLOv4 - FPA
        # self.Q0 = np.array([[291.71422894, 443.21930535, 39.98808662, 94.17458302, -112.23653598, 62.95155309],
        #                    [285.51058816, -549.53343393, 14.08942975, -296.84503489, 241.77512019, 95.22981414],
        #                    [262.44084484, -50.22498091, 472.06670268, 418.10139137, -392.04868554, -30.76019913],
        #                    [-218.6827873, -19.71426144, -159.61428227, 326.41519807, -1127.32549064, -15.73908837],
        #                    [-30.22955979, -35.94772889, -115.99420366, -257.08575841, 73.65728508, 5.8732821],
        #                    [-291.13944199, 77.43590737, 147.71565308, 101.87394321, 102.83911743, -272.4701189]])

        # YOLOv4 - GA
        # self.Q0 = np.array([[-440.6, 422.6, -699.8, -578.3, -64.55, -264.6],
        #                    [550.9, 459.1, 466.8, -357., -485.7, 753.9],
        #                    [-1227., 173.7, 331.5, 47.3, 807.7, 684.7],
        #                    [-254.7, 797.5, 433.3, -452.2, 109.2, 16.46],
        #                    [584.2, -252.5, -694.7, 364.8, -110.3, 96.09],
        #                    [-473.8, 704.5, 630.4, -342.4, 842., -96.43]])

        # MANUAL TUNING
        self.Q0 = np.eye(6)*100

    def predict(self, dt, x_prev, P_prev):
        # Get previous state & covariance
        self.kf.x = x_prev
        self.kf.P = P_prev

        # Update the state transition matrix
        self.kf.F = np.array([[1, 0, 0, dt, 0, 0],
                              [0, 1, 0, 0, dt, 0],
                              [0, 0, 1, 0, 0, dt],
                              [0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1]])

        # Update the process noise covariance matrix
        # print("dt = ", dt)
        self.kf.Q = dt * self.Q0
        # print("Q = ", self.kf.Q)
        # print("FPF = ", self.kf.F.dot(self.kf.P.dot(np.transpose(self.kf.F))))

        # Predict step
        self.kf.predict()
        x_new = np.array(self.kf.x.copy())
        P_new = np.array(self.kf.P.copy())

        # print("Predict P = ", P_new)
        # print("Predict x = ", x_new)

        return x_new, P_new

    def update(self, x_pred, P_pred, z): #z hasil pengukuran
        # Get prediction result (states & state covariance)
        self.kf.x = x_pred
        self.kf.P = P_pred

        # Update sensor variance
        # self.sigma_z = 0.001063 + 0.0007278 * z[2] + 0.003949 * (z[2]**2)
        self.sigma_z = 3e-05 * (z[2] ** 2.9826)

        # Update sensor noise covariance matrix
        self.R[2, 2] = self.sigma_z

        # Update step
        self.kf.update(z)
        x_updated = np.array(self.kf.x.copy())
        P_updated = np.array(self.kf.P.copy())

        # print("Update P = ", P_updated)
        # print("Update x = ", x_updated)

        return x_updated, P_updated

# Oriented FAST and Rotated BRIEF + Brute-Force Matcher
class matching_object:
    def __init__(self, max_obj = 2, edge_thres = 5, scale_fac = 1.04, n_level = 1, fast = 11, scale = 323, n_features = 312, match_model = 'matching_model/neural_network.json'):
        self.model_path = match_model
        self.softmax = False
        try:
            self.weight_path = match_model[:-4] + "h5"
            #load json and create model
            json_file = open(self.model_path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.loaded_model = model_from_json(loaded_model_json)

            # load weights into new model
            self.loaded_model.load_weights(self.weight_path)

            self.softmax = True

        except:
            # Load from file
            with open(self.model_path, 'rb') as file:
                self.loaded_model = pickle.load(file)

        # Initiate ORB Detector
        self.orb = cv2.ORB_create(nfeatures = n_features, edgeThreshold = edge_thres, scaleFactor = scale_fac, nlevels = n_level, patchSize = edge_thres, fastThreshold = fast)

        # Initiate Brute Force Matcher
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Rescale factor
        self.scale = scale

        # States and States Covariance Initialization
        # YOLOv3_Tiny - FPA
        # self.P0 = np.array([[-2.63339417e+01, 5.37110064e+02, -2.27836845e+02, -1.29641026e+02, 1.98971362e+02, -7.91052584e+02],
        #                     [-4.43861051e+01, 3.98193138e+00, 5.80850382e+01, -5.01077443e+02, -2.64485612e+02, 1.36941712e+02],
        #                     [-8.57817504e+01, -5.51628396e+02, 1.22004698e+01, -3.02455207e+02, 2.08189824e+02, -8.86414650e+01],
        #                     [5.22649973e+02, -1.41445176e+02, 1.20855288e+02, -3.35835264e+02, 1.56582015e+01, -5.12228545e+02],
        #                     [-2.53342172e+01, 5.32855274e+01, -1.67492235e+02, -4.39886541e+01, -8.46290507e+01, -2.57161674e+02],
        #                     [3.80585008e+02, -7.74445726e+01, -7.93217472e+02, -1.37162103e+02, -4.20536487e-01, 1.48495054e+02]])

        # YOLOv3_TINY - GA
        # self.P0 = np.array([[-637.2, 26.06, 361.2, 149., 698.5, -899.8],
        #                     [908.6, -746.8, 346.8, -248.3, 992.6, -959.2],
        #                     [-324.6, 126.8, 958., -416.3, -537.6, -843.7],
        #                     [-297.1, -738.3, 3.346, -676.6, 454.4, 689.4],
        #                     [-467.1, -624.7, -448.9, 711.9, 528., 548.],
        #                     [-216.3, -907.1, -300.1, 777.4, 951.7, 1002.]])

        # YOLOv4 - FPA
        # self.P0 = np.array([[622.99464052, 194.75529227, -106.44412042, -15.82356943, 126.56416772, -71.48488417],
        #                     [80.76717425, 199.92230131, 24.68479265, -193.1691668, -414.19101141, -242.80489416],
        #                     [-43.88521731, -551.89906649, 288.91947211, -373.92022623, -745.61685261, -5.88593641],
        #                     [279.91520874, 79.62071026, -110.91760359, 278.34219807, -555.67936901, -242.58231478],
        #                     [123.08236495, -22.68342948, 213.87060908, -269.46768151, 97.26196312, -69.51838078],
        #                     [68.4394968, -64.2448926, -350.34811878, 101.26502252, 322.82234041, 36.33535218]])

        # YOLOv4 - GA
        # self.P0 = np.array([[-214.7, -697.5, -916.2, 283.1, 813.5, -263.8],
        #                     [-430.3, 1004., 593.5, 818.1, 872., -1023.],
        #                     [-395.1, 93.63, 849.2, 374.8, -147.2, -759.3],
        #                     [1034., -450.7, 784.7, -564.8, -272.6, -402.9],
        #                     [969., -201.8, 841.6, 364.3, 521.9, 423.6],
        #                     [1025., -866.2, 998.2, -183.8, -614., -329.9]])

        # MANUAL TUNING
        self.P0 = np.ones((6, 6))*100

        self.x0 = np.array([1.00, 1.00, 1.00, 0.00, 0.00, 0.00])

        # Dictionary Initialization
        self.object_dict = {}
        self.yolo_centroid = {}
        self.label_dict = {}
        self.id_dict = {}
        self.covariance = {}

        for i in range(max_obj):
            self.object_dict[i] = None
            self.id_dict[i] = False
            self.label_dict[i] = None
            self.yolo_centroid[i] = None
            self.covariance[i] = self.P0

        # Another useful variable initialization
        self.vector_array = np.zeros((max_obj, 3000, 3))
        self.kalman_array = np.tile(self.x0, (max_obj, 3000, 1))
        self.count_dict = [0] * max_obj
        self.case = None

        # Initiate observer class
        self.observer = kalman()

        #Save max_obj for further use
        self.max_obj = max_obj

    def reset_id_memory(self):
        for i in range(self.max_obj):
            self.id_dict[i] = False

    def clear_memory(self, frame=100, max_miss=50):
        for i in range(len(self.count_dict)):
            if (frame - self.count_dict[i]) >= max_miss and self.object_dict[i] is not None:
                print("Frame = ", frame, " objek ke- ", i, " = ", self.count_dict[i], " dihapus")
                self.count_dict[i] = 0
                self.object_dict[i] = None
                self.yolo_centroid[i] = None
                self.vector_array[i, :, :] = 0
                self.kalman_array[i, :, :] = np.tile(self.x0, (3000, 1))
                self.label_dict[i] = None
                self.covariance[i] = self.P0
            else:
                continue

    def predict_and_save(self, dt_pred, frame):
        for i in range(len(self.kalman_array)):
            if self.count_dict[i] != 0:
                # print("Predict object - ", i)
                P_prev = self.covariance[i].copy()
                x_prev = self.kalman_array[i, frame - 1, :].copy()
                self.kalman_array[i, frame, :], self.covariance[i] = self.observer.predict(dt_pred, x_prev, P_prev)
            else:
                continue

    def write_missing_objects(self, frame, total_time, save_path, save_txt):
        # Write missing objects
        if save_txt:  # Write to file
            with open(save_path + '.txt', 'a') as file:
                for i in range(len(self.count_dict)):
                    if frame != self.count_dict[i] and self.object_dict[i] is not None:
                        file.write(('%g ' * 8 + '\n') % (
                            total_time, i,
                            self.kalman_array[i, frame, 0], self.kalman_array[i, frame, 1],
                            self.kalman_array[i, frame, 2], self.kalman_array[i, frame, 3],
                            self.kalman_array[i, frame, 4], self.kalman_array[i, frame, 5]))
                    else:
                        continue

    def update_max_id(self, prev_n, new_n):
        max = None
        if new_n > prev_n:
            max = new_n
        elif new_n <= prev_n:
            allow = True
            for i in range(self.max_obj):
                if self.object_dict[i] is not None:
                    max = i+1
                    allow = False
            if allow:
                max = new_n
        max = min(max, self.max_obj)
        return max

    def resize(self, image, scale_percent):
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return image

    def extract_and_match(self, image0, image1):
        # Feature Extraction with ORB
        kp1, des1 = self.orb.detectAndCompute(image0, None)  # extract feature dari image0
        kp2, des2 = self.orb.detectAndCompute(image1, None)  # extract feature dari image1

        if des2 is None or des1 is None:
            print("No descriptor")
            banyak_match = 0
            match_rate = 0

        else:
            # Match Descriptors
            matches = self.bf.match(des1, des2)  # perform BF matcher
            matches = sorted(matches, key=lambda x: x.distance)  # sort based on the distance'

            # Average Distance
            if len(matches) != 0:
                banyak_match = len(matches)
                match_rate = 2 * banyak_match * 100 / (len(kp1) + len(kp2))

            else:
                print("Tidak ada matches")
                banyak_match = 0
                match_rate = 0

        return banyak_match, match_rate

    def CalculateProbability(self, center_old, center_new, old, new):
        # Resize
        old = self.resize(old, self.scale) #Hyperparameter Scale, GA Abel
        new = self.resize(new, self.scale) #Hyperparameter Scale, GA Abel

        # Feature Extraction and Matching
        n_match, match_rate = self.extract_and_match(old, new)

        # Pixel Distance
        pixel_dist = np.sqrt((center_new[0] - center_old[0]) ** 2 + (center_new[1] - center_old[1]) ** 2)
        if np.isnan(pixel_dist):
            print("Pixel dist = ", pixel_dist)
            print("Center_old = ", center_old)
            print("Center_new = ", center_new)

        x_test = ([pixel_dist, n_match, match_rate]) #centroid_dist, n_match, match_rate
        x_test = np.reshape(x_test, (-1, 3))
        probability = self.loaded_model.predict(x_test)

        return probability

    def main(self, object, depth_pixel, frame, label, depth_intrin, jarak, max):
        # Initialize id
        id = None
        start = 0
        stop = 0

        # Create a mask
        mask = [False] * self.max_obj
        for i in range(self.max_obj):
            if self.object_dict[i] is None:
                mask[i] = True

        # Condition #1, if Dictionary is still empty
        if np.all(mask):
            id = 0
            self.object_dict[id] = object
            self.count_dict[id] = frame
            self.label_dict[id] = label
            self.id_dict[id] = True
            self.yolo_centroid[id] = depth_pixel

            # Measurement + deprojection
            self.vector_array[id, frame, :] = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, jarak)

            # Observer update step
            start = time.time()
            self.kalman_array[id, frame, :], self.covariance[id] = self.observer.update(self.kalman_array[id, frame, :], self.covariance[id], self.vector_array[id, frame, :])
            stop = time.time()

        # Condition #2, if one or more lists in object_dict are NOT EMPTY
        else:
            self.case = np.zeros((self.max_obj, 2))
            # self.z_diff = np.zeros((self.max_obj, 1))
            for i in range(self.max_obj):
                if self.object_dict[i] is not None and self.id_dict[i] == False and i < max and self.label_dict[i] == label:
                    probability = self.CalculateProbability(self.yolo_centroid[i], depth_pixel, self.object_dict[i], object)
                    self.case[i, 0] = i
                    # self.z_diff[i, 0] = abs(jarak - self.kalman_array[i, frame - 1, 6])
                    if self.softmax:
                        self.case[i, 1] = probability[0, 1] #softmax
                    else:
                        self.case[i, 1] = probability

                elif self.object_dict[i] is None or self.id_dict[i] == True or i >= max or self.label_dict[i] != label:
                    self.case[i, 0] = i
                    self.case[i, 1] = 0
                    # self.z_diff[i, 0] = abs(jarak - self.kalman_array[i, frame - 1, 6])

            # if len(self.z_diff) > 1:
            #     self.z_diff = self.z_diff/(self.z_diff.max())
            #     self.z_diff = 1 - self.z_diff
            #     self.case[:, 1] = self.case[:, 1] * self.z_diff[:, 0]

            self.case = np.sort(self.case.view('i8, i8'), order=['f1'], axis=0).view(np.float)
            self.case = self.case[::-1]

            temp_id = self.case[0, 0]
            probs = self.case[0, 1]

            if probs < 0.4:
                kosong = False
                for l in range(self.max_obj):
                    if self.object_dict[l] is not None:
                        kosong = False
                    else:
                        kosong = True
                        break

                if kosong:
                    for k in range(self.max_obj):
                        if self.object_dict[k] is None and k <= max and self.id_dict[k] == False:
                            id = int(k)
                            self.object_dict[id] = object
                            self.count_dict[id] = frame
                            self.label_dict[id] = label
                            self.id_dict[id] = True
                            self.yolo_centroid[id] = depth_pixel

                            # Measurement + deprojection
                            self.vector_array[id, frame, :] = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, jarak)

                            # Observer update step
                            start = time.time()
                            self.kalman_array[id, frame, :], self.covariance[id] = self.observer.update(self.kalman_array[id, frame, :], self.covariance[id], self.vector_array[id, frame, :])
                            stop = time.time()
                            break
                        else:
                            continue

                elif kosong == False:
                    # print("Objek baru tidak mendapatkan id (slot penuh)")
                    pass

            else:
                id = int(temp_id)
                self.object_dict[id] = object
                self.count_dict[id] = frame
                self.label_dict[id] = label
                self.id_dict[id] = True
                self.yolo_centroid[id] = depth_pixel

                # Measurement + deprojection
                self.vector_array[id, frame, :] = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, jarak)

                # Observer update step
                start = time.time()
                self.kalman_array[id, frame, :], self.covariance[id] = self.observer.update(self.kalman_array[id, frame, :], self.covariance[id], self.vector_array[id, frame,:])
                stop = time.time()

        update_time = stop-start
        return id, update_time

#Calculating Depth
def calcdepth(xmin, ymin, xmax, ymax, depth, depth_scale):
    # Crop depth data
    depth = depth[ymin:ymax,xmin:xmax].astype(float)
    # Get data scale from the device and convert to meters
    depth = depth * depth_scale
    xc = int((xmax - xmin)/2)
    yc = int((ymax - ymin)/2)
    dist = depth[yc, xc]
    return dist

#Bounding box visualization
def plot_one_box_gilbert(x, img, color=None, dist = None, id = None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(str(id) + " - " + str(id), 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img, c1, c2, color, -1)  # filled
    cv2.putText(img, str(id), (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    cv2.putText(img, str(dist) + " m", (c2[0], c2[1] - 5), 0, tl / 3, [225, 255, 255], thickness=tf,
                lineType=cv2.LINE_AA)
    # if label:
    #     tf = max(tl - 1, 1)  # font thickness
    #     t_size = cv2.getTextSize(str(label)+" - " +str(id), 0, fontScale=tl / 3, thickness=tf)[0]
    #     c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    #     cv2.rectangle(img, c1, c2, color, -1)  # filled
    #     # cv2.putText(img, str(label)+" - "+ str(id), (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    #     cv2.putText(img, str(id), (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    #     cv2.putText(img, str(dist) + " m", (c2[0], c2[1] - 5), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

#text visualization
def put_txt(img, str1 = None, fps = None, warning = None):
    x, y = 50, 50
    offset = 35
    color = (0, 0, 0)
    cv2.putText(img, str1, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(img, fps, (x, y + offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(img, warning, (x, y + 2 * offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)