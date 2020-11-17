import pyrealsense2 as rs
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
from numpy import random
import cv2
import math
from utils.utils import *

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
        self.sigma_x, self.sigma_y, self.sigma_z = .05, .05, .05

        # Dimension, class initialization
        self.kf = KalmanFilter(dim_x=9, dim_z=3)

        # Initial state
        self.kf.x = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])

        # Initial sensor noise covariance matrix
        self.kf.R = np.diag([self.sigma_x ** 2, self.sigma_y ** 2, self.sigma_z ** 2])

        # Initial measurement matrix
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0]])

    def predict(self, dt, x_prev, P_prev):
        # Get previous state & covariance
        self.kf.x = x_prev
        self.kf.P = P_prev

        # Update the state transition matrix
        self.kf.F = np.array([[1, dt, dt * dt / 2, 0, 0, 0, 0, 0, 0],
                      [0, 1, dt, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, dt, dt * dt / 2, 0, 0, 0],
                      [0, 0, 0, 0, 1, dt, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, dt, dt * dt / 2],
                      [0, 0, 0, 0, 0, 0, 0, 1, dt],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1]])

        # Update the process noise covariance matrix
        self.kf.Q[0:3, 0:3] = Q_discrete_white_noise(3, dt=dt, var=0.02)
        self.kf.Q[3:6, 3:6] = Q_discrete_white_noise(3, dt=dt, var=0.02)
        self.kf.Q[6:9, 6:9] = Q_discrete_white_noise(3, dt=dt, var=0.02)

        # Predict step
        self.kf.predict()
        x_new = np.array(self.kf.x.copy())
        P_new = np.array(self.kf.P.copy())

        return x_new, P_new

    def update(self, x_pred, P_pred, z): #z hasil pengukuran
        # Get prediction result (states & state covariance)
        self.kf.x = x_pred
        self.kf.P = P_pred

        # Update sensor variance
        self.sigma_z = 0.001063 + 0.0007278 * z[2] + 0.003949 * (z[2]**2)
        self.sigma_x = self.sigma_z
        self.sigma_y = self.sigma_z

        # Update sensor noise covariance matrix
        self.kf.R = np.diag([self.sigma_x ** 2, self.sigma_y ** 2, self.sigma_z ** 2])

        # Update step
        self.kf.update(z)
        x_updated = np.array(self.kf.x.copy())
        P_updated = np.array(self.kf.P.copy())

        return x_updated, P_updated

class observer_centroid:
        def __init__(self):

            # Dimension, class initialization
            self.kalman = KalmanFilter(dim_x = 4, dim_z = 2)

            # Initial state
            self.kalman.x = np.array([0., 0., 0., 0.])

            # Sensor noise covariance matrix
            self.kalman.R = np.diag([0.1, 0.1])

            # Process noise covariance matrix
            self.kalman.Q = np.diag([1, 0.1, 0.5, 0.5])

            # Initial measurement matrix
            self.kalman.H = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0]])

        def predict_center(self, dt, x_prev, P_prev):
            # Get previous state & covariance
            self.kalman.x = x_prev
            self.kalman.P = P_prev

            # Update the state transition matrix
            self.kalman.F = np.array([[1, 0, dt, 0],
                                  [0, 1, 0, dt],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

            # Predict step
            self.kalman.predict()
            self.x_new = np.array(self.kalman.x.copy())
            self.P_new = np.array(self.kalman.P.copy())

            return self.x_new, self.P_new

        def update_center(self, z, x_pred, P_pred):  # z hasil pengukuran
            # Get prediction result (states & state covariance)
            self.kf.x = x_pred
            self.kf.P = P_pred

            # Update step
            self.kf.update(z)
            self.x_updated = np.array(self.kf.x.copy())
            self.P_updated = np.array(self.kf.P.copy())

            return self.x_updated, self.P_updated

# Oriented FAST and Rotated BRIEF + Brute-Force Matcher
class matching_object:
    def __init__(self, max_obj = 2, edge_thres = 5, scale_fac = 1.1, n_level = 1, patch = 5, scale = 324, match_model = 'NN_GA'):
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
        self.orb = cv2.ORB_create(edgeThreshold = edge_thres, scaleFactor = scale_fac, nlevels = n_level, patchSize = patch, fastThreshold = edge_thres)

        # Initiate Brute Force Matcher
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Rescale factor
        self.scale = scale

        # Dictionary Initialization
        self.object_dict = {}
        self.wh_dict = {}
        self.yolo_centroid = {}
        self.label_dict = {}
        self.id_dict = {}
        self.covariance = {}
        self.covariance_ctr = {}

        for i in range(max_obj):
            self.object_dict[i] = None
            self.wh_dict[i] = None
            self.id_dict[i] = False
            self.label_dict[i] = None
            self.yolo_centroid[i] = None
            self.covariance[i] = np.zeros((9, 9))
            self.covariance_ctr[i] = np.zeros((4, 4))

        # Another useful variable initialization
        self.vector_array = np.zeros((max_obj, 3000, 3))
        self.kalman_array = np.zeros((max_obj, 3000, 9))
        self.centroid_array = np.zeros((max_obj, 3000, 4))
        self.count_dict = [0] * max_obj
        self.case = None

        # Initiate observer class
        self.observer = kalman()
        self.centroid_obs = observer_centroid()

        #Save max_obj for further use
        self.max_obj = max_obj

    def reset_id_memory(self):
        for i in range(self.max_obj):
            self.id_dict[i] = False

    def clear_memory(self, frame = 100, max_miss = 50):
        for i in range(len(self.count_dict)):
            if (frame - self.count_dict[i]) >= max_miss and self.object_dict[i] is not None:
                print("Frame = ", frame, " objek ke- ", i, " = ", self.count_dict[i], " dihapus")
                self.count_dict[i] = 0
                self.object_dict[i] = None
                self.wh_dict[i] = None
                self.yolo_centroid[i] = None
                self.vector_array[i, :, :] = 0
                self.kalman_array[i, :, :] = 0
                self.centroid_array[i, :, :] = 0
                self.label_dict[i] = None
                self.covariance[i] = np.zeros((9, 9))
                self.covariance_ctr[i] = np.zeros((4, 4))
            else:
                continue

    def plot_missing_objects(self, frame, im0, colors):
        radius = 3
        thickness = 2
        for i in range(len(self.count_dict)):
            if frame != self.count_dict[i] and self.object_dict[i] is not None:
                color = colors[int(i)]
                self.centroid = self.centroid_array[i, frame, :2].copy()
                self.xc = int(round(self.centroid[0], 0))
                self.yc = int(round(self.centroid[1], 0))
                self.centroid = tuple((self.xc, self.yc))
                cv2.circle(im0, self.centroid, radius, color, thickness)
                print("Object - " + str(i) + " is missing")
            else:
                continue

    def write_missing_objects(self, frame, save_path, save_txt):
        # Write missing objects
        if save_txt:  # Write to file
            with open(save_path + '.txt', 'a') as file:
                for i in range(len(self.count_dict)):
                    if frame != self.count_dict[i] and self.object_dict[i] is not None:
                        file.write(('%g ' * 11 + '\n') % (
                            total_time, i,
                            self.kalman_array[i, frame, 0], self.kalman_array[i, frame, 3],
                            self.kalman_array[i, frame, 6], self.kalman_array[i, frame, 1],
                            self.kalman_array[i, frame, 4], self.kalman_array[i, frame, 7],
                            self.kalman_array[i, frame, 2], self.kalman_array[i, frame, 5],
                            self.kalman_array[i, frame, 8]))
                    else:
                        continue

    def predict_and_save(self, dt_pred, frame):
        for i in range(len(self.kalman_array)):
            if self.object_dict[i] is not None:
                # Trajectory
                P_prev = self.covariance[i].copy()
                x_prev = self.kalman_array[i, frame - 1, :].copy()
                self.kalman_array[i, frame, :], self.covariance[i] = self.observer.predict(dt_pred, x_prev, P_prev)

                #Pixel Centroid
                P_ctr_prev = self.covariance_ctr[i].copy()
                ctr_prev = self.centroid_array[i, frame - 1, :].copy()
                self.centroid_array[i, frame, :], self.covariance_ctr[i] = self.centroid_obs.predict_center(dt_pred, ctr_prev, P_ctr_prev)
                # print("Centroid predict " + str(frame) + " = ", self.centroid_array[i, frame, :2])
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

    def main(self, object, depth_pixel, frame, label, depth_intrin, jarak, max, wh):
        # Initialize id
        id = None

        # Create a mask
        mask = [False] * self.max_obj
        for i in range(self.max_obj):
            if self.object_dict[i] is None:
                mask[i] = True

        # Condition #1, if Dictionary is still empty
        if np.all(mask):
            # print("ID = 0 karena seluruh slot masih kosong")
            id = 0
            self.object_dict[id] = object
            self.count_dict[id] = frame
            self.label_dict[id] = label
            self.id_dict[id] = True
            self.yolo_centroid[id] = depth_pixel

            # Measurement + deprojection
            self.vector_array[id, frame, :] = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, jarak)

            # Observer update step
            P_pred = self.covariance[id].copy()
            x_pred = self.kalman_array[id, frame, :].copy()
            self.kalman_array[id, frame, :], self.covariance[id] = self.observer.update(x_pred, P_pred, self.vector_array[id, frame, :])

            if self.wh_dict[id] is None:
                P_ctr_pred = self.covariance_ctr[id].copy()
                ctr_pred = self.centroid_array[id, frame, :].copy()
                self.centroid_array[id, frame, :], self.covariance_ctr[id] = self.centroid_obs.update_center(depth_pixel, ctr_pred, P_ctr_pred)

            else:
                self.dw = abs(self.wh_dict[id][0] - wh[0])/self.wh_dict[id][0]
                self.dh = abs(self.wh_dict[id][1] - wh[1])/self.wh_dict[id][1]
                if self.dw < 0.3 and self.dh < 0.3:
                    P_ctr_pred = self.covariance_ctr[id].copy()
                    ctr_pred = self.centroid_array[id, frame, :].copy()
                    self.centroid_array[id, frame, :], self.covariance_ctr[id] = self.centroid_obs.update_center(depth_pixel, ctr_pred, P_ctr_pred)
                else:
                    print("Update rules are not satisfied (dw, dh)", self.dw, self.dh)
                    pass

            # Update wh_dict
            self.wh_dict[id] = wh

        # Condition #2, if one or more lists in object_dict are NOT EMPTY
        else:
            self.case = np.zeros((self.max_obj, 2))
            # self.z_diff = np.zeros((self.max_obj, 1))
            for i in range(self.max_obj):
                if self.object_dict[i] is not None and self.id_dict[i] == False and i < max and self.label_dict[i] == label:
                    if frame > self.count_dict[i] + 1 and self.count_dict[i] != 0:
                        probability = self.CalculateProbability(self.centroid_array[i, frame, :2], depth_pixel, self.object_dict[i], object)
                    else:
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
            #     # print("Before Normalization ", self.z_diff)
            #
            #     self.z_diff = self.z_diff/(self.z_diff.max())
            #     self.z_diff = 1 - self.z_diff
            #
            #     # print("After Normalization ", self.z_diff)
            #
            #     # print("Before multiplied ", self.case[:, 1])
            #     self.case[:, 1] = self.case[:, 1] * self.z_diff[:, 0]
            #     # print("After multiplied ", self.case[:, 1])

            self.case = np.sort(self.case.view('i8, i8'), order=['f1'], axis=0).view(np.float)
            self.case = self.case[::-1]

            temp_id = self.case[0, 0]
            probs = self.case[0, 1]

            # print("Probabilitas = " + str(probs) + " untuk id = " + str(temp_id))

            if probs < 0.5:
                kosong = False
                for l in range(self.max_obj):
                    if self.object_dict[l] is not None:
                        kosong = False
                    else:
                        kosong = True

                if kosong:
                    for k in range(self.max_obj):
                        if self.object_dict[k] is None and k < max and self.id_dict[k] == False:
                            id = int(k)
                            self.object_dict[id] = object
                            self.count_dict[id] = frame
                            self.label_dict[id] = label
                            self.id_dict[id] = True
                            self.yolo_centroid[id] = depth_pixel

                            # Measurement + deprojection
                            self.vector_array[id, frame, :] = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, jarak)

                            # Observer update step
                            P_pred = self.covariance[id].copy()
                            x_pred = self.kalman_array[id, frame, :].copy()
                            self.kalman_array[id, frame, :], self.covariance[id] = self.observer.update(x_pred, P_pred,
                                                                                                        self.vector_array[
                                                                                                        id, frame, :])

                            if self.wh_dict[id] is None:
                                P_ctr_pred = self.covariance_ctr[id].copy()
                                ctr_pred = self.centroid_array[id, frame, :].copy()
                                self.centroid_array[id, frame, :], self.covariance_ctr[id] = self.centroid_obs.update_center(depth_pixel, ctr_pred, P_ctr_pred)

                            else:
                                self.dw = abs(self.wh_dict[id][0] - wh[0]) / self.wh_dict[id][0]
                                self.dh = abs(self.wh_dict[id][1] - wh[1]) / self.wh_dict[id][1]
                                if self.dw < 0.3 and self.dh < 0.3:
                                    P_ctr_pred = self.covariance_ctr[id].copy()
                                    ctr_pred = self.centroid_array[id, frame, :].copy()
                                    self.centroid_array[id, frame, :], self.covariance_ctr[id] = self.centroid_obs.update_center(depth_pixel, ctr_pred, P_ctr_pred)
                                else:
                                    print("Update rules are not satisfied (dw, dh)", self.dw, self.dh)
                                    pass

                            # Update wh_dict
                            self.wh_dict[id] = wh

                            # print("ID = " + str(id) + " Objek baru ditambahkan ke slot kosong")
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
                P_pred = self.covariance[id].copy()
                x_pred = self.kalman_array[id, frame, :].copy()
                self.kalman_array[id, frame, :], self.covariance[id] = self.observer.update(x_pred, P_pred,
                                                                                            self.vector_array[id, frame,
                                                                                            :])

                if self.wh_dict[id] is None:
                    P_ctr_pred = self.covariance_ctr[id].copy()
                    ctr_pred = self.centroid_array[id, frame, :].copy()
                    self.centroid_array[id, frame, :], self.covariance_ctr[id] = self.centroid_obs.update_center(depth_pixel, ctr_pred, P_ctr_pred)
                else:
                    self.dw = abs(self.wh_dict[id][0] - wh[0]) / self.wh_dict[id][0]
                    self.dh = abs(self.wh_dict[id][1] - wh[1]) / self.wh_dict[id][1]
                    if self.dw < 0.3 and self.dh < 0.3:
                        P_ctr_pred = self.covariance_ctr[id].copy()
                        ctr_pred = self.centroid_array[id, frame, :].copy()
                        self.centroid_array[id, frame, :], self.covariance_ctr[id] = self.centroid_obs.update_center(depth_pixel, ctr_pred, P_ctr_pred)
                    else:
                        print("Update rules are not satisfied (dw, dh)", self.dw, self.dh)
                        pass

                # Update wh_dict
                self.wh_dict[id] = wh
                # print("ID = " + str(id) + " berdasarkan hasil matching")

        return id

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