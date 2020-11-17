import pyrealsense2 as rs
from filterpy.common import Q_discrete_white_noise
import numpy as np
from numpy import dot, zeros, eye
from numpy import random
import cv2
import math
from utils.utils import *
import time

from copy import deepcopy
from math import log, exp, sqrt
import sys
import scipy.linalg as linalg
from filterpy.stats import logpdf
from filterpy.common import pretty_str, reshape_z

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
class ExtendedKalmanFilter:
    def __init__(self, dim_x = 6, dim_z = 3, dim_u = 0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        # Depth Intrin
        self.cx = 320.483
        self.fx = 382.727
        self.fy = 382.727
        self.cy = 240.989
        self.width = 640
        self.height = 480

        # Initial state (x, y, z, vx, vy, vz, u, v, udot, vdot)
        self.x = zeros((dim_x, 1))  # state

        # Initial sensor noise covariance matrix
        # YOLOv4
        self.R = np.array([[0.871042, 0.0258208, 0.0182336],
                           [0.0258208, 0.196295, 0.01083142],
                           [0.0182336, 0.01083142, 0.112137]])

        # YOLOv3-TINY
        # self.R = np.array([[1.16004, -2.22669e-01, -2.43000e-03],
        #                    [-2.22669e-01, 2.45931, -0.00549840],
        #                    [-2.43000e-03, -0.00549840, 0.0921667]])

        self.P = eye(dim_x)  # uncertainty covariance
        self.B = 0  # control transition matrix
        self.F = np.eye(dim_x)  # state transition matrix

        # process uncertainty
        # YOLOv4-FPA
        # self.Q = np.array([[-443.41079468, -358.90610745, 535.20090123, 231.67216484, 50.53904637, -967.83895533],
        #                    [-62.2377352, -85.85934761, -17.33154452, 1159.06664336, 187.97567269, -524.02191456],
        #                    [193.37563382, -1136.78496253, 45.02825935, 136.7607329, -328.44163203, -262.87582515],
        #                    [240.3523849, 18.92543319, -66.83460693, 99.34325575, -556.13579225, 782.95328197],
        #                    [-322.17528865, 1212.12436511, -201.9143389, -568.16373061, -92.46911073, -59.94544782],
        #                    [898.37391912, -1751.73821745, -193.87691015, -341.19158534, 43.09834664, 354.51139545]])

        # YOLOv4-GA
        # self.Q = np.array([[10.99, -177.5, 633.5, 264.4, 787.6, -135.2],
        #                    [69.14, -149.6, 670.7, 81.81, -83.52, -226.9],
        #                    [-295.9, -340.1, -368.5, -297.2, -226.6, 963.2],
        #                    [1565., -300.9, -180.1, -1158., -431.4, -779.],
        #                    [300.4, 118.6, -493.4, 317.5, 343.8, 2426.],
        #                    [-531.7, -371.5, 134.1, 2745., -1126., 372.8]])

        # YOLOv3-TINY - FPA
        # self.Q = np.array([[393.65907384, -632.01049651, -655.84201708, 65.82501235, 352.9992332, 31.34979353],
        #                    [1408.50112215, 366.47784416, 809.55048538, 24.88796751, -308.25125961, 478.58156594],
        #                    [-113.16559873, -878.91502059, -623.69067302, -707.78449457, 323.85567834, -673.72482586],
        #                    [61.96316554, 60.38983418, -340.98461622, 710.5313091, 464.97829656, 235.08908004],
        #                    [461.09785533, -475.43673466, -279.57745462, -543.50424672, -411.18778121, 322.51126298],
        #                    [-538.7140235, -120.44031535, -132.1593989, -52.53300445, -143.08075452, 771.68903977]])

        # YOLOv3-TINY - GA
        # self.Q = np.array([[-1.52e+05, -3.40e+02, 1.43e+02, -1.21e+03, 1.58e+02, -7.15e+02],
        #                 [-1.27e+01, 5.97e+02, 2.25e+02, 3.62e+02, 3.14e+01, -1.38e+02],
        #                 [-3.97e+01, -8.34e+01, -2.54e+01, -4.58e+02, -9.00e+02, -6.49e+02],
        #                 [-5.54e+02, -8.31e+02, -2.23e+02, 2.86e+02, 3.46e+02, -3.22e+02],
        #                 [2.11e+01, -6.13e+01, 5.50e+01, -5.62e+01, 8.19e+01, -2.77e+02],
        #                 [-3.14e+02, 3.91e+02, 2.09e+01, -3.20e+02, -4.14e+00, 5.51e+02]])

        # MANUAL TUNING
        self.Q = np.eye(6) * 100

        # measurement
        z = np.array([None] * self.dim_z)
        self.z = reshape_z(z, self.dim_z, self.x.ndim)

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros(self.x.shape)  # kalman gain
        self.y = zeros((dim_z, 1)) # residual
        self.S = np.zeros((dim_z, dim_z))  # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        # these will always be a copy of x, P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x, P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def wrap(self, x):
        vx_out = 0
        vy_out = 0
        vz_out = 0
        x_max = x[2] * (self.width - self.cx) / self.fx
        y_max = x[2] * (self.height - self.cy) / self.fy
        z_min = 0.15
        x_out  = min(x[0], x_max)
        y_out = min(x[1], y_max)
        z_out = max(x[2], z_min)
        if x[3] > 0:
            vx_out = min(x[3], 7)
        elif x[3] < 0:
            vx_out = max(x[3], -7)
        if x[4] > 0:
            vy_out = min(x[4], 7)
        elif x[4] < 0:
            vy_out = max(x[4], -7)
        if x[5] > 0:
            vz_out = min(x[5], 7)
        elif x[5] < 0:
            vz_out = max(x[5], -7)

        out = np.array([x_out, y_out, z_out, vx_out, vy_out, vz_out])
        return out

    def predict_x(self, u=0):
        """
        Predicts the next state of X. If you need to
        compute the next state yourself, override this function. You would
        need to do this, for example, if the usual Taylor expansion to
        generate F is not providing accurate results for you.
        """
        self.x = dot(self.F, self.x) + dot(self.B, u)

    def predict(self, dt, x_prev, P_prev, u=0):
        # Get previous state & covariance
        self.x = x_prev.reshape(self.dim_x, 1)
        self.P = P_prev

        # Update the state transition matrix
        self.F = np.array([[1, 0, 0, dt, 0, 0],
                           [0, 1, 0, 0, dt, 0],
                           [0, 0, 1, 0, 0, dt],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])

        self.predict_x(u)
        self.P = dot(self.F, self.P).dot(self.F.T) + dt * self.Q

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

        # print("Predict P = ", self.P_prior)
        # print("Predict x = ", self.x_prior)

        x_predicted = self.x_prior.copy()
        x_predicted = x_predicted.reshape(self.dim_x, 1).ravel()

        # x_predicted = self.wrap(x_predicted)

        return x_predicted, self.P_prior

    def HJacobian(self, x):
        # The Jacobian Measurement Matrix (H_j)
        # z = np.clip(x[2][0], 1e-20, None)
        H_j = np.array([[self.fx/x[2][0], 0, (-self.fx*x[0][0])/(x[2][0]**2), 0, 0, 0],
                        [0, self.fy/x[2][0], (-self.fy*x[1][0])/(x[2][0]**2), 0, 0, 0],
                        [0, 0, 1, 0, 0, 0]])
        return H_j

    def Hx(self, x):
        # The Non-Linear Measurement Model
        # z = np.clip(x[2][0], 1e-20, None)
        H_x = np.array([[self.fx*(x[0][0]/x[2][0]) + self.cx],
                        [self.fy*(x[1][0]/x[2][0]) + self.cy],
                        [x[2][0]]])
        return H_x

    def update(self, z, x_pred, P_pred, R=None, residual=np.subtract):
        # Get prediction result (states & state covariance)
        self.x = x_pred.reshape(self.dim_x, 1)
        self.P = P_pred

        z = z.reshape(self.dim_z, 1)
        z[2, 0] = max(z[2, 0], 0.15)

        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        # Update sensor variance
        # self.sigma_z = 0.001063 + 0.0007278 * z[2] + 0.003949 * (z[2] ** 2)
        self.sigma_z = 3e-05 * (z[2] ** 2.9826)

        # Update sensor noise covariance matrix
        self.R[2, 2] = self.sigma_z

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = eye(self.dim_z) * R

        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        H = self.HJacobian(self.x)

        PHT = dot(self.P, H.T)
        self.S = dot(H, PHT) + R
        self.K = PHT.dot(linalg.inv(self.S))


        hx = self.Hx(self.x)
        self.y = residual(z, hx)
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK' is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
        I_KH = self._I - dot(self.K, H)
        self.P = dot(I_KH, self.P).dot(I_KH.T) + dot(self.K, R).dot(self.K.T)

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # print("Update P = ", self.P_post)
        # print("Update x = ", self.x_post)

        x_updated = self.x_post.copy()
        x_updated = x_updated.reshape(self.dim_x, 1).ravel()

        # x_updated = self.wrap(x_updated)

        return x_updated, self.P_post

    @property
    def log_likelihood(self):
        """
        log-likelihood of the last measurement.
        """

        if self._log_likelihood is None:
            self._log_likelihood = logpdf(x=self.y, cov=self.S)
        return self._log_likelihood

    @property
    def likelihood(self):
        """
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
        """
        if self._likelihood is None:
            self._likelihood = exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self):
        """
        Mahalanobis distance of innovation. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.

        Returns
        -------
        mahalanobis : float
        """
        if self._mahalanobis is None:
            self._mahalanobis = sqrt(float(dot(dot(self.y.T, self.SI), self.y)))
        return self._mahalanobis

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

            print("succesfully load json & h5 files")

        except:
            # Load from file
            with open(self.model_path, 'rb') as file:
                self.loaded_model = pickle.load(file)
                print("succesfully load pickle file")

        # Initiate ORB Detector
        self.orb = cv2.ORB_create(nfeatures = n_features, edgeThreshold = edge_thres, scaleFactor = scale_fac, nlevels = n_level, patchSize = edge_thres, fastThreshold = fast)

        # Initiate Brute Force Matcher
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Rescale factor
        self.scale = scale

        # Initial Process Covariance Matrix P0
        # YOLOv4 - FPA
        # self.P0 = np.array([[861.64304712, 20.59901352, -508.34061519, -1034.95089263, -74.3838339, 1212.96782551],
        #                     [-470.09877182, -196.07727173, -611.24263017, 186.63088359, 1217.72381404, -949.47888841],
        #                     [671.96644915, -862.52429218, 943.62574005, 303.19154873, -808.21946315, 656.34410788],
        #                     [580.40575124, -190.12348733, -433.37251672, 1292.04036073, 613.3311984, 213.18832569],
        #                     [82.1204067, 453.82140685, -513.9027965, 218.88522697, 724.79933643, 2271.93966865],
        #                     [640.77708668, 678.72760852, -1208.02691117, 823.78860058, -1190.23853977, -769.51945018]])

        # YOLOv4 - GA
        # self.P0 = np.array([[-603.6, 623.1, 1159., 743.5, 503.5, -467.6],
        #                     [956., -465.3, -602.7, -282.9, -341.2, 866.3],
        #                     [-1061., -601.8, 284.2, 296.6, -210.1, 1422.],
        #                     [652.6, 496.1, -287.3, -191., -71.48, -357.6],
        #                     [486.3, 445.2, -2164., 132.7, -639.1, -113.9],
        #                     [473.4, 594.5, -1327., 750.7, 2355., 86.4]])

        # YOLOv3-TINY - FPA
        # self.P0 = np.array([[-158.21298575, -6.65156418, 992.58532228, -32.27785515, -105.14130332, -1227.10762614],
        #                     [-422.48666615, -172.33656935, 754.01514619, -616.86010997, -29.61321216, 994.3934797],
        #                     [32.59488777, 238.16654732, -1231.02499999, 2958.12691212, -501.46519461, 96.63332145],
        #                     [-303.43513361, -378.35935966, -1883.66853603, 393.3135997, 804.23458196, 20.25956441],
        #                     [355.88708705, 72.08512174, 287.38213707, 349.71264954, 698.04780862, -847.87830938],
        #                     [815.14368432, -804.81383225, -610.09887518, 136.5300026, -849.80055482, 2235.446481]])

        # YOLOv3-TINY - GA
        # self.P0 = np.array([[-3.38e+05, 9.62e+01, 5.24e+00, -5.56e+02, 4.19e+02, 2.69e+01],
        #                  [1.17e+02, -3.90e+02, 8.48e+01, -1.00e+02, -7.53e+02, -3.35e+02],
        #                  [-3.37e+02, 2.99e+02, -6.16e+02, 1.46e+02, -4.17e+02, 1.17e+03],
        #                  [-4.14e+02, -5.89e+02, -3.92e+02, 4.86e+02, -2.04e+02, -5.54e+02],
        #                  [-4.59e+02, -1.37e+03, -6.88e+02, 5.49e+01, -5.88e+01, 2.70e+02],
        #                  [-3.50e+01, -3.24e+01, -3.77e+02, -1.43e+02, -1.92e+02, 4.25e+02]])

        # MANUAL TUNING
        self.P0 = np.ones((6, 6)) * 100

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
        self.observer = ExtendedKalmanFilter(dim_x = 6, dim_z = 3)

        #Save max_obj for further use
        self.max_obj = max_obj

        # Maximum allowable id
        self.max_id = None

    def reset_id_memory(self):
        for i in range(self.max_obj):
            self.id_dict[i] = False

    def clear_memory(self, frame = 100, max_miss = 50):
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

    def write_missing_objects(self, frame, total_time, save_path, save_txt):
        # Write missing objects
        if save_txt:  # Write to file
            with open(save_path + '.txt', 'a') as file:
                for i in range(len(self.count_dict)):
                    if frame != self.count_dict[i] and self.object_dict[i] is not None:
                        file.write(('%g ' * 9 + '\n') % (
                            total_time, i,
                            self.kalman_array[i, frame, 0], self.kalman_array[i, frame, 1],
                            self.kalman_array[i, frame, 2], self.kalman_array[i, frame, 3],
                            self.kalman_array[i, frame, 4], self.kalman_array[i, frame, 5],
                            frame))
                    else:
                        continue

    def predict_and_save(self, dt_pred, frame):
        for i in range(len(self.kalman_array)):
            if self.object_dict[i] is not None:
                # Predict Trajectory
                self.kalman_array[i, frame, :], self.covariance[i] = self.observer.predict(dt_pred, self.kalman_array[i, frame - 1, :], self.covariance[i])
            else:
                continue

    def update_max_id(self, prev_n, new_n):
        if new_n > prev_n:
            self.max_id = new_n
        elif new_n <= prev_n:
            allow = True
            for i in range(self.max_obj):
                if self.object_dict[i] is not None:
                    self.max_id = i+1
                    allow = False
            if allow:
                self.max_id = new_n
        self.max_id = min(self.max_id, self.max_obj)
        # print("prev - new " + str(prev_n) + " " + str(new_n))
        # print("max id = ", self.max_id)

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

    def main(self, object, depth_pixel, frame, label, jarak):
        # Initialize id
        id = None
        start = 0
        stop = 0
        # # Get Intrinsics
        # cx = depth_intrin.ppx
        # cy = depth_intrin.ppy
        # fx = depth_intrin.fx
        # fy = depth_intrin.fy

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
            self.vector_array[id, frame, :] = [depth_pixel[0], depth_pixel[1], jarak]

            # Observer update step
            start = time.time()
            self.kalman_array[id, frame, :], self.covariance[id] = self.observer.update(self.vector_array[id, frame, :],
                                                                                        self.kalman_array[id, frame, :], self.covariance[id])
            stop = time.time()

        # Condition #2, if one or more lists in object_dict are NOT EMPTY
        else:
            self.case = np.zeros((self.max_obj, 2))
            # self.z_diff = np.zeros((self.max_obj, 1))
            for i in range(self.max_obj):
                if self.object_dict[i] is not None and self.id_dict[i] == False and i < self.max_id and self.label_dict[i] == label:
                    probability = self.CalculateProbability(self.yolo_centroid[i], depth_pixel, self.object_dict[i], object)

                    self.case[i, 0] = i

                    # self.z_diff[i, 0] = abs(jarak - self.kalman_array[i, frame - 1, 6])

                    if self.softmax:
                        self.case[i, 1] = probability[0, 1] #softmax
                    else:
                        self.case[i, 1] = probability

                elif self.object_dict[i] is None or self.id_dict[i] == True or i >= self.max_id or self.label_dict[i] != label:
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

            if probs < 0.4:
                kosong = False
                for l in range(self.max_obj):
                    if self.object_dict[l] is not None:
                        kosong = False
                    else:
                        kosong = True
                        break

                if kosong == True:
                    for k in range(self.max_id):
                        if self.object_dict[k] is None and k < self.max_id and self.id_dict[k] == False:
                            id = int(k)
                            self.object_dict[id] = object
                            self.count_dict[id] = frame
                            self.label_dict[id] = label
                            self.id_dict[id] = True
                            self.yolo_centroid[id] = depth_pixel

                            # Measurement + deprojection
                            self.vector_array[id, frame, :] = [depth_pixel[0], depth_pixel[1], jarak]

                            # Observer update step
                            start = time.time()
                            self.kalman_array[id, frame, :], self.covariance[id] = self.observer.update(self.vector_array[id, frame, :], self.kalman_array[id, frame, :], self.covariance[id])
                            stop = time.time()
                            print("objek berhasil disimpan di - ", id)
                            break
                        else:
                            # print("tidak memenuhi kondisi if")
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
                self.vector_array[id, frame, :] = [depth_pixel[0], depth_pixel[1], jarak]

                # Observer update step
                start = time.time()
                self.kalman_array[id, frame, :], self.covariance[id] = self.observer.update(
                    self.vector_array[id, frame, :], self.kalman_array[id, frame, :], self.covariance[id])
                stop = time.time()

        return id, (stop-start)

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