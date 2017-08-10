import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

from vehicles.utils import get_hog_features, bin_spatial, color_hist
from vehicles.dataset import CarDataset
from vehicles.params import Params
import numpy as np
import matplotlib.image as mpimg
import pickle

class CarTrainer:

    def __init__(self):
        self.params = Params()
        self.dataset = CarDataset()

    def train(self):
        car_features = self.extract_features(self.dataset.cars)
        notcar_features = self.extract_features(self.dataset.notcars)

        print('Car features: ', len(car_features))
        print('Not-car features: ', len(notcar_features))
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        self.X_scaler = StandardScaler().fit(X)

        scaled_X = self.X_scaler.transform(X)
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)

        self.svc = LinearSVC(loss='hinge')
        self.svc.fit(X_train, y_train)

        print('Feature vector length:', len(X_train[0]))
        print('Test accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))

    def extract_features(self, images):
        features = []
        for file in tqdm(images):
            image = mpimg.imread(file)
            if self.params.color_space != 'RGB':
                if self.params.color_space == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif self.params.color_space == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif self.params.color_space == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif self.params.color_space == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif self.params.color_space == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else:
                feature_image = np.copy(image)

            for augmented_image in [feature_image]: # , cv2.flip(feature_image, 1)
                file_features = []
                if self.params.spatial_feat:
                    spatial_features = bin_spatial(augmented_image, spatial_size=self.params.spatial_size)
                    file_features.append(spatial_features)
                if self.params.hist_feat:
                    hist_features = color_hist(augmented_image, nbins=self.params.hist_bins)
                    file_features.append(hist_features)
                if self.params.hog_feat:
                    if self.params.hog_channel == 'ALL':
                        hog_features = []
                        for channel in range(augmented_image.shape[2]):
                            hog_features.append(get_hog_features(augmented_image[:, :, channel], self.params))
                        hog_features = np.ravel(hog_features)
                    else:
                        hog_features = get_hog_features(augmented_image[:, :, self.params.hog_channel], self.params)
                    file_features.append(hog_features)

                features.append(np.concatenate(file_features))

        return features

    def dump_training_data(self):

        data = {
            'svc': self.svc,
            'scaler': self.X_scaler,
            'params': self.params
        }

        with open('data.pickle', 'wb') as f:
            pickle.dump(data, f)

