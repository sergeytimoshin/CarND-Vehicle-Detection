from scipy.ndimage import label
from vehicles.utils import get_hog_features, bin_spatial, color_hist
import cv2
import numpy as np
import pickle
from collections import deque

class CarFinder:
    def __init__(self):

        with open('data.pickle', 'rb') as f:
            data = pickle.load(f)

        self.scaler = data['scaler']
        self.svc = data['svc']
        self.params = data['params']
        self.params.feature_vec = False
        self.alpha = 0.5
        self.boxes_len_threshold = 30
        self.tracks = deque(maxlen=3)

    # Define a function that takes an image,
    # start and stop positions in both x and y,
    # window size (x and y dimensions),
    # and overlap fraction (for both x and y)
    def slide_window(self, image, x_start_stop=[None, None], y_start_stop=[None, None],
                     xy_window=(128, 128),  # (64, 64), (96, 96)
                     xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = image.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = image.shape[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
        ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
        nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
        ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    # Define a function to draw bounding boxes
    def draw_boxes(self, image, bboxes, color=(255, 0, 0), thick=6):
        # Make a copy of the image
        imcopy = np.copy(image)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy


    def single_img_features(self, image):
        #1) Define an empty list to receive features
        img_features = []
        #2) Apply color conversion if other than 'RGB'
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
        else:
            feature_image = np.copy(image)
        #3) Compute spatial features if flag is set
        if self.params.spatial_feat:
            # Apply bin_spatial() to get spatial color features
            spatial_features = bin_spatial(feature_image, spatial_size=self.params.spatial_size)
            #4) Append features to list
            img_features.append(spatial_features)
        #5) Compute histogram features if flag is set
        if self.params.hist_feat:
            # Apply color_hist() also with a color space option now
            hist_features = color_hist(feature_image, nbins=self.params.hist_bins)
            #6) Append features to list
            img_features.append(hist_features)
        #7) Compute HOG features if flag is set
        if self.params.hog_feat:
            if self.params.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], self.params))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,self.params.hog_channel], self.params)
            #8) Append features to list
            img_features.append(hog_features)
        #9) Return concatenated array of features
        return np.concatenate(img_features)


    def search_windows(self, image, windows):
        on_windows = []
        for window in windows:
            test_img = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            features = self.single_img_features(test_img)
            test_features = self.scaler.transform(np.array(features).reshape(1, -1))
            prediction = self.svc.predict(test_features)
            if prediction == 1:
                on_windows.append(window)
        return on_windows

    def add_heat(self, heatmap, bbox_list):
        for box in bbox_list:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        return heatmap

    def apply_threshold(self, heatmap, threshold):
        heatmap[heatmap <= threshold] = 0
        return heatmap

    def draw_labeled_bboxes(self, image, labels):
        for car_number in range(1, labels[1] + 1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            cv2.rectangle(image, bbox[0], bbox[1], (255, 0, 0), 8)
        # Return the image
        return image

    def convert_color(self, image, conv='RGB2YCrCb'):
        if conv == 'RGB2YCrCb':
            return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        if conv == 'BGR2YCrCb':
            return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        if conv == 'RGB2LUV':
            return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)

    def find_cars(self, image, ystart, ystop, scale):

        draw_img = np.copy(image)
        image = image.astype(np.float32) / 255

        img_tosearch = image[ystart:ystop, :, :]
        ctrans_tosearch = self.convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                         (np.int(imshape[1] / scale),
                                          np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        nxblocks = (ch1.shape[1] // self.params.pix_per_cell) - self.params.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.params.pix_per_cell) - self.params.cell_per_block + 1

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.params.pix_per_cell) - self.params.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        hog1 = get_hog_features(ch1, self.params)
        hog2 = get_hog_features(ch2, self.params)
        hog3 = get_hog_features(ch3, self.params)

        tracks = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * self.params.pix_per_cell
                ytop = ypos * self.params.pix_per_cell

                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                spatial_features = bin_spatial(subimg, spatial_size=self.params.spatial_size)
                hist_features = color_hist(subimg, nbins=self.params.hist_bins)

                # Scale features and make a prediction
                test_features = self.scaler.transform(np.hstack((spatial_features,
                                                              hist_features,
                                                              hog_features)).reshape(1, -1))

                # Scale features and make a prediction
                test_prediction = self.svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                                  (xbox_left + win_draw, ytop_draw + win_draw + ystart),
                                  (255, 0, 0), 8)
                    tracks.append(((xbox_left, ytop_draw + ystart),
                                      (xbox_left + win_draw, ytop_draw + win_draw + ystart)))



        return tracks

    def process_image(self, image):


        ystart = 360
        ystop = 560
        scale = 1.0
        self.tracks.append(self.find_cars(image, ystart, ystop, scale))

        ystart = 380
        ystop = 580
        scale = 1.5
        self.tracks.append(self.find_cars(image, ystart, ystop, scale))

        ystart = 400
        ystop = 660
        scale = 2.0
        self.tracks.append(self.find_cars(image, ystart, ystop, scale))
        tracks = [item for sublist in self.tracks for item in sublist]

        if len(tracks) > 0:
            last_track = tracks[-1]
            dist = []
            for track in tracks:
                dist.append(self.len_points(track, last_track))
            min_d = min(dist)
            if min_d < self.boxes_len_threshold:
                ind = dist.index(min_d)
                tracks[-1] = np.asarray(tracks[ind]) * self.alpha + (1.0 - self.alpha) * np.asarray(tracks[-1])
                tracks[-1] = np.asarray(tracks[-1]).astype(int)

        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        heat = self.add_heat(heat, tracks)
        heat = self.apply_threshold(heat, 2)

        # Find final boxes from heatmap using label function
        labels = label(heat)
        new_img = self.draw_labeled_bboxes(np.copy(image), labels)

        return new_img


    def len_points(self, p1, p2):  # Distance beetween two points
        return np.sqrt((p1[0][0] - p2[0][0]) ** 2 + (p1[0][1] - p2[0][1]) ** 2)
