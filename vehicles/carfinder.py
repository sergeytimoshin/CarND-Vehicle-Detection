from scipy.ndimage import label
from vehicles.utils import get_hog_features, bin_spatial, color_hist
import cv2
import numpy as np
import pickle

class CarFinder:
    def __init__(self):

        with open('data.pickle', 'rb') as f:
            data = pickle.load(f)

        self.scaler = data['scaler']
        self.svc = data['svc']
        self.params = data['params']

        self.params.feature_vec = False

        self.previous_heat = np.zeros((self.params.height, self.params.width))
        self.previous_boxes = []
        self.frame_counter = 0
        self.tracks = []
        self.boxes_overlap_threshold = 3
        self.boxes_len_threshold = 30
        self.alpha = 0.5 # Alpha for low-pass smooth filtering
        self.y_min = self.params.height / 2

    def frame_proc(self, img, video=False, vis=False):
        if (video and self.frame_counter % 2 == 0) or not video:  # Skip every second video frame
            heat = np.zeros_like(img[:, :, 0]).astype(np.float)

            boxes = self.find_cars(img, self.params.height/2+50, self.params.height, self.params.width*3/4, self.params.width, 1.5, 2)
            boxes += self.find_cars(img, self.params.height/2+50, self.params.height, 0, self.params.width/4, 1.5, 2)

            boxes += self.find_cars(img, self.params.height/2+50, self.params.height*3/4, self.params.width*3/4, self.params.width, 1.2, 3)
            boxes += self.find_cars(img, self.params.height/2+50, self.params.height*3/4, 0, self.params.width/4, 1.2, 3)

            boxes += self.find_cars(img, self.params.height/2+50, self.params.height * 3 / 4, self.params.width/4, self.params.width * 3/ 4,
                                    0.75, 4)

            heat = self.add_heat(heat, boxes)
            n_heat= self.previous_heat + heat
            self.previous_heat = heat
            n_heat = self.apply_threshold(n_heat, self.boxes_overlap_threshold)
            heatmap = np.clip(n_heat, 0, 255)
            labels = label(heatmap)
            cars_boxes = self.draw_labeled_bboxes(labels)
            self.previous_boxes = cars_boxes

        else:
            cars_boxes = self.previous_boxes
        imp = self.draw_boxes(np.copy(img), cars_boxes, color=(0, 0, 255), thick=6)
        if vis:
            imp = self.draw_boxes(imp, boxes, color=(0, 255, 255), thick=2)
            for track in self.tracks:
                cv2.circle(imp, (int(track[0]), int(track[1])), 5, color=(255, 0, 255), thickness=4)
        self.frame_counter += 1
        return imp

    def find_cars(self, img, ystart, ystop, xstart, xstop, scale, step):
        ystart = int(ystart)
        ystop = int(ystop)
        xstart = int(xstart)
        xstop = int(xstop)
        boxes = []
        draw_img = np.zeros_like(img)
        img_tosearch = img[ystart:ystop,xstart:xstop,:]
        ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        ch1 = ctrans_tosearch[:,:,0]
        nxblocks = (ch1.shape[1] // self.params.pix_per_cell)-1
        nyblocks = (ch1.shape[0] // self.params.pix_per_cell)-1
        window = 64
        nblocks_per_window = (window // self.params.pix_per_cell) -1
        cells_per_step = step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        hog = get_hog_features(ch1, params=self.params)
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                hog_features = hog[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                xleft = xpos * self.params.pix_per_cell
                ytop = ypos * self.params.pix_per_cell
                subimg = ctrans_tosearch[ytop:ytop+window, xleft:xleft+window]

                spatial_features = bin_spatial(subimg, spatial_size=self.params.spatial_size)
                hist_features = color_hist(subimg, nbins=self.params.hist_bins)

                test_features = self.scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_prediction = self.svc.predict(test_features)
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)+xstart
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    boxes.append(((int(xbox_left), int(ytop_draw+ystart)),(int(xbox_left+win_draw),int(ytop_draw+win_draw+ystart))))
        return boxes

    # Define a function to draw bounding boxes on an image
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        imcopy = np.copy(img)  # Make a copy of the image
        for bbox in bboxes:  # Iterate through the bounding boxes
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        return imcopy

    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        return heatmap  # Return updated heatmap

    def apply_threshold(self, heatmap, threshold):  # Zero out pixels below the threshold in the heatmap
        heatmap[heatmap < threshold] = 0
        return heatmap

    def len_points(self, p1, p2):  # Distance beetween two points
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def draw_labeled_bboxes(self, labels):
        last_tracks = []
        for car_number in range(1, labels[1] + 1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bounding_box = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            size_x = (bounding_box [1][0] - bounding_box [0][0]) / 2.0  # Size of the found box
            size_y = (bounding_box [1][1] - bounding_box [0][1]) / 2.0
            aspect_dimensions = size_x / size_y
            x = size_x + bounding_box[0][0]
            y = size_y + bounding_box[0][1]
            aspect_ratio = (y - self.y_min) / 150.0 + 1.3
            if x > self.params.width*3/4 or x < self.params.width/4:
                aspect_ratio *= 1.5
            aspect_ratio = max(aspect_ratio, aspect_dimensions)
            size_yy = np.sqrt(size_x * size_y / aspect_ratio)
            size_yy = int(size_yy)
            size_xx = int(size_yy * aspect_ratio)

            if (x + 3 * y) > self.params.width*1.5:
                last_tracks.append(np.array([x, y, size_xx, size_yy]))
                if len(self.tracks) > 0:
                    last_track = last_tracks[-1]
                    dist = []
                    for track in self.tracks:
                        dist.append(self.len_points(track, last_track))
                    min_d = min(dist)
                    if min_d < self.boxes_len_threshold:
                        ind = dist.index(min_d)
                        last_tracks[-1] = self.tracks[ind]*self.alpha+(1.0-self.alpha)*last_tracks [-1] # Smooth filter
        self.tracks = last_tracks
        boxes = []
        for track in last_tracks:
            x0 = int(track[0] - track[2])
            x1 = int(track[0] + track[2])
            y0 = int(track[1] - track[3])
            y1 = int(track[1] + track[3])
            boxes.append(((x0, y0), (x1, y1)))
        return boxes