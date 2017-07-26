class Params:
    def __init__(self):
        # Define parameters for feature extraction
        self.color_space = 'LUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 8  # HOG orientations
        self.pix_per_cell = 8  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.hog_channel = 0  # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (16, 16)  # Spatial binning dimensions
        self.hist_bins = 32  # Number of histogram bins
        self.spatial_feat = True  # Spatial features on or off
        self.hist_feat = True  # Histogram features on or off
        self.hog_feat = True  # HOG features on or off
        self.feature_vec = True
        self.height = 720
        self.width = 1280