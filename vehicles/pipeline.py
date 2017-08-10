import glob
import cv2
from moviepy.editor import VideoFileClip

from vehicles.trainer import CarTrainer
from vehicles.carfinder import CarFinder
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
from scipy.ndimage import label
import numpy as np


def train():
    trainer = CarTrainer()
    trainer.train()
    trainer.dump_training_data()


def draw_boxes(image_name, image):
    carfinder = CarFinder()
    boxes = carfinder.process_image(image)
    plt.imshow(boxes)
    # plt.show()
    plt.savefig('out_' + image_name[-5:])


def test_on_images():
    ### 1
    carfinder = CarFinder()

    test_img = mpimg.imread('../test_images/test1.jpg')

    windows = carfinder.slide_window(test_img,
                                     x_start_stop=[None, None],
                                     y_start_stop=[400, 660],  # tune the parameters
                                     xy_window=(64, 64),
                                     xy_overlap=(0.5, 0.5))

    window_img = carfinder.draw_boxes(test_img, windows)
    plt.imshow(window_img);
    matplotlib.rc('xtick', labelsize=15)
    matplotlib.rc('ytick', labelsize=15)
    plt.title('Sliding Windows Technique:', fontsize=15);
    plt.savefig('../output_images/sliding_windows.png', bbox_inches="tight")

    ### 2
    for i in range(1, 6):
        fname = '../test_images/test{}.jpg'.format(i)
        image = mpimg.imread(fname)
        draw_image = np.copy(image)

        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        image = image.astype(np.float32) / 255

        windows = carfinder.slide_window(image,
                                         x_start_stop=[600, None],
                                         y_start_stop=[400, 660],
                                         xy_window=(128, 128),
                                         xy_overlap=(.7, .7))

        hot_windows = carfinder.search_windows(image, windows)

        window_img = carfinder.draw_boxes(draw_image, hot_windows)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
        plt.tight_layout()
        ax1.imshow(draw_image)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(window_img)
        ax2.set_title('Cars found', fontsize=30)
        plt.savefig('../output_images/windows{}.png'.format(i), bbox_inches="tight")

    ### 3

    for i in range(1, 6):
        fname = '../test_images/test{}.jpg'.format(i)
        img = mpimg.imread(fname)

        tracks = []

        ystart = 360
        ystop = 560
        scale = 1.0
        tracks.append(carfinder.find_cars(img, ystart, ystop, scale))

        ystart = 380
        ystop = 580
        scale = 1.5
        tracks.append(carfinder.find_cars(img, ystart, ystop, scale))

        ystart = 400
        ystop = 660
        scale = 2.0
        tracks.append(carfinder.find_cars(img, ystart, ystop, scale))

        tracks = [item for sublist in tracks for item in sublist]

        out_img = carfinder.draw_boxes(img, tracks)

        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        heat = carfinder.add_heat(heat, tracks)
        heat = carfinder.apply_threshold(heat, 2)

        labels = label(heat)
        new_img = carfinder.draw_labeled_bboxes(np.copy(img), labels)

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 9))
        plt.tight_layout()
        ax1.imshow(out_img)
        ax1.set_title('Search boxes', fontsize=30)
        ax2.imshow(heat, cmap='hot')
        ax2.set_title('Heatmap', fontsize=30)
        ax3.imshow(new_img)
        ax3.set_title('Bounding boxes', fontsize=30)
        plt.savefig('../output_images/heat_map{}.png'.format(i), bbox_inches="tight")

def test_on_video():
    global carfinder
    carfinder = CarFinder()
    clip = VideoFileClip("../lanes.mp4")
    clip = clip.fl_image(carfinder.process_image)
    clip.write_videofile("../project_video_output.mp4", audio=False)


# train()



test_on_images()

# test_on_video()
