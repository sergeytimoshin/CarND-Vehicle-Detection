import glob
import cv2
from moviepy.editor import VideoFileClip

from vehicles.trainer import CarTrainer
from vehicles.carfinder import CarFinder
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def train():
    trainer = CarTrainer()
    trainer.train()
    trainer.dump_training_data()

def draw_boxes(image):
    carfinder = CarFinder()
    boxes = carfinder.frame_proc(image)
    plt.imshow(boxes)
    plt.show()

def test_on_images():
    test_images = glob.glob('../test_images/*.jpg')
    print(test_images)
    for image_name in test_images:
        image = mpimg.imread(image_name)
        draw_boxes(image=image)

def process_video_frame(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(carfinder.frame_proc(image, video=True), cv2.COLOR_BGR2RGB)

def test_on_video():
    clip = VideoFileClip("../project_video.mp4")
    clip = clip.fl_image(process_video_frame)
    clip.write_videofile("../project_video_output.mp4", audio=False)


# train()
carfinder = CarFinder()
# test_on_images()
test_on_video()


