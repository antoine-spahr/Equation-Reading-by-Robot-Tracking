import click
import os
import sys
sys.path.append('../')

import imageio

from src.detection.detector import Detector
from src.tracking.tracker import Tracker
from src.utils.output_utils import draw_output_frame, save_video

@click.commande()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--output_path', type=click.Path(exists=False), default='')
def main(video_path, output_path):
    """
    Read the equation showed on the video at `video_path` by tracking the robot.
    The video with the written equation and the robot trajectory is saved at
    `output_path`.
    """
    # laod video from video_path
    video = imageio.get_reader(video_path)

    # Read 1st frame and read the environment (classify operator and digits)
    fram1 = video.get_data(0)

    detector = Detector(fram1, None, '../models/Operators_model.pickle', '../models/KMeanse_centers.json')
    eq_element_list = detector.analyse_frame()

    # initialize equation <- '' and output-video <- []
    equation = ''
    output_frames = [] # list of frame

    # while frame < video_length or '=' found:
    #       position ; bbox <- track robot position
    #       append position to list
    #       check if bbox more than 90% overlap with any digit/operator
    #           append character to equation string
    #           if character is '='
    #               exit tracking
    #           evaluate equation and append results to strin
    #       draw track and equation on output frame

    # if '=' not found add error message on output-video

    # save output-video at output_path

if __name__ == '__main__':
    main()

################################################################################
"""
TASK
    |--- Fran : Tracker
    |           return position and bbox of arrow in one frame.
    |--- Mat : Training MLP-digit and 1-NN operator
    |           save model in pickle
    |--- Ant : Training KMeans
    |           return centers
    |          generate data
"""
