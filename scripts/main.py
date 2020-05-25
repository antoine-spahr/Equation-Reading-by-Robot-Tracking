import click
import os
import sys
sys.path.append('../')

import imageio
import numexpr

import matplotlib.pyplot as plt

from src.detection.detector import Detector
from src.tracking.tracker import Tracker
from src.utils.output_utils import draw_output_frame, save_video
from src.utils.print_utils import print_progessbar

@click.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--output_path', type=click.Path(exists=False), default='output.avi')
def main(video_path, output_path):
    """
    Read the equation showed on the video at `video_path` by tracking the robot.
    The video with the written equation and the robot trajectory is saved at
    `output_path`.
    """
    # load video from video_path
    video = imageio.get_reader(video_path)
    N_frames = int(video.get_meta_data()['duration'] * video.get_meta_data()['fps'])

    # Read 1st frame and analyse the environment (classify operator and digits)
    frame1 = video.get_data(0)

    detector = Detector(frame1, '../models/Digit_model.pickle',
                               '../models/Operators_model.pickle',
                               '../models/KMeans_centers.json')
    eq_element_list = detector.analyse_frame(verbose=True)

    # initialize equation and output-video
    equation = ''
    output_frames = [] # list of frame

    # initialize tracker
    tracker = Tracker()

    # iterate over frames
    print('>>> Equation reading with arrow tracking.')
    is_free = True # state if the arrow has moved to another element (i.e. if there has been an absence of overlap before)
    unsolved = True # whether the equation has been solved or not
    for i, frame in enumerate(video):
        # get position <- track robot position
        tracker.track(frame)
        trajectory = tracker.position_list
        arrow_bbox = tracker.bbox
        # check if bbox overlap with any digit/operator
        overlap_elem_list = [elem for elem in eq_element_list if elem.has_overlap(arrow_bbox, frac=1.0)]
        # append character to equation string
        if len(overlap_elem_list) == 1 and is_free:
            equation += overlap_elem_list[0].value
            is_free = False

        # reset the is_free if no element overlapped
        if len(overlap_elem_list) == 0:
            is_free = True

        # solve expression if '=' is detected
        if len(equation) > 0:
            if equation[-1] == '=' and unsolved:
                # evaluate equation
                results = numexpr.evaluate(equation[:-1]).item()
                # add results to equation
                equation += str(results)
                unsolved = False

        # draw track and equation on output frame
        output_frames.append(draw_output_frame(frame, trajectory, equation, eq_elem_list=eq_element_list))

        # print progress bar
        print_progessbar(i, N_frames, Name='Frame', Size=40, erase=False)

    # check if equation has been solved.
    if unsolved:
        print('>>> WARNING : The equation could not be solved!')
    else:
        print('>>> Successfully read and solve the equation.')

    # save output-video at output_path
    save_video(output_path, output_frames, fps=video.get_meta_data()['fps']*2)
    print(f'>>> Output video saved at {output_path}.')

if __name__ == '__main__':
    main()
