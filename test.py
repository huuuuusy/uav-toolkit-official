from __future__ import absolute_import
from metaverse.experiments import *
import os
import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_number', type=str, help='the number of GPU you would like to use', default="3")
args = parser.parse_args()

from tracker.siamfc import TrackerSiamFC
from tracker.siamrpn import TrackerSiamRPN
from tracker.cvtracker import TrackerKCF, TrackerCSRT, TrackerTLD

def choose_tracker(tracker_name):
    """
    Select a tracker based on its name
    """
    if tracker_name == 'SiamFC':
        net_path = '/home/user1/projects/VIS/videocube-toolkit-dev/toolkit/pretrained/siamfc/model.pth'
        tracker = TrackerSiamFC(net_path=net_path)
    elif tracker_name == 'SiamRPN':
        net_path = '/home/user1/projects/VIS/videocube-toolkit-dev/toolkit/pretrained/siamrpn/model.pth'
        tracker = TrackerSiamRPN(net_path=net_path)
    elif tracker_name == 'KCF':
        tracker = TrackerKCF()
    elif tracker_name == 'CSRT':
        tracker = TrackerCSRT()
    elif tracker_name == 'TLD':
        tracker = TrackerTLD()
    
    return tracker
    

    
if __name__ == '__main__':
    # the path of VideoCube data folder
    root_dir = "/mnt/second/hushiyu/UAV/"
    # the path to save the experiment result
    save_dir = os.path.join(root_dir, 'result')
    repetitions = 1

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

    # ### RUN
    tracker_name = 'SiamRPN'

    subset = 'train'

    tracker = choose_tracker(tracker_name)

    experiment = ExperimentUAV(root_dir, save_dir, subset, repetitions)
    experiment.run(tracker, visualize=False, save_img=False)


    ### EVALUATION
    # tracker_names =  ['ATOM','DiMP','ECO','KeepTrack','KYS','MixFormer','PrDiMP','SiamFC','SiamRCNN','SiamRPN','SuperDiMP']

    # subset = 'test'

    # experiment = ExperimentUAV(root_dir, save_dir, subset, 1)
    # experiment.report(tracker_names)
