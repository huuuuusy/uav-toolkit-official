from __future__ import absolute_import
from uav.experiments import *
import os
import json

from tracker.siamfc import TrackerSiamFC

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # the path of data folder
    root_dir = "/mnt/second/hushiyu/UAV/"

    # the path to save the experiment result
    save_dir = os.path.join(root_dir, 'result')
    repetitions = 1

    # please select train/test/val
    subset = 'train'

    """
    I. RUN TRACKER
    """ 

    net_path = os.path.join(os.path.split(os.path.realpath(__file__))[0],'pretrained', 'siamfc','model.pth')

    tracker = TrackerSiamFC(net_path=net_path)

    experiment = ExperimentUAV(root_dir, save_dir, subset, repetitions)
    experiment.run(tracker, visualize=False, save_img=False)


    ### EVALUATION
    # tracker_names =  ['ATOM','DiMP','ECO','KeepTrack','KYS','MixFormer','PrDiMP','SiamFC','SiamRCNN','SiamRPN','SuperDiMP']

    # subset = 'test'

    # experiment = ExperimentUAV(root_dir, save_dir, subset, 1)
    # experiment.report(tracker_names)
