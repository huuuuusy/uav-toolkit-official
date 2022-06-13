from __future__ import absolute_import, division, print_function

import os
from time import time
import numpy as np

import json
import matplotlib.pyplot as plt
import matplotlib

from ..datasets import Turing
from ..utils.metrics import center_error,normalized_center_error, iou, diou, giou
from ..utils.ioutils import compress
from ..utils.help import makedir
import cv2 as cv
import pandas as pd
import seaborn as sns
from .turing_exp import get_exp_info


class ExperimentTuring(object):
    r"""Experiment pipeline and evaluation toolkit for Metaverse dataset.
    
    Args:
        root_dir (string): 
            Root directory of Metaverse dataset where ``train``, ``val`` and ``test`` folders exist.
        save_dir (string): 
            Save directory of Metaverse dataset to save the experiment results.
        subset (string): 
            Specify ``train``, ``val`` or ``test`` subset of Metaverse.
        repetition (int): 
            The num of repetition. To ensure the accuracy of the experimental results, it is generally repeated three times.
    """
    def __init__(self, root_dir, save_dir, repetition):
        super(ExperimentTuring, self).__init__()
        self.root_dir = root_dir
        self.dataset = Turing(root_dir)
        self.result_dir = os.path.join(save_dir, 'results') 
        self.report_dir = os.path.join(save_dir, 'reports') 
        self.time_dir = os.path.join(save_dir, 'time')
        self.analysis_dir = os.path.join(save_dir, 'analysis')
        self.img_dir = os.path.join(save_dir, 'image')
        
        self.nbins_iou = 101 
        self.nbins_ce = 401 
        self.ce_threshold = 20 
        self.corrcoef_threshold = 101
        self.corrcoef_sort = 0.75

        self.repetition = repetition 
        makedir(save_dir)
        makedir(self.result_dir)
        makedir(self.report_dir)
        makedir(self.time_dir)
        makedir(self.analysis_dir)
        makedir(self.img_dir)
        

    def run(self, tracker, visualize, save_img, method):
        """
        Run the tracker on Metaverse subset.
        """
        print('Running tracker %s on Metaverse...' % tracker.name)
        print('Evaluation mechanism: %s' % method)

        for s, (img_files, anno, restart_flag) in enumerate(self.dataset):
            seq_name = str(self.dataset.seq_names[s]) 
            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))

            print('  Repetition: %d'%self.repetition)
            if method == None:
                # tracking in OPE mechanism
                record_name = tracker.name
            else:
                # tracking in R-OPE mechanism
                record_name = tracker.name + '_' + method

            makedir(os.path.join(self.result_dir, record_name))
            makedir(os.path.join(self.time_dir, record_name))

            tracker_result_dir = os.path.join(self.result_dir, record_name)
            tracker_time_dir = os.path.join(self.time_dir, record_name)
            makedir(tracker_result_dir)                
            makedir(tracker_time_dir)

            # setting the dir for saving tracking result images
            makedir( os.path.join(self.img_dir, record_name))
            tracker_img_dir = os.path.join(self.img_dir, record_name)
            makedir(tracker_img_dir)
            seq_result_dir = os.path.join(tracker_img_dir, seq_name)
            makedir(seq_result_dir)

            # setting the path for saving tracking result
            record_file = os.path.join(tracker_result_dir, '%s_%s_%s.txt'%(record_name , seq_name , str(self.repetition)))

            # setting the path for saving tracking result (restart position in R-OPE mechanism)
            init_positions_file = os.path.join(tracker_result_dir, 'init_%s_%s_%s.txt'%(record_name , seq_name , str(self.repetition)))

            # setting the path for saving tracking time 
            time_file = os.path.join(tracker_time_dir, '%s_%s_%s.txt'%(record_name , seq_name , str(self.repetition)))
            
            if os.path.exists(record_file):
                print('  Found results, skipping ', seq_name)
                continue
            
            # tracking loop
            if method == None:
                # tracking in original OPE mechanism
                boxes, times = tracker.track(seq_name, img_files, anno, restart_flag, visualize, seq_result_dir, save_img, method)
            elif method == 'restart':
                # tracking in novel R-OPE mechanism
                boxes, times, init_positions = tracker.track(seq_name, img_files, anno,  restart_flag, visualize, seq_result_dir, save_img, method)
                # save the restart locations
                f_init = open(init_positions_file, 'w')
                for num in init_positions:
                    f_init.writelines(str(num)+'\n')
                f_init.close()

            self._record(record_file, time_file, boxes, times)


    def report(self, tracker_names, exp_number):
        """
        Evaluate the tracker on Metaverse subset.
        """
        assert isinstance(tracker_names, (list, tuple))
        
        subjects,_ = get_exp_info(exp_number)

        report_dir = os.path.join(self.report_dir, exp_number)
        makedir(report_dir)

        performance = {}

        info_path = os.path.join(self.root_dir, 'turing', 'attribute', 'turing_info.json')

        with open(info_path) as f:
            turing_info = json.load(f)

        for name in tracker_names:
            
            single_report_file = os.path.join(self.analysis_dir, '%s_%s.json'%(name, exp_number))
            
            if os.path.exists(single_report_file):
                f = open(single_report_file,'r',encoding='utf-8')
                single_performance = json.load(f)
                performance.update({name:single_performance})
                f.close()
                print('Existing result in {}'.format(name))
                continue
            else:
                performance.update({name: {
                    'overall': {},
                    'seq_wise': {}}})

            seq_num = len(subjects)

            # save the original precision value for original precision plot
            prec_curve = np.zeros((seq_num, self.nbins_ce))
            # save the novel precision value for normalized precision plot
            norm_prec_curve = np.zeros((seq_num, self.nbins_ce))
            # save the normalize precision score
            norm_prec_score  = np.zeros(seq_num)

            # save the original precision value for original precision plot
            prec_curve_length = np.zeros((seq_num, self.nbins_ce))
            # save the novel precision value for normalized precision plot
            norm_prec_curve_length = np.zeros((seq_num, self.nbins_ce))
            # save the normalize precision score
            norm_prec_score_length  = np.zeros(seq_num)
            seq_length = np.zeros((seq_num,1))

            for s in range(len(subjects)):

                num = subjects[s]

                seq_length[s] = turing_info[num]['length']

                filename = turing_info[num]['name']

                start = 0
                end = int(turing_info[num]['length'])

                # get the information of selected video
                index = self.dataset.seq_names.index(filename)
                img_files, anno, _ = self.dataset[index]

                img_files = img_files[start:end]
                anno = anno[start:end]

                print('repetition %s: Evaluate tracker %s in video num %s'%(self.repetition, name, num))
                
                absent_path = os.path.join(self.root_dir, 'turing', 'attribute','absent', '{}.txt'.format(filename))
                absent = np.loadtxt(absent_path)[start:end]
                
                corrcoef_path = os.path.join(self.root_dir, 'turing', 'attribute','absent', '{}.txt'.format(filename))
                corrcoef = np.loadtxt(corrcoef_path)[start:end]

                assert len(corrcoef) == len(absent) == len(anno) == len(img_files)

                absent = pd.DataFrame(absent, columns= ['absent'])
                corrcoef = pd.DataFrame(corrcoef, columns= ['corrcoef'])

                # frame resolution
                img_height = cv.imread(img_files[0]).shape[0]
                img_width = cv.imread(img_files[0]).shape[1]
                img_resolution = (img_width,img_height)
                bound = img_resolution

                if 'Experimenter' in name:
                    boxes_dir = os.path.join(self.result_dir, 'Experimenter')
                    boxes_files = []
                    for boxes_file in os.listdir(boxes_dir):
                        if str(filename) == boxes_file.split('.')[0][6:]:
                            boxes_files.append(boxes_file)

                    prec_curve_human = np.zeros((len(boxes_files), self.nbins_ce))

                    norm_prec_curve_human = np.zeros((len(boxes_files), self.nbins_ce))

                    norm_prec_score_human  = np.zeros(len(boxes_files))

                    for i in range(len(boxes_files)):

                        boxes_file = boxes_files[i]

                        boxes = np.loadtxt(os.path.join(boxes_dir,boxes_file), delimiter=',')[start:end]    

                        anno = np.array(anno)
                        boxes = np.array(boxes)

                        assert len(boxes) == len(anno)

                        seq_center_errors, seq_norm_center_errors, flags = self._calc_metrics(boxes, anno, bound)

                        seq_center_errors = pd.DataFrame(seq_center_errors, columns = ['seq_center_errors'])

                        seq_norm_center_errors = pd.DataFrame(seq_norm_center_errors, columns= ['seq_norm_center_errors'])
                        flags = pd.DataFrame(flags, columns = ['flags'])

                        data = pd.concat([seq_center_errors,seq_norm_center_errors,flags, absent, corrcoef],axis=1) 
                        
                        # Frames without target and transition frames are not included in the evaluation
                        data = data[data.apply(lambda x: x['absent']== 0, axis=1)]

                        data = data.drop(labels=['absent'], axis = 1)

                        seq_center_errors = data['seq_center_errors']
                        seq_norm_center_errors = data['seq_norm_center_errors']   
                        flags = data['flags']  
                        
                        # Calculate the proportion of all the frames that fall into area 5 (groundtruth area)
                        norm_prec_score_human[i] = np.nansum(flags)/len(flags)

                        # Save the 5 curves of the tracker on the current video
                        prec_curve_human[i], norm_prec_curve_human[i]= self._calc_curves(seq_center_errors, seq_norm_center_errors)
                    

                    if name == 'Experimenter_Mean':

                        prec_curve[s] = np.nanmean(prec_curve_human, axis=0)

                        norm_prec_curve[s] = np.nanmean(norm_prec_curve_human, axis=0)

                        norm_prec_score[s] = np.nanmean(norm_prec_score_human, axis=0)
                    
                    elif name == 'Experimenter_Bottom':
                        prec_curve[s] = np.min(prec_curve_human, axis=0)

                        norm_prec_curve[s] = np.min(norm_prec_curve_human, axis=0)

                        norm_prec_score[s] = np.min(norm_prec_score_human, axis=0)

                    elif name == 'Experimenter_Middle':
                        prec_curve[s] = np.median(prec_curve_human, axis=0)

                        norm_prec_curve[s] = np.median(norm_prec_curve_human, axis=0)

                        norm_prec_score[s] = np.median(norm_prec_score_human, axis=0)
                    
                    elif name == 'Experimenter_Top':
                        prec_curve[s] = np.max(prec_curve_human, axis=0)

                        norm_prec_curve[s] = np.max(norm_prec_curve_human, axis=0)

                        norm_prec_score[s] = np.max(norm_prec_score_human, axis=0)

                else:
                    # evaluate algorithms
                    boxes = np.loadtxt(os.path.join(self.result_dir, name, '{}_{}_{}.txt'.format(name, filename, self.repetition)), delimiter=',')[start:end]

                    anno = np.array(anno)
                    boxes = np.array(boxes)

                    for box in boxes:
                        # correction of out-of-range coordinates
                        box[0] = box[0] if box[0] > 0 else 0
                        box[2] = box[2] if box[2] < img_width - box[0] else img_width - box[0]
                        box[1] = box[1] if box[1] > 0 else 0
                        box[3] = box[3] if box[3] < img_height - box[1] else img_height - box[1]

                    assert boxes.shape == anno.shape
                    
                    # calculate ious, gious, dious for success plot
                    # calculate center errors and normalized center errors for precision plot
                    seq_center_errors, seq_norm_center_errors, flags = self._calc_metrics(boxes, anno, bound)
                    
                    seq_center_errors = pd.DataFrame(seq_center_errors, columns = ['seq_center_errors'])
                    seq_norm_center_errors = pd.DataFrame(seq_norm_center_errors, columns= ['seq_norm_center_errors'])
                    flags = pd.DataFrame(flags, columns = ['flags'])

                    data = pd.concat([seq_center_errors,seq_norm_center_errors,flags, absent, corrcoef],axis=1) 
                    
                    # Frames without target and transition frames are not included in the evaluation
                    data = data[data.apply(lambda x: x['absent']== 0, axis=1)]

                    data = data.drop(labels=['absent'], axis = 1)

                    seq_center_errors = data['seq_center_errors']
                    seq_norm_center_errors = data['seq_norm_center_errors']   
                    flags = data['flags']  
                    
                    # Calculate the proportion of all the frames that fall into area 5 (groundtruth area)
                    norm_prec_score[s] = np.nansum(flags)/len(flags)

                    # Save the 5 curves of the tracker on the current video
                    prec_curve[s], norm_prec_curve[s]= self._calc_curves(seq_center_errors, seq_norm_center_errors)

                # Update the results in current video (Only save scores)
                performance[name]['seq_wise'].update({num: {
                    'precision_score': prec_curve[s][self.ce_threshold],
                    'norm_prec_score':norm_prec_score[s]}})
            
            prec_curve_length = sum(prec_curve*seq_length)/sum(seq_length)
            norm_prec_curve_length = sum(norm_prec_curve*seq_length)/sum(seq_length)

            norm_prec_score_length = float(sum(sum(norm_prec_score*seq_length.T))/sum(seq_length))

            # Average each curve
            prec_curve = np.nanmean(prec_curve, axis=0)
            norm_prec_curve = np.nanmean(norm_prec_curve, axis=0)

            # Generate average score
            prec_score = prec_curve[self.ce_threshold]
            norm_prec_score = np.nansum(norm_prec_score) / np.count_nonzero(norm_prec_score)

            prec_score_length = prec_curve_length[self.ce_threshold]

            # store overall performance
            performance[name]['overall'].update({
                'precision_score': prec_score,
                'norm_prec_score':norm_prec_score,
                'precision_score_length': prec_score_length,
                'norm_prec_score_length':norm_prec_score_length,
                'precision_curve': prec_curve.tolist(),
                'normalized_precision_curve': norm_prec_curve.tolist(),
                'precision_curve_length': prec_curve_length.tolist(),
                'normalized_precision_curve_length': norm_prec_curve_length.tolist()})

            with open(single_report_file, 'w') as f:
                json.dump(performance[name], f, indent=4)

        report_file = os.path.join(report_dir, 'performance.json')
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)

        self.plot_curves_([report_file], tracker_names, exp_number)
        return performance


    def _calc_metrics(self, boxes, anno, bound):
        """
        Calculate the evaluation metrics.
        """
        # valid = (anno[...,2]*anno[...,3] != 0)
        valid = ~np.any(np.isnan(anno), axis=1)
        if len(valid) == 0:
            print('Warning: no valid annotations')
            return None, None, None
        else:
            # calculate ious, dious and gious for success plot
            # ious = iou(boxes[valid, :], anno[valid, :])
            # dious = diou(boxes[valid, :], anno[valid, :])
            # gious = giou(boxes[valid, :], anno[valid, :])
            # calculate center error for original precision plot
            center_errors = center_error(
                boxes[valid, :], anno[valid, :])
            # calculate normalized center error for the normalized precision plot
            norm_center_errors, flags = normalized_center_error(boxes[valid, :], anno[valid, :], bound)
            
            return center_errors, norm_center_errors, flags


    def _calc_curves(self, center_errors, norm_center_errors):
        """
        Calculate the evaluation curves.
        """
        # ious = np.asarray(ious, float)[:, np.newaxis]
        # dious = np.asarray(dious, float)[:, np.newaxis]
        # gious = np.asarray(gious, float)[:, np.newaxis]
        center_errors = np.asarray(center_errors, float)[:, np.newaxis]
        norm_center_errors = np.asarray(norm_center_errors, float)[:, np.newaxis]

        # corrcoefs = np.asarray(corrcoefs, float)[:, np.newaxis]

        # thr_iou = np.linspace(0, 1, self.nbins_iou)[np.newaxis, :]
        thr_ce = np.arange(0, self.nbins_ce)[np.newaxis, :]
        thr_nce = np.linspace(0, 1, self.nbins_ce)[np.newaxis, :]
        # thr_corrcoef = np.linspace(0, 1, self.corrcoef_threshold)[np.newaxis, :]
 
        # bin_iou = np.greater(ious, thr_iou)
        # bin_diou = np.greater(dious, thr_iou)
        # bin_giou = np.greater(gious, thr_iou)
        bin_ce = np.less(center_errors, thr_ce)
        bin_nce = np.less(norm_center_errors, thr_nce)

        # bin_corrcoef = np.less(corrcoefs, thr_corrcoef)
        # bin_succ = np.greater(ious, 0.5).T

        # corrcoef_curve = []

        # for i in range(bin_corrcoef.shape[1]):
        #     frame_bin_corrcoef = bin_corrcoef[:,i] # 判断当前帧与101个分段阈值的比较
        #     succ_corrcoef = frame_bin_corrcoef*bin_succ # 判断当corrcoef小于特定阈值的帧中是否跟踪成功，True代表当前帧corrcoef小于特定阈值且当前帧跟踪成功
        #     score = np.sum(succ_corrcoef)/np.sum(frame_bin_corrcoef) if np.sum(frame_bin_corrcoef)!=0 else 0

        #     if i == 0:
        #         previous_score = 0

        #     # score = max(previous_score, score)
        #     corrcoef_curve.append(score)
        #     previous_score = score

        # succ_curve = np.nanmean(bin_iou, axis=0)
        # succ_dcurve = np.nanmean(bin_diou, axis=0)
        # succ_gcurve = np.nanmean(bin_giou, axis=0)
        prec_curve = np.nanmean(bin_ce, axis=0)
        norm_prec_curve = np.nanmean(bin_nce, axis=0)
        # corrcoef_curve = np.array(corrcoef_curve)

        return prec_curve, norm_prec_curve

    
    def presentation_name(self, name):
        """将restart名字规范化"""
        if '_restart' in name:
            name = name.replace('_restart', '_Human')
        return name


    def plot_curves_(self, report_files, tracker_names, exp_number):
        """
        Drow Plot
        """
        assert isinstance(report_files, list), \
            'Expected "report_files" to be a list, ' \
            'but got %s instead' % type(report_files)
        
        _,exp_number_name = get_exp_info(exp_number)
    
        report_dir = os.path.join(self.report_dir, exp_number)
        
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        performance = {}
        for report_file in report_files:
            with open(report_file) as f:
                performance.update(json.load(f))
    
        prec_file = os.path.join(report_dir, 'precision_plot_{}.png'.format(exp_number))
        norm_prec_file = os.path.join(report_dir, 'norm_precision_plot_{}.png'.format(exp_number))
        prec_length_file = os.path.join(report_dir, 'precision_length_plot_{}.png'.format(exp_number))
        norm_prec_length_file = os.path.join(report_dir, 'norm_precision_length_plot_{}.png'.format(exp_number))
        
        key = 'overall'

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # filter performance by tracker_names
        performance = {k:v for k,v in performance.items() if k in tracker_names}

        # sort trackers by precision score
        tracker_names = list(performance.keys())
        
        prec = [t[key]['precision_score'] for t in performance.values()]

        inds = np.argsort(prec)[::-1]

        tracker_names = [tracker_names[i] for i in inds]

        # plot precision curves
        thr_ce = np.arange(0, self.nbins_ce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_ce,
                            performance[name][key]['precision_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            

            legends.append('%s: [%.3f]' % (self.presentation_name(name), performance[name][key]['precision_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='lower right', bbox_to_anchor=(1., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Location error threshold',
               ylabel='Precision',
               xlim=(0, thr_ce.max()), ylim=(0, 1),
               title='Precision plots \n ({})'.format(exp_number_name))
        ax.grid(True)
        fig.tight_layout()

        print('Saving precision plots to', prec_file)
        fig.savefig(prec_file, dpi=300)

        # plot normalized precision curves
        tracker_names = list(performance.keys())
        # prec = [t[key]['normalized_precision_score'] for t in performance.values()]
        prec = [t[key]['norm_prec_score'] for t in performance.values()]

        inds = np.argsort(prec)[::-1]

        tracker_names = [tracker_names[i] for i in inds]

        # plot normalized precision curves
        thr_nce = np.linspace(0, 1, self.nbins_ce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_nce,
                            performance[name][key]['normalized_precision_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (self.presentation_name(name), performance[name][key]['norm_prec_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        # legend = ax.legend(lines, legends, loc='center left', bbox_to_anchor=(1, 0.5))
        legend = ax.legend(lines, legends, loc='lower right', bbox_to_anchor=(1., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Normalized location error threshold',
               ylabel='Normalized precision',
               xlim=(0, thr_nce.max()), ylim=(0, 1),
               title='Normalized precision plots \n ({})'.format(exp_number_name))
        ax.grid(True)
        fig.tight_layout()

        print('Saving normalized precision plots to', norm_prec_file)
        fig.savefig(norm_prec_file, dpi=300)

        # plot weighted precision curves
        thr_ce = np.arange(0, self.nbins_ce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_ce,
                            performance[name][key]['precision_curve_length'],
                            markers[i % len(markers)])
            lines.append(line)
            

            legends.append('%s: [%.3f]' % (self.presentation_name(name), performance[name][key]['precision_score_length']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='lower right', bbox_to_anchor=(1., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Location error threshold',
               ylabel='Precision',
               xlim=(0, thr_ce.max()), ylim=(0, 1),
               title='Precision plots (weighted by length) \n ({})'.format(exp_number_name))
        ax.grid(True)
        fig.tight_layout()

        print('Saving weighted precision plots to', prec_length_file)
        fig.savefig(prec_length_file, dpi=300)

        # plot weighted normalized precision curves
        tracker_names = list(performance.keys())
        # prec = [t[key]['normalized_precision_score'] for t in performance.values()]
        prec = [t[key]['norm_prec_score_length'] for t in performance.values()]

        inds = np.argsort(prec)[::-1]

        tracker_names = [tracker_names[i] for i in inds]

        # plot normalized precision curves
        thr_nce = np.linspace(0, 1, self.nbins_ce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_nce,
                            performance[name][key]['normalized_precision_curve_length'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (self.presentation_name(name), performance[name][key]['norm_prec_score_length']))
        matplotlib.rcParams.update({'font.size': 7.4})
        # legend = ax.legend(lines, legends, loc='center left', bbox_to_anchor=(1, 0.5))
        legend = ax.legend(lines, legends, loc='lower right', bbox_to_anchor=(1., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Normalized location error threshold',
               ylabel='Normalized precision',
               xlim=(0, thr_nce.max()), ylim=(0, 1),
               title='Normalized precision plots (weighted by length) \n ({})'.format(exp_number_name))
        ax.grid(True)
        fig.tight_layout()

        print('Saving normalized precision plots to', norm_prec_length_file)
        fig.savefig(norm_prec_length_file, dpi=300)


    def _record(self, record_file, time_file, boxes, times):
        np.savetxt(record_file, boxes, fmt='%d', delimiter=',')
        print('Results recorded at', record_file)

        times = times[:, np.newaxis]
        if os.path.exists(time_file):
            exist_times = np.loadtxt(time_file, delimiter=',')
            if exist_times.ndim == 1:
                exist_times = exist_times[:, np.newaxis]
            times = np.concatenate((exist_times, times), axis=1)
        np.savetxt(time_file, times, fmt='%.8f', delimiter=',')
