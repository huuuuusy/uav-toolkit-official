from __future__ import absolute_import, print_function

import os
import glob
import numpy as np
from numpy.lib.utils import info
import six
import pandas as pd
import json


class MetaverseAttribute(object):
    r"""`Metaverse <http://videocube.aitestunion.com>`_ Dataset.
    
    Args:
        root_dir (string): Root directory of dataset where ``train``,
            ``val`` and ``test`` folders exist.
        subset (string, optional): Specify ``train``, ``val`` or ``test``
            subset of Metaverse.
    """
    def __init__(self, root_dir, subset):
        super(MetaverseAttribute, self).__init__()
        self.root_dir = root_dir
        self.subset = subset

        dataset_list = ['OTB','VOTLT2019','VOT2016', 'VOT2018', 'VOT2019','GOT-10k','VideoCube','LaSOT']

        img_dir = {}
        for dataset in dataset_list:
            img_file = os.path.join(root_dir,dataset,'attribute','img_dir.json')
            f = open(img_file,'r',encoding='utf-8')
            single_img_dir = json.load(f)            
            img_dir.update({ dataset:single_img_dir})

        info_path = os.path.join(root_dir, 'metaverse','attribute','{}.txt'.format(subset))
        infos = pd.read_csv(info_path)

        self.seq_dirs = []
        self.anno_files = []
        self.restart_files = []

        self.seq_names = []

        self.starts = []
        self.ends = []

        self.indexs = []

        for i in range(len(infos)):
            info = infos.iloc[i]
            dataset = info['datasets']
            filename = info['filenames']
            start = info['starts']
            end = info['ends']
            index = info['indexes']

            seq_dir = img_dir[dataset][filename]['img_dir']
            anno_file = os.path.join(root_dir, dataset, 'attribute','groundtruth','{}.txt'.format(filename))
            restart_file =  os.path.join(root_dir, dataset, 'attribute','restart','{}.txt'.format(filename))

            seq_name = '{}_{}_{}_{}_{}'.format(index, dataset, filename,start,end)
            
            self.seq_dirs.append(seq_dir)
            self.anno_files.append(anno_file)
            self.restart_files.append(restart_file)
            self.starts.append(start)
            self.ends.append(end)
            self.seq_names.append(seq_name)
            


    def __getitem__(self, index):
        r"""        
        Args:
            index (integer or string): Index or name of a sequence.
        
        Returns:
            tuple:
                (img_files, anno, restart_flag), where ``img_files`` is a list of
                file names, ``anno`` is a N x 4 (rectangles) numpy array, while
                ``restart_flag`` is a list of
                restart frames.
        """
        # if isinstance(index, six.string_types):
        #     if not index in self.seq_names:
        #         raise Exception('Sequence {} not found.'.format(index))
        #     index = self.seq_names.index(index)

        start = self.starts[index]
        end = self.ends[index]

        img_files = sorted(glob.glob(os.path.join(
            self.seq_dirs[index], '*.jpg')))[start:end]
        anno = np.loadtxt(self.anno_files[index], delimiter=',')[start:end]
        restart_flag = np.loadtxt(self.restart_files[index], delimiter=',', dtype=int)

        return img_files, anno, restart_flag
        

    def __len__(self):
        return len(self.seq_names)

