from __future__ import absolute_import, print_function

import os
import glob
import numpy as np
from numpy.lib.utils import info
import six
import pandas as pd
import json


class Turing(object):
    r"""`Metaverse <http://videocube.aitestunion.com>`_ Dataset.
    
    Args:
        root_dir (string): Root directory of dataset where ``train``,
            ``val`` and ``test`` folders exist.
        subset (string, optional): Specify ``train``, ``val`` or ``test``
            subset of Metaverse.
    """
    def __init__(self, root_dir):
        super(Turing, self).__init__()
        self.root_dir = root_dir
        info_path = os.path.join(root_dir, 'turing', 'attribute', 'turing_info.json')

        with open(info_path) as f:
            turing_info = json.load(f)

        def reverse_dict(dic):
            keys = list(dic.keys())
            values = list(dic.values())
            keys.reverse()
            values.reverse()
            return dict(zip(keys,values))

        # turing_info = reverse_dict(turing_info)

        self.seq_dirs = []
        self.anno_files = []
        self.restart_files = []

        self.seq_names = []

        for key, val in turing_info.items():
            dataset = val['dataset']
            filename = val['name']
            seq_dir = val['img_dir']

            anno_file = os.path.join(root_dir, 'turing', 'attribute','groundtruth','{}.txt'.format(filename))
            restart_file =  os.path.join(root_dir, 'turing', 'attribute','restart','{}.txt'.format(filename))

            self.seq_dirs.append(seq_dir)
            self.anno_files.append(anno_file)
            self.restart_files.append(restart_file)

            self.seq_names.append(filename)
            


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


        img_files = sorted(glob.glob(os.path.join(
            self.seq_dirs[index], '*.jpg')))
        anno = np.loadtxt(self.anno_files[index], delimiter=',')
        restart_flag = np.loadtxt(self.restart_files[index], delimiter=',', dtype=int)

        return img_files, anno, restart_flag
        

    def __len__(self):
        return len(self.seq_names)

