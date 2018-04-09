#!/usr/bin/env python
"""
Author: Héctor Sánchez
Date: 2018-03-11
Description: Main script to run grabcut3.py
"""

import os
import numpy as np

from grabcut import GrabCut

if __name__=='__main__':
  ims_dir = '../1803_010203_01/GOPR7860_PPL/'
  out_dir = '../1803_010203_01/GOPR7860_PPL_MATT/'
  files = os.listdir(ims_dir)
  files.sort()
  
  files = np.array(files)
  num_files = files.shape[0]
  num2take = int(num_files*.2)

  ind = np.arange(num_files)
  np.random.shuffle(ind)
  files[ind] = files
  files = files[:num2take]
  files = list(files)

  for i,file in enumerate(files):
    print('File {0}/{1}: {2}'.format(i,num2take,file))
    gc = GrabCut()
    gc.load_image(ims_dir+file)
    gc.set_output(direct=out_dir,name=file[:-4])
    gc.runme()