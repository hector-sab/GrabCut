#!/usr/bin/env python
"""
Author: Héctor Sánchez
Date: 2018-03-11
Description: Main script to run grabcut3.py
"""

import os

from grabcut import GrabCut

if __name__=='__main__':
  ims_dir = '../1803_010203_01/GOPR7860_PPL/'
  out_dir = '../1803_010203_01/GOPR7860_PPL_MATT/'
  files = os.listdir(ims_dir)
  files.sort()
  for file in files:
    print('File: ',file)
    gc = GrabCut()
    gc.load_image(ims_dir+file)
    gc.set_output(direct=out_dir,name=file[:-4])
    gc.runme()