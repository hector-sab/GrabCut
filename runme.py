#!/usr/bin/env python
"""
Author: Héctor Sánchez
Date: 2018-03-11
Description: Main script to run grabcut3.py
"""

from grabcut3 import GrabCut

if __name__=='__main__':
  gc = GrabCut()
  gc.load_image('../ims/a.JPG')
  gc.matte()