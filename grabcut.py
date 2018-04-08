#!/usr/bin/env python
"""
1- Load Image
2- Select ROI
3- Grabcut over ROI
"""

import numpy as np
import cv2

BLUE = [255,0,0]        # Rectangle color
RED = [0,0,255]         # PBG
GREEN = [0,255,0]       # FG
BLACK = [0,0,0]         # BG
WHITE = [255,255,255]   # PFG

COLOR_SET = {'bg':BLACK,'fg':GREEN,'pbg':RED,'pfg':WHITE}
MASK_SET = {'bg':0,'fg':1,'pbg':2,'pfg':3}


class GrabCut:
  def __init__(self):
    self.im_tmp = None # Multi-use

    self.im_original = None # Not touched
    self.im_front = None
    self.im_front_orig = None # used by cv2.grabcut
    self.im_matted = None
    self.mask = None
    self.gc_mask = None
    self.gc_mask2 = None

    self.IMAGE_LOADED = False
    self.ROI_SELECTED = False
    self.LBUTTON_DOWN = False

    self.ROI = [(0,0),(0,0)]
    self.LINEP = [(0,0),(0,0)]
    self.THICKNESS = 2
    self.CLIC_TYPE = 'fg'

    self.GC_REC = True
    self.RECTANGLE = None
    self.BGMODEL = np.zeros((1,65),dtype=np.float64)
    self.FGMODEL = np.zeros((1,65),dtype=np.float64)

  def load_image(self,path):
    self.im_original = cv2.imread(path)
    self.im_front = self.im_original.copy()
    self.IMAGE_LOADED = True

  def runme(self):
    if self.IMAGE_LOADED:
      WIN_NAME_1 = 'Original Image'
      WIN_NAME_2 = 'ROI'
      WIN_NAME_3 = 'Matted'
      WIN_NAME_4 = 'Mask'
      WIN_NAME_5 = 'GC Mask'

      WIN_TYPE = cv2.WINDOW_GUI_NORMAL

      while True:
        if not self.ROI_SELECTED:
          cv2.namedWindow(WIN_NAME_1,WIN_TYPE)
          cv2.setMouseCallback(WIN_NAME_1,self.onMouse)
          cv2.imshow(WIN_NAME_1,self.im_front)
        else:
          cv2.destroyWindow(WIN_NAME_1)
          cv2.namedWindow(WIN_NAME_2,WIN_TYPE)
          cv2.namedWindow(WIN_NAME_3,WIN_TYPE)
          cv2.namedWindow(WIN_NAME_4,WIN_TYPE)
          cv2.namedWindow(WIN_NAME_5,WIN_TYPE)

          cv2.setMouseCallback(WIN_NAME_2,self.onMouse)
          cv2.imshow(WIN_NAME_2,self.im_front)
          cv2.imshow(WIN_NAME_3,self.im_matted)
          cv2.imshow(WIN_NAME_4,self.mask*80)
          cv2.imshow(WIN_NAME_5,self.gc_mask*80)

        k = 0xFF & cv2.waitKey(1)
        self.keyboard_instructions(k)
    else:
      print('Image not loaded. Load an image first.')
  
  def keyboard_instructions(self,k):
    if k==27: # Esc
      cv2.destroyAllWindows()
      exit()
    
    elif k ==ord('s'):
      print('Starting Matting')

      if self.GC_REC:
        self.gc_mask2,self.BGMODEL,self.FGMODEL = cv2.grabCut(self.im_front_orig,self.gc_mask,
                    self.RECTANGLE,self.BGMODEL,self.FGMODEL,5,cv2.GC_INIT_WITH_RECT)
        self.GC_REC = not self.GC_REC
      else:
        self.gc_mask2,self.BGMODEL,self.FGMODEL = cv2.grabCut(self.im_front_orig,self.gc_mask,
                    self.RECTANGLE,self.BGMODEL,self.FGMODEL,5,cv2.GC_INIT_WITH_MASK)
      self.mask = np.where((self.gc_mask2==3)|(self.gc_mask2==1),255,0).astype('uint8')

      self.im_matted = cv2.bitwise_and(self.im_front_orig,self.im_front_orig,mask=self.mask)
      print('All done!')


    elif k==ord('1'):
      self.CLIC_TYPE = 'fg'
    elif k==ord('2'):
      self.CLIC_TYPE = 'bg'
    elif k==ord('3'):
      self.CLIC_TYPE = 'pfg'
    elif k==ord('4'):
      self.CLIC_TYPE = 'pbg'
    
    elif k==ord('+'): # bigger line thickness
      if self.THICKNESS<20:
        self.THICKNESS += 1
    elif k==ord('-'): # smaller line thickness
      if self.THICKNESS>1:
        self.THICKNESS -= 1

  
  def onMouse(self,event,x,y,flags,params):
    if not self.ROI_SELECTED:
      if event==cv2.EVENT_LBUTTONDOWN:
        self.LBUTTON_DOWN = True
        self.ROI[0] = (x,y)
        self.im_tmp = self.im_front.copy()
      elif event==cv2.EVENT_MOUSEMOVE and self.LBUTTON_DOWN:
        self.ROI[1] = (x,y)
        self.im_front = self.im_tmp.copy()
        cv2.rectangle(self.im_front,self.ROI[0],self.ROI[1],BLUE,2)
      elif event==cv2.EVENT_LBUTTONUP and self.LBUTTON_DOWN:
        self.LBUTTON_DOWN = False
        self.ROI_SELECTED = True

        self.ROI[1] = (x,y)
        
        # Ensure order
        if self.ROI[0][0]>self.ROI[1][0]:
          x1 = self.ROI[1][0]
          x2 = self.ROI[0][0]
        else:
          x1 = self.ROI[0][0]
          x2 = self.ROI[1][0]
        if self.ROI[0][1]>self.ROI[1][1]:
          y1 = self.ROI[1][1]
          y2 = self.ROI[0][1]
        else:
          y1 = self.ROI[0][1]
          y2 = self.ROI[1][1]
        
        self.ROI[0] = (x1,y1)
        self.ROI[1] = (x2,y2)
        
        self.im_front = self.im_tmp[self.ROI[0][1]:self.ROI[1][1],
                                    self.ROI[0][0]:self.ROI[1][0]]
        self.im_tmp = self.im_front.copy()
        self.im_front_orig = self.im_front.copy()
        self.im_matted = self.im_front.copy()
        self.mask = np.zeros_like(self.im_front[:,:,0])
        self.gc_mask = np.zeros_like(self.im_front[:,:,0])
        self.gc_mask2 = np.zeros_like(self.im_front[:,:,0])
        self.RECTANGLE = (0,0,self.im_front.shape[1]-1,self.im_front.shape[0]-1)
    else:
      if event==cv2.EVENT_MOUSEMOVE:
        self.im_front = self.im_tmp.copy()
        #self.im_tmp = self.im_front.copy()
        cv2.circle(self.im_front,(x,y),self.THICKNESS//2,COLOR_SET[self.CLIC_TYPE],-1)

      # Draw lines
      if event==cv2.EVENT_LBUTTONDOWN:
        self.LBUTTON_DOWN = True
        self.LINEP[0] = (x,y)
        self.im_front = self.im_tmp.copy()
        #self.im_tmp = self.im_front.copy()
      elif event==cv2.EVENT_MOUSEMOVE and self.LBUTTON_DOWN:
        self.LINEP[1] = (x,y)
        self.im_front = self.im_tmp.copy()
        cv2.line(self.im_front,self.LINEP[0],self.LINEP[1],color=COLOR_SET[self.CLIC_TYPE],
          thickness=self.THICKNESS)
      elif event==cv2.EVENT_LBUTTONUP and self.LBUTTON_DOWN:
        self.LBUTTON_DOWN = False
        self.im_tmp = self.im_front.copy()
        self.LINEP[1] = (x,y)
        cv2.line(self.im_front,self.LINEP[0],self.LINEP[1],color=COLOR_SET[self.CLIC_TYPE],
          thickness=self.THICKNESS)
        
        mask = np.zeros_like(self.im_front[:,:,0],dtype=np.uint8)

        cv2.line(mask,self.LINEP[0],self.LINEP[1],color=WHITE,thickness=self.THICKNESS)
        self.gc_mask[mask!=0] = MASK_SET[self.CLIC_TYPE]

if __name__=='__main__':
  gc = GrabCut()
  gc.load_image('a.JPG')
  gc.runme()