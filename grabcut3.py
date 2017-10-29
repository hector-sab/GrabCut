#!/usr/bin/env python
'''
===============================================
                  GrabCut!
===============================================

Interactive Image Segmentation using GrabCut cv2 algorithm.

Press ESC to exit the Matting


How to use it:

from grabcut import GrabCut
#Create a GrabCut object
gc = GrabCut()
gc.load_image('avengers.jpg')
gc.matte()
'''
# TODO: Block the creation of more than one rectangles
# TODO: Fix the reset function

import numpy as np
import cv2

BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED, 'val' : 2}

class GrabCut(object):
  """
  How to Use it:

  from grabcut import GrabCut
  # Create a GrabCut object
  gc = GrabCut()
  # Load the image you want to matte
  gc.load_image('avengers.jpg')
  # Start process of matting
  gc.matte()


  How it works:

  Two windows will appear. In one of theme the image will show up, and in the other will be black.
  With left-clic draw a rectangle over the subject to be matted. Press 'n' to matte...

  Key '0' - To select areas of sure background
  Key '1' - To select areas of sure foreground
  Key '2' - To select areas of probable background
  Key '3' - To select areas of probable foreground

  Key 'n' - To update the segmentation
  Key 'r' - To reset the setup
  Key 's' - To save the results
  """

  def __init__(self):
    self.img = None
    self.img2 = None
    self.img3 = None
    self.mask = None
    self.mask2 = None
    self.output = None

    self.out_dir = './'
    self.mask_name = 'mask'
    self.img_name = 'img'

    self.ixy = (0,0)
    self.rectangle_flag = False # Drawing dynamic rectangle
    self.rectangle = (0,0,1,1) # Structure of the rectangle (x,y,w,h)
    self.rectangle2 = (0,0,1,1) 
    self.rect_or_mask = 100
    self.rect_over = False

    self.drawing_flag = False
    self.saved_flag = False
    self.value = DRAW_FG # Drawing initialized to FG


    self.bgdmodel = np.zeros((1,65),dtype=np.float64)
    self.fgdmodel = np.zeros((1,65),dtype=np.float64)

    self.exit = False
    self.THICKNESS = 3 # Brush thickness

    self.crop_roi = True # Crop the roi to be matted

  def load_image(self, file):
    """ 
    Load the image to be matted
    """
    self.img = cv2.imread(file) # Front image to be displayed
    self.img2 = self.img.copy() # End rgb image
    self.img3 = self.img.copy() # Full original image

  def output_params(self, out_dir=None, mask_name=None, img_name=None):
    """
    out_dir: directory where it's going to be stored
    mask_name: name for the mask image to be saved
    img_name: name for the cropped image to be saved
    """
    if out_dir!=None:
      print('Output directory changed')
      self.out_dir = out_dir
    if mask_name!=None:
      print('Mask output name changed')
      self.mask_name = mask_name
    if img_name!=None:
      print('Image output name changed')
      self.img_name = img_name

  def pts2rectangle(self,p1,p2):
    """
    Returns the 'rectangle' parameter used in the grabCut()
    function (x,y,w,h)

    p1: (x1,y1)
    p2: (x2,y2)
    """
    if p1[0]<p2[0]:
      x_l = p1[0]; x_r = p2[0]
    else:
      x_l = p2[0]; x_r = p1[0]

    if p1[1]<p2[1]:
      y_t = p1[1]; y_b = p2[1]
    else:
      y_t = p2[1]; y_b = p1[1]
    w = abs(x_l-x_r)
    h = abs(y_t-y_b)

    return((x_l,y_t,w,h))

  def onmouse(self, event, x, y, flags, params):
    # Draw dynamic rectangle
    if self.rect_over==False:
      if event==cv2.EVENT_RBUTTONDOWN:
        self.rectangle_flag = True
        self.ixy = (x,y)
      elif event==cv2.EVENT_MOUSEMOVE and self.rectangle_flag==True:
        self.img = self.img2.copy()
        cv2.rectangle(self.img, self.ixy, (x,y), BLUE, 2)
        self.rectangle = self.pts2rectangle(self.ixy,(x,y))
        self.rect_or_mask = 0
      elif event==cv2.EVENT_RBUTTONUP:
        print('ixy: ',self.ixy,' xy: ',x,' ',y)
        self.rectangle_flag = False
        self.rect_over = True
        cv2.rectangle(self.img, self.ixy, (x,y), BLUE, 2)
        #if CROP_FLAG==False:
        self.rectangle = self.pts2rectangle(self.ixy,(x,y))
        self.rect_or_mask = 0

        ### Crop
        if self.crop_roi:
          print(self.rectangle)
          self.img = self.img2[self.rectangle[1]:self.rectangle[1]+self.rectangle[3],
                      self.rectangle[0]:self.rectangle[0]+self.rectangle[2]]
          self.img2 = self.img.copy()

        self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        self.output = np.zeros_like(self.img, dtype=np.uint8)
        self.rectangle = (0,0,self.rectangle[2],self.rectangle[3])
        self.rectangle2 = (1,1,self.rectangle[2]-1,self.rectangle[3]-1)
        ###

    # Draw curves
    if event==cv2.EVENT_LBUTTONDOWN:
      if self.rect_over==False:
        print('First draw a rectangle')
      else:
        self.drawing_flag = True
        cv2.circle(self.img, (x,y), self.THICKNESS, self.value['color'], -1)
        cv2.circle(self.mask, (x,y), self.THICKNESS, self.value['val'], -1)
    elif event==cv2.EVENT_MOUSEMOVE and self.drawing_flag==True:
      cv2.circle(self.img, (x,y), self.THICKNESS, self.value['color'], -1)
      cv2.circle(self.mask, (x,y), self.THICKNESS, self.value['val'], -1)
    elif event==cv2.EVENT_LBUTTONUP and self.drawing_flag==True:
      self.drawing_flag = False
      cv2.circle(self.img, (x,y), self.THICKNESS, self.value['color'], -1)
      cv2.circle(self.mask, (x,y), self.THICKNESS, self.value['val'], -1)

  def matte(self):
    """
    Starts the matting process
    """
    cv2.namedWindow('input', cv2.WINDOW_NORMAL)
    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('input', self.onmouse)

    while(True):
      cv2.imshow('input', self.img)

      if self.rect_over:
        cv2.imshow('output', self.output)

      k = 0xFF & cv2.waitKey(1)

      if k==27: # End the loop
        cv2.destroyAllWindows()
        break
      elif k==ord('e'):
        self.exit = True
        cv2.destroyAllWindows()
        break
      elif k==ord('0'):
        print('Marking background')
        self.value = DRAW_BG
      elif k==ord('1'):
        print('Marking foreground')
        self.value = DRAW_FG
      elif k==ord('2'):
        print('Marking probable background')
        self.value = DRAW_PR_BG
      elif k==ord('3'):
        print('Marking probable foreground')
        self.value = DRAW_PR_FG
      elif k==ord('n'):
        if self.rect_over:
          print('Starting Matting!')
          if self.rect_or_mask==0:
            cv2.grabCut(self.img2, self.mask, self.rectangle2,
              self.bgdmodel, self.fgdmodel, 5, cv2.GC_INIT_WITH_RECT)
            self.rect_or_mask = 1
          elif self.rect_or_mask==1:
            cv2.grabCut(self.img2, self.mask, self.rectangle2,
                        self.bgdmodel, self.fgdmodel, 5, cv2.GC_INIT_WITH_MASK)

          self.mask2 = np.where((self.mask==1) + (self.mask==3),255,0).astype('uint8')
          self.output = cv2.bitwise_and(self.img2,self.img2,mask=self.mask2)
          print('Matting has finished!')
        else:
          print('First draw a rectangle!')
      elif k==ord('s'):
        print('Saving image...')
        self.saved_flag = True
        bar = np.zeros((self.img.shape[0],5,3),dtype=np.uint8)
        res = np.hstack((self.img2,bar,self.img,bar,self.output))

        kernel = np.ones((10,10),np.uint8)
        self.mask2 = cv2.erode(self.mask2,kernel,iterations = 1)
        self.mask2 = cv2.dilate(self.mask2, kernel,iterations = 1)

        cv2.imwrite(self.out_dir+self.mask_name+'.png',self.mask2)
        cv2.imwrite(self.out_dir+self.img_name+'.png',self.img2)
        print('Image saved...')
      elif k==ord('r'):
        print('Reseting all parameters....')
        self.img = self.img3.copy()
        self.img2 = self.img3.copy()
        self.mask = None
        self.mask2 = None
        self.output = None

        self.ixy = (0,0)
        self.rectangle_flag = False # Drawing dynamic rectangle
        self.rectangle = (0,0,1,1) # Structure of the rectangle (x,y,w,h)
        self.rectangle2 = (0,0,1,1) 
        self.rect_or_mask = 100
        self.rect_over = False

        self.drawing_flag = False
        self.saved_flag = False
        self.value = DRAW_FG # Drawing initialized to FG


        self.bgdmodel = np.zeros((1,65),dtype=np.float64)
        self.fgdmodel = np.zeros((1,65),dtype=np.float64)

        self.THICKNESS = 3 # Brush thickness
      elif k==ord('+'):
        """
        Increase brush size
        """
        self.THICKNESS += 1
      elif k==ord('-'):
        """
        Decrease brush size
        """
        self.THICKNESS -= 1
        if self.THICKNESS==0:
          self.THICKNESS = 1
      elif k==ord('c'):
        """
        Activate or deactivate roi crop
        """
        if self.rect_over==False
          if self.crop_roi:
            self.crop_roi = False
          else:
            self.crop_roi = True

          print('Crop ROI ',self.crop_roi)
        else:
          print('Rectangle already drawn. \nReset("r") the parameters to use this function