#!/usr/bin/env python
"""
"""

"""
TODO: Fix left button click with cv2.WINDOW_KEEPRATIO
"""
import numpy as np
import cv2

BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PBG
GREEN = [0,255,0]       # FG
BLACK = [0,0,0]         # BG
WHITE = [255,255,255]   # PFG

COLOR_SET = {'bg':BLACK,'fg':GREEN,'pbg':RED,'pfg':WHITE}
MASK_SET = {'bg':0,'fg':1,'pbg':2,'pfg':3}
class GrabCut:
  def __init__(self):
    self.FLAG_loaded_image = False
    self.FLAG_crop_rectangle = False # Indicates if the selected rectangle
                                     # should be cropped or just used as
                                     # zoom for the ROI
    self.FLAG_LBUTTON_CLIC = False # Indicates if the left button has
                                    # been pressed
    self.FLAG_RBUTTON_CLIC = False
    self.FLAG_RECTANLE = False # Indicate if we are working in a rectangle
    self.FLAG_rclic = 'fg' # bg, fg, pbg, pfg= 'bg'
    self.FLAG_GC_RECT = True
    self.lxy = [(0,0),(0,0)]
    self.rxy = [(0,0),(0,0)]
    self.rectangle = (0,0,1,1)
    self.THICKNESS = 4
    self.out_name = 'output'
    self.init()
  
  def init(self):
    self.im = None # Original images. Not touched.
    self.im1 = None # Original image to work with.
    self.im2 = None # Images where we will select fg/bg
    self.im2_ = None # temporal im2 for drawings
    self.im3 = None # Matted image
    self.mask = None
    self.gc_mask = None

  def load_image(self,path):
    """
    Indicates which image will be edited
    """
    self.im = cv2.imread(path)
    self.reset_image(self.im)
    self.FLAG_loaded_image = True

  def reset_image(self,im):
    """
    Reset the image
    """
    self.rectangle = (0,0,im.shape[0]-1,im.shape[1]-1)
    self.im1 = im.copy()
    self.im2 = im.copy()
    self.im3 = im.copy()
    self.mask = np.zeros(shape=(self.im2.shape[0],self.im2.shape[1]),
                         dtype=np.uint8)
    self.gc_mask = np.zeros(shape=(self.im2.shape[0],self.im2.shape[1]),
                         dtype=np.uint8)
    
    self.bgdmodel = np.zeros((1,65),dtype=np.float64)
    self.fgdmodel = np.zeros((1,65),dtype=np.float64)
  
  def set_save_path(self,path):
    """
    Indicates where the edited image will be saved
    """
    self.out_name = path

  def save_image(self):
    """
    Saves the images
    """
    cv2.imwrite(self.out_name+'_mask'+'.png',self.gc_mask)
    cv2.imwrite(self.out_name+'.png',self.im3)
  
  def onmouse(self,event,x,y,flags,params):
    """
    Controls keys and mouse inputs
    """

    # Select ROI 
    if event==cv2.EVENT_RBUTTONDOWN:
      self.rxy[0] = (x,y)
      self.FLAG_RBUTTON_CLIC = True
      self.FLAG_RECTANLE = True
      self.im2_ = self.im2.copy()
    elif event==cv2.EVENT_MOUSEMOVE and self.FLAG_RBUTTON_CLIC:
      self.rxy[1] = (x,y)
      self.im2 = self.im2_.copy()
      cv2.rectangle(self.im2, self.rxy[0], self.rxy[1], BLUE, 2)
    elif event==cv2.EVENT_RBUTTONUP and self.FLAG_RBUTTON_CLIC:
      self.FLAG_RBUTTON_CLIC = False

      x1 = self.rxy[0][0]; x2 = self.rxy[1][0]
      y1 = self.rxy[0][1]; y2 = self.rxy[1][1]

      # Ensures order
      if x1>x2:
        tmp = x1; x1 = x2; x2 = tmp
      if y1>y2:
        tmp = y1; y1 = y2; y2 = tmp

      im = self.im[y1:y2,x1:x2]
      self.reset_image(im)
    

    # Select fg/bg
    if event==cv2.EVENT_LBUTTONDOWN:
      self.lxy[0] = (x,y)
      self.FLAG_LBUTTON_CLIC = True
      self.im2_ = self.im2.copy()
    elif event==cv2.EVENT_MOUSEMOVE and self.FLAG_LBUTTON_CLIC:
      self.lxy[1] = (x,y)
      self.im2 = self.im2_.copy()
      cv2.line(self.im2,self.lxy[0],self.lxy[1],
               color=COLOR_SET[self.FLAG_rclic],thickness=self.THICKNESS)
    elif event==cv2.EVENT_LBUTTONUP and self.FLAG_LBUTTON_CLIC:
      self.lxy[1] = (x,y)
      #cv2.line(self.im2,self.lxy[0],self.lxy[1],
      #         color=COLOR_SET[self.FLAG_rclic],thickness=self.THICKNESS)
      
      mask = np.zeros(shape=(self.mask.shape[0],self.mask.shape[1]),
                       dtype=np.uint8)

      cv2.line(mask,self.lxy[0],self.lxy[1],color=WHITE,thickness=self.THICKNESS)
      self.mask[mask!=0] = MASK_SET[self.FLAG_rclic]

      self.FLAG_LBUTTON_CLIC = False
  
  def matte(self):
    """
    Starts the matting process
    """
    if self.FLAG_loaded_image:
      while(True):
        OUT_WIND_NAME = 'Output'
        INP_WIND_NAME = 'Input' + ' Crop '+str(self.FLAG_crop_rectangle)

        #WIN_TYPE = cv2.WINDOW_KEEPRATIO
        WIN_TYPE = cv2.WINDOW_GUI_NORMAL
        cv2.namedWindow(OUT_WIND_NAME,WIN_TYPE)
        cv2.namedWindow(INP_WIND_NAME, WIN_TYPE)
        cv2.setMouseCallback(INP_WIND_NAME, self.onmouse)

        cv2.imshow(INP_WIND_NAME,self.im2)
        cv2.imshow(OUT_WIND_NAME,self.im3)

        cv2.namedWindow('mask',WIN_TYPE)
        cv2.imshow('mask',self.mask*80)

        cv2.namedWindow('gc mask',WIN_TYPE)
        cv2.imshow('gc mask',self.gc_mask*80)

        k = 0xFF & cv2.waitKey(1)

        if k==27: # If ESC, end loop
          cv2.destroyAllWindows()
          break
        elif k==ord('c'):
          """
          Indicates if the selected rectangle should be a crop or a
          zoom of the ROI
          """
          cv2.destroyAllWindows()
          self.FLAG_crop_rectangle = not self.FLAG_crop_rectangle
        elif k==ord('1'): # bg
          self.FLAG_rclic = 'bg'
        elif k==ord('2'): # fg
          self.FLAG_rclic = 'fg'
        elif k==ord('3'): # pbg
          self.FLAG_rclic = 'pbg'
        elif k==ord('4'): # pfg
          self.FLAG_rclic = 'pfg'
        
        elif k==ord('+'): # bigger line thickness
          if self.THICKNESS<20:
            self.THICKNESS += 1
        elif k==ord('-'): # smaller line thickness
          if self.THICKNESS>1:
            self.THICKNESS -= 1
            
        elif k==ord('m'): # Start matting
          print('Starting matting!')
          # self.mask,self.bgdmodel,self.fgdmodel = 
          if self.FLAG_GC_RECT:
            cv2.grabCut(self.im3, self.mask, self.rectangle,
              self.bgdmodel, self.fgdmodel, 5, cv2.GC_INIT_WITH_RECT)
            self.FLAG_GC_RECT = not self.FLAG_GC_RECT
          else:
            mask,bgM,fgM = cv2.grabCut(self.im,self.mask,self.rectangle,self.bgdmodel,
                                  self.fgdmodel,5,cv2.GC_INIT_WITH_MASK)
            self.mask,self.bgdmodel,self.fgdmodel = (mask,bgM,fgM)

          self.gc_mask = np.where((self.mask==3)|(self.mask==1),255,0).astype('uint8')

          """
          # Smooth the edges
          self.gc_mask = cv2.pyrUp(self.gc_mask)
          for _ in range(5):
            self.gc_mask = cv2.medianBlur(self.gc_mask,7)
          
          self.gc_mask = cv2.pyrDown(self.gc_mask)
          """
          self.im3 = cv2.bitwise_and(self.im1,self.im1,mask=self.gc_mask)

          print('All done!')
        elif k==ord('s'): # Save images
          self.save_image()
          print('Images saved!')

    
    else:
      msg = "Image wasn't loaded. Please load an image first."
      print(msg)


if __name__=='__main__':
  grab = GrabCut()
  grab.load_image('a.JPG')
  grab.matte()
  

  
