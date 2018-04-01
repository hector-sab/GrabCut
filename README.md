# Interactive GrabCut
Interactive implementation of OpenCV GrabCut

## How to use it?

```python
from grabcut3 import GrabCut
#Create a GrabCut object
gc = GrabCut()
gc.load_image('avengers.jpg')
gc.matte()
```

## Options
Once you have executed the matte() method there are several options available:

```
r: reset the image and all grabcut parameters
n: run grabcut algorithm to matte the image
s: save the results
e: exit
0: mark background
1: mark foreground
2: mark probable background
3: mark probable foreground
+: increase size of brush
-: decrease size of brush
c: Crop ROI
```

You can also specify the name of the resulting image and mask, and its output directory

```python
gc.output_params(out_dir='Output/Directory/', mask_name='MaskName0', img_name='ImageName0')
```