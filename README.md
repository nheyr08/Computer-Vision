# Image Stitching
Image Stitching algorithm with multi-panoramas, gain compensation, and multi-band blending.
#### part of the computer vision assignment @NYCU spring 2024
This program takes in multiple images 2+ and returns a unified stitched image, 
## Usage
```
git clone https://github.com/nheyr08/Image-Stiching.git
cd Image-Stiching
```
Please make sure your Python env supports both cv2 and numpy.
The following snippet creates a new environment (img_stich) with the tested version of python and installs opencv and numpy in parallel ~
```
conda create --name img_stitch python=3.8
conda activate img_stitch
conda install -c conda-forge opencv & conda install numpy
```

Now simply run: 
``sh run.sh ``

the results will be saved under Photos/Base/result and Photos/Challenge/result
Respectively.

For more options, see the command line help:

    python main.py --help
Feel free to add your pictures to test the program!  
For any issues welcome to open a PR and Don't forget to star 👍 if you find this repository helpful~

## Reference

The implementation is strongly based on the 2007 paper **Automatic Panoramic Image Stitching using Invariant Features** by Matthew Brown and David G. Lowe : <http://matthewalunbrown.com/papers/ijcv2007.pdf>
