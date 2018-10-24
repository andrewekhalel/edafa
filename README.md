
# edafa

edafa is a simple wrapper that implements Test Time Augmentations (TTA) on images for computer vision problems like: segmentation, super-resolution, Pansharpening, etc. TTAs guarantees better results in most of the tasks.

### Installation
PyPI pacakage will be avaialble soon

### Getting started
The easiest way to get up and running is to follow this [notebook](https://github.com/andrewekhalel/tta_predictor/blob/master/examples/pascal_voc.ipynb) showing an example on using TTA Predictor to improve segmentation score on PASCAL VOC dataset.

### How to use TTA Predictor
The whole process can be done in 4 steps:
1.  Import `BasePredictor` abstract class 
```python
from tta_predictor import BasePredictor
```
2. Inherit `BasePredictor` to your own class and implement the main 3 functions 
	* `preprocess(self,img)` :  Implement preprocessing needed after reading image from disk (e.g. normalization, .. )
	* `postprocess(self,pred)` :  Implement postprocessing needed after predicting an image (e.g. clipping, argmax, .. )
	* `predict_patches(self,patches)` :  The function where your model takes image patches and return their prediction

```python
class myPredictor(BasePredictor):
    def __init__(self,model,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = model
        
    def preprocess(self,img):
        return normalize(img)

    def postprocess(self,pred):
        return clipping(pred)

    def predict_patches(self,patches):
        return self.model.predict(patches)
```
3. Create an instance of you class
```python
p = myPredictor(model,patch_size,model_output_channels,conf_file_path)
```
4.  Call `predict_dir()` to run the prediction process 
``` python
p.predict_dir(in_dir,out_dir,overlap=0,extension='.png')
```
### Configuration file
Configuration file is a json file containing two pieces of information
1. Augmentations to apply (**augs**). Supported augmentations:
	* **NO** : No augmentation
	* **ROT90** : Rotate 90 degrees
	* **ROT180** : Rotate 180 degrees
	* **ROT270** : Rotate 270 degrees
	* **FLIP_UD** : Flip upside-down
	* **FLIP_LR** : Flip left-right
2. Combination of the results (**mean**). Supported mean types:
	* **ARITHMETIC** : Arithmetic mean
	* **GEOMETRIC** : Geometric mean

Example of a conf file
```json
{
"augs":["NO",
"FLIP_UD",
"FLIP_LR"],
"mean":"ARITHMETIC"
}
```
