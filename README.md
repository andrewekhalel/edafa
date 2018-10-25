
# Edafa

Edafa is a simple wrapper that implements Test Time Augmentations (TTA) on images for computer vision problems like: segmentation, classification, super-resolution, Pansharpening, etc. TTAs guarantees better results in most of the tasks.

### Installation
```
pip install edafa
```

### Getting started
The easiest way to get up and running is to follow example notebooks for [segmentation](https://github.com/andrewekhalel/edafa/blob/master/examples/seg_pascal_voc.ipynb) and [classification](https://github.com/andrewekhalel/edafa/blob/master/examples/class_imagenet.ipynb) showing effect of TTA on performance.

### How to use Edafa
The whole process can be done in 4 steps:
1.  Import Predictor class based on your task category (Segmentation or Classification)
```python
from edafa import SegPredictor
```
2. Inherit Predictor class and implement the main function 
	* `predict_patches(self,patches)` : where your model takes image patches and return prediction

```python
class myPredictor(SegPredictor):
    def __init__(self,model,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = model

    def predict_patches(self,patches):
        return self.model.predict(patches)
```
3. Create an instance of you class
```python
p = myPredictor(model,patch_size,model_output_channels,conf_file_path)
```
4.  Call `predict_images()` to run the prediction process 
```python
p.predict_images(images,overlap=0)
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
	* **ARITH** : Arithmetic mean
	* **GEO** : Geometric mean

Example of a conf file
```json
{
"augs":["NO",
"FLIP_UD",
"FLIP_LR"],
"mean":"ARITH"
}
```
