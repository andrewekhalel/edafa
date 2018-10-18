# tta_predictor

tta_predictor is a wrapper that implements Test Time Augmentations (TTA) to images for computer vision tasks like: segmentation. TTAs guarantees significant gain in most of the computer vision tasks. All you need is inherit the base class, implement virtual methods according to your model and decide which augmentations do you want.
