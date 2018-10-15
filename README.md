# tf_predictor

tf_predictor is a wrapper that implements Test Time Augmentations (TTA) to images in Tensorflow. TTAs guarantees significant gain in most of the computer vision tasks. All you need is inherit the base class, pass your model and decide which augmentations do you want.