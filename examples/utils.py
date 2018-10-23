from sklearn.metrics import jaccard_similarity_score as iou
import numpy as np

def color_map(N=256, normalized=False):
    """
    Python implementation of the color map function for the PASCAL VOC data set
    source: https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
    """
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def multi_iou(a,b):
    jk = 0.0
    vals = np.unique(a)[1:]
    for v in vals:
        ac = a.copy()
        ac[ac != v] = 0
        bc = b.copy()
        bc[bc != v] = 0
        jk += iou(ac,bc)
    return jk/len(vals)