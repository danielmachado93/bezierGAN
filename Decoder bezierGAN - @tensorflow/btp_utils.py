import numpy as np
import csv
from glob import glob
import os
from itertools import islice
import scipy.misc
import imageio
from six.moves import xrange

dataset = "./data/bezier_dataset"
image_file_format = "*.jpg"
image_paths = glob(os.path.join(dataset, image_file_format))

def loadbatch(batch_size = 10, start_index = 0, paths = image_paths):
    n_total = len(image_paths)
    # get pixel space dimension
    img = scipy.misc.imread(paths[0], flatten=True).astype(np.float32)
    pixelSpace_w = int(img.shape[0])
    pixelSpace_h = int(img.shape[1])

    ps = np.zeros(shape=[batch_size, pixelSpace_w, pixelSpace_h], dtype=np.float32)
    bs = np.zeros(shape=[batch_size,8],dtype=np.float32)
    batch_paths = paths[start_index:start_index+batch_size]
    count = 0
    for path in batch_paths:
        # Pixel Space --> JPG
        img = scipy.misc.imread(path, flatten=True).astype(np.float32)
        ps[count] = img
        # Bezier Space --> File Name
        basename_with_ext = os.path.basename(path)
        basename_without_ext = os.path.splitext(basename_with_ext)[0]
        bezierSpace_string = basename_without_ext.split("_")[1]
        bezierSpace_list = bezierSpace_string.split(",")
        bezierSpace_array = np.array(bezierSpace_list).astype(dtype=np.float32)
        bs[count] = bezierSpace_array
        count = count + 1

    # PixelSpace --> [BATCH, W, H,1]
    ps = np.reshape(ps, newshape=[batch_size, pixelSpace_w, pixelSpace_h, 1])
    return bs, ps

def btp_normalize_ps(tensor):
    return 1.0-(tensor/255.)
def btp_normalize_bs(tensor, ps_out):
    return (tensor/ps_out)

#bs,ps = loadbatch(batch_size=1,start_index=10000,bezier_space_csv_reader=reader,pixel_space_paths=image_paths)