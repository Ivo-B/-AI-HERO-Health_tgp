import numpy as np 
from PIL import Image
import os
from collections import Counter
import pandas as pd

# Get the file paths

path= '/hkfs/work/workspace/scratch/im9193-H5/data/imgs/'
im_files = os.listdir(path)

# imagine we only want to load PNG files (or JPEG or whatever...)
EXTENSION = '.png'

def get_characteristics(path, f):
    image = Image.open(path + f)
    img = np.array(image.getdata()).reshape(image.size)
    name = f
    shape = img.shape
    max = np.max(img)
    min = np.min(img)
    mean = np.mean(img)
    return name, shape, max, min, mean

# Load using matplotlib
#images_plt_shape = [plt.imread(path + f).shape for f in im_files if f.endswith(EXTENSION)]
# convert your lists into a numpy array of size (N, H, W, C)

images_characteristics = [get_characteristics(path, f)for f in im_files if f.endswith(EXTENSION)]
#[print(image.shape) for image in  images_plt]
#images = np.array(images_plt)

df = pd.DataFrame(images_characteristics, columns =['name','shape', 'max', 'min', "mean"])

df.to_csv("metadata.tsv", sep ="\t", index=False)
#counter = Counter(images_plt_shape)

#print(counter)