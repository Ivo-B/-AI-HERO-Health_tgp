from PIL import Image
import torch
from tqdm import tqdm
import logging
import os
import re

import numpy as np

if __name__ == '__main__':
    def recursive_walk(rootdir):
        for r, dirs, files in os.walk(rootdir):
            for f in files:
                yield os.path.join(r, f)

    #path="/hkfs/work/workspace/scratch/im9193-H5/AI-HERO-Health_gp/readme_imgs/210.png"
    #path="/hkfs/work/workspace/scratch/im9193-H5/AI-HERO-Health_gp/readme_imgs/774.png"
    from torchvision.transforms import Resize

    data_root = "/hkfs/work/workspace/scratch/im9193-H5/data"

    file_paths = []
    for file_path in recursive_walk(os.path.join(data_root, "imgs")):
        file_paths.append(file_path)

    #file_paths = ["/hkfs/work/workspace/scratch/im9193-H5/AI-HERO-Health_gp/readme_imgs/219.png"]
    res_layer = Resize((512, 512))
    for f in tqdm(file_paths):
        image = Image.open(f)
        image_resized = res_layer(image)
        img_np = np.array(image_resized)

        # clip
        p_low = np.percentile(img_np, 0.5)
        p_high = np.percentile(img_np, 99.9)
        img_np = np.clip(img_np, p_low, p_high)

        PIL_image = Image.fromarray(img_np)
        PIL_image = PIL_image.convert("L")
        PIL_image.save(f.replace("/data/", "/data_pre/"))


