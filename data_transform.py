from PIL import Image
import torch


#path="/hkfs/work/workspace/scratch/im9193-H5/AI-HERO-Health_gp/readme_imgs/210.png"

path="/hkfs/work/workspace/scratch/im9193-H5/AI-HERO-Health_gp/readme_imgs/219.png"

#path="/hkfs/work/workspace/scratch/im9193-H5/AI-HERO-Health_gp/readme_imgs/774.png"

from torchvision.transforms import Resize

res_layer = Resize((320,320))

image = Image.open(path)
image = torch.ByteTensor(image.getdata()).reshape((1,image.size[0],image.size[1]))

image_resized = res_layer(image)

PIL_image = Image.fromarray(image_resized.reshape((320,320)).numpy())

PIL_image.save("/hkfs/work/workspace/scratch/im9193-H5/AI-HERO-Health_gp/readme_imgs/219_resized_to_320.png")