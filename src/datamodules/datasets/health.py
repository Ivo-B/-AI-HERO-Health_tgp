from torch.utils.data import Dataset
from torchvision.transforms import transforms
import pandas as pd
import torch


class HealthDataset(Dataset):
    def __init__(self, label_path, img_path, transform=None, target_transform=None):
        self.df = pd.read_csv(label_path,header=0)
        self.labels = torch.FloatTensor([0 if label == "negative" else 1 for label in self.df["label"]]).squeeze()
        self.names = self.df["image"].tolist()
        self.transform = transform
        self.target_transform = target_transform
        self.image_path=img_path

    def load_image(self, idx:int) -> torch.Tensor:

        from PIL import Image

        trans = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
        image = Image.open(self.image_path + self.names[idx])
        return trans(torch.FloatTensor(image.getdata()).reshape((1,image.size[0],image.size[1])))

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image = self.load_image(idx)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

if __name__ == "__main__":
    transform = transforms.Compose(
            [transforms.Resize((320,320)),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
             ]
        )
    dataset = HealthDataset(label_path = "/hkfs/work/workspace/scratch/im9193-H5/data/train.csv", img_path = "/hkfs/work/workspace/scratch/im9193-H5/data/imgs/",transform=transform)
    example = dataset.__getitem__(0)

    print(example)
