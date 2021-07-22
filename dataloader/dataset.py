import os
import glob
import cv2

from PIL import Image
from typing import Tuple
from torch.utils.data import Dataset

class DeepClusteringDataset(Dataset):
    def __init__(self,
                data_dir: str,
                transform = None,
                image_size: Tuple[int, int] = (224, 224)):
        
        self.transform = transform
        self.all_samples = glob.glob(os.path.join(data_dir, '*.jpg'))
        self.num_samples = len(self.all_samples)
        print('Loaded {} samples'.format(self.num_samples))
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.all_samples[idx]
        image = cv2.imread(sample)
        image = Image.fromarray(image)
        
        X = self.transform(image)
        return X

if __name__ == '__main__':
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    data_dir = '/home/santapo/OnlineLab/deep_clustering/data/no_label/images_mr'
    train_transforms = transforms.Compose([
                    transforms.Resize((224,224), interpolation = InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ])
    dataset = DeepClusteringDataset(data_dir=data_dir, transform=train_transforms)

    sample = dataset[1]

    print(sample)
    print(type(sample))
    print(sample.shape)
    sample = sample.numpy().transpose(1,2,0)
    cv2.imshow('', sample)
    cv2.waitKey(0)