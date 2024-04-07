from torch.utils.data import Dataset
import os
import torchvision.transforms as T
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_root, image_size):
        name_list = os.listdir(data_root)
        self.img_paths = [os.path.join(data_root, name) for name in name_list]

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        path = self.img_paths[index]
        img = Image.open(path)
        return self.transform(img)
