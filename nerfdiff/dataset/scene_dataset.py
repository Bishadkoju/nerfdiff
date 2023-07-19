import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class SceneDataset(Dataset):
    def __init__(self, path, filename, N=None):

        print("\nScene Dataset\n--------------")
        print(f"Using: {path}/{filename}")
        self.path = path

        #! Get data
        self.names, self.qctcs = self.get_names_qctcs(f"{path}/{filename}")
        available_data_points = len(self.names)
        print(f"Unique Data Points: {available_data_points}")

        #! Get indexes
        if N is None:
            self.index = np.linspace(0,available_data_points-1, available_data_points).astype(int)
        else:
            self.index = np.random.randint(0,available_data_points, size=N)

        print(f"Total Data: {self.__len__()}\n")

        #! Image pre-process (as used in ImageNet classification)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def get_names_qctcs(self, poses_path):
        names = []
        qctcs = []

        with open(poses_path) as file:
            data = file.readlines()

            for line in data:
                line_split = line[:-1].split(' ')
                name = line_split[0]
                qctc = []
                for i in range(1,8):
                    qctc.append(float(line_split[i]))

                names.append(name)
                qctcs.append(qctc)

        return names, torch.as_tensor(qctcs, dtype=torch.float32)


    def __len__(self):
        return len(self.index)


    def __getitem__(self, idx):
        index = self.index[idx]

        #! Get Name
        name = self.names[index]
        # print(f"Getting image: {name}")

        #! Get image
        img = Image.open(f"{self.path}/images/{name}")
        img_tensor = self.preprocess(img)

        #! Get qctc
        qctc = self.qctcs[index]

        # print(img_tensor.shape)
        # print(torch.min(img_tensor))
        # print(torch.max(img_tensor))

        #! Return
        return {
            'name': name,
            'image': img_tensor,
            'qctc': qctc
        }


if __name__=='__main__':
    sd = SceneDataset(N=1000)
    print(sd.__len__())

    #! Get a data
    data = sd[2]
    print(data['name'])
    print(data['image'].shape)
    print(data['qctc'])