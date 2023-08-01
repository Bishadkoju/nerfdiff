import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from nerfdiff.utils.base import qvec2c2w


class SceneDataset(Dataset):
    def __init__(self, path, device='cpu', output_raybundle=False, filename='poses.txt', N=None):

        print("\nScene Dataset\n--------------")
        print(f"Using: {path}/{filename}")
        self.path = path

        #! Get data
        self.names, self.qctcs, self.metadata = self.get_names_qctcs_metadata(f"{path}/{filename}")
        self.device = device
        self.output_raybundle = output_raybundle
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
            # When input images are normalized, NeRF Training gets stuck
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Setup Initial Raybundle Directions
        image_width = int(self.metadata['w'])
        image_height = int(self.metadata['h'])
        xs = torch.arange(image_width) - (image_width / 2 - 0.5)
        ys = torch.arange(image_height) - (image_height / 2 - 0.5)
        (xs, ys) = torch.meshgrid(xs, -ys, indexing="xy")
        focal = float(self.metadata['fx'])

        pixel_coords = torch.stack([xs, ys, torch.full_like(xs, -focal)], dim=-1)
        camera_coords = pixel_coords / focal

        self.initial_ray_directions = camera_coords.to(device)  

    def get_names_qctcs_metadata(self, poses_path):
        names = []
        qctcs = []
        metadata = {}

        with open(poses_path) as file:
            data = file.readlines()

            for line in data:
                if line[0] == '>':
                    k,v = line[1:].split(':')
                    metadata[k.strip()]= v.strip()
                    continue
                line_split = line[:-1].split(' ')
                name = line_split[0]
                qctc = []
                for i in range(1,8):
                    qctc.append(float(line_split[i]))

                names.append(name)
                qctcs.append(qctc)

        return names, torch.as_tensor(qctcs, dtype=torch.float32), metadata
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        index = self.index[idx]

        #! Get Name
        name = self.names[index]
        # print(f"Getting image: {name}")

        #! Get image
        img = Image.open(f"{self.path}/images/{name}")
        img_tensor = self.preprocess(img).to(self.device)

        #! Get qctc
        qctc = self.qctcs[index]

        output = {
            'name': name,
            'image': img_tensor,
            'qctc': qctc,
        }

        if self.output_raybundle:
            #! Convert quaternion and translation vector to Camera to World Matrix
            camera2world_matrix = qvec2c2w(qctc).to(self.device)
            rotation_matrix = camera2world_matrix[:3, :3]

            #! Rotate the initial ray directions using the rotation matrix
            ray_directions = torch.einsum("ij,hwj->hwi", rotation_matrix, self.initial_ray_directions)
            ray_origins = camera2world_matrix[:3,-1].expand(ray_directions.shape)

            output['ray_directions'] = ray_directions
            output['ray_origins'] = ray_origins
        
        #! Return
        return output


if __name__=='__main__':
    sd = SceneDataset(path='/usr/dataset/cars', filename='poses.txt', N=1000)
    print(sd.__len__())

    #! Get a data
    data = sd[2]
    print(data['name'])
    print(data['image'].shape)
    print(data['qctc'])
    print(data['metadata'])