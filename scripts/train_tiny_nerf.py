import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from torch import nn, optim

from nerfdiff.nerf.tiny_nerf import TinyNeRF

def load_data(device):
    data_f = "/usr/dataset/cars.npz"
    # data_f = "/usr/dataset/lego.npz"
    data = np.load(data_f)

    images = data["images"] / 255
    img_size = images.shape[1]
    #TODO here image_size is used for x only
    xs = torch.arange(img_size) - (img_size / 2 - 0.5)
    ys = torch.arange(img_size) - (img_size / 2 - 0.5)
    (xs, ys) = torch.meshgrid(xs, -ys, indexing="xy")
    focal = float(data["focal"])

    pixel_coords = torch.stack([xs, ys, torch.full_like(xs, -focal)], dim=-1)
    camera_coords = pixel_coords / focal

    init_ds = camera_coords.to(device)
    init_o = torch.Tensor(np.array([0, 0, float(data["camera_distance"])])).to(device)

    #TODO no point in returning img_size here!!! ???
    return (images, data["poses"], init_ds, init_o, img_size)


def set_up_test_data(images, device, poses, init_ds, init_o):
    test_idx = 0
    test_img = torch.Tensor(images[test_idx]).to(device)
    test_R = torch.Tensor(poses[test_idx, :3, :3]).to(device)
    test_ds = torch.einsum("ij,hwj->hwi", test_R, init_ds)
    test_os = (test_R @ init_o).expand(test_ds.shape)
    train_idxs = np.arange(len(images)) != test_idx

    return (test_ds, test_os, test_img, train_idxs)


if __name__ == "__main__":
    #TODO add argparse arguments for: device, dataset path, seed etc.. (that needs change)
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda:0"
    nerf = TinyNeRF(device)

    lr = 5e-3
    optimizer = optim.Adam(nerf.F_c.parameters(), lr=lr)
    criterion = nn.MSELoss()

    (images, poses, init_ds, init_o, test_img) = load_data(device)
    (test_ds, test_os, test_img, train_idxs) = set_up_test_data(
        images, device, poses, init_ds, init_o
    )
    images = torch.Tensor(images[train_idxs])
    poses = torch.Tensor(poses[train_idxs])

    psnrs = []
    iternums = []
    num_iters = 20000
    display_every = 1000
    nerf.F_c.train()
    for i in tqdm(range(num_iters)):
        target_img_idx = np.random.randint(images.shape[0])
        target_pose = poses[target_img_idx].to(device)
        R = target_pose[:3, :3]

        ds = torch.einsum("ij,hwj->hwi", R, init_ds)
        # os = (R @ init_o).expand(ds.shape) #TODO Why t not used?
        os = target_pose[:3,-1].expand(ds.shape)

        C_rs_c = nerf(ds, os)
        loss = criterion(C_rs_c, images[target_img_idx].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % display_every == 0:
            torch.save(nerf.F_c.state_dict(), f"/usr/nerfdiff/nerf/model_{i}.pth")

            print("Saving")
            nerf.F_c.eval()
            with torch.no_grad():
                C_rs_c = nerf(test_ds, test_os)

            loss = criterion(C_rs_c, test_img)
            print(f"Loss: {loss.item()}")
            psnr = -10.0 * torch.log10(loss)

            psnrs.append(psnr.item())
            iternums.append(i)

            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.imshow(C_rs_c.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.subplot(122)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")
            # plt.show()
            plt.savefig(f"out2/{i}.jpg")
            plt.close()

            nerf.F_c.train()

    
    torch.save(nerf.F_c.state_dict(), f"/usr/nerfdiff/nerf/model_last.pth")
    print("Done!")