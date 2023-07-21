import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

from torch import nn, optim

from nerfdiff.nerf.tiny_nerf import TinyNeRF


def load_data(device):
    # data_f = "/usr/dataset/cars.npz"
    data_f = "/usr/dataset/lego.npz"
    data = np.load(data_f)
    print(data)
    # print(data['camera_distance'])
    # print(data['poses'])

    # images = data["images"] / 255
    images = data['images']
    img_size = images.shape[1]
    #TODO here image_size is used for x only
    xs = torch.arange(img_size) - (img_size / 2 - 0.5)
    ys = torch.arange(img_size) - (img_size / 2 - 0.5)
    (xs, ys) = torch.meshgrid(xs, -ys, indexing="xy")
    focal = float(data["focal"])

    pixel_coords = torch.stack([xs, ys, torch.full_like(xs, -focal)], dim=-1)
    camera_coords = pixel_coords / focal

    init_ds = camera_coords.to(device)
    # init_o = torch.Tensor(np.array([0, 0, float(data["camera_distance"])])).to(device)
    # print(init_o)
    print(init_ds.shape)
    

    #TODO no point in returning img_size here!!! ???
    return (images, data["poses"], init_ds)


def set_up_test_data(images, device, poses, init_ds):
    test_idx = 0
    test_img = torch.Tensor(images[test_idx]).to(device)
    test_R = torch.Tensor(poses[test_idx, :3, :3]).to(device)
    test_ds = torch.einsum("ij,hwj->hwi", test_R, init_ds)
    # test_os = (test_R @ init_o).expand(test_ds.shape)
    test_os = torch.Tensor(poses[test_idx, :3, 3]).to(device).expand(test_ds.shape)
    train_idxs = np.arange(len(images)) != test_idx

    return (test_ds, test_os, test_img, train_idxs)

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


if __name__ == "__main__":
    #TODO add argparse arguments for: device, dataset path, seed etc.. (that needs change)

    parser = ArgumentParser(description='Train Tiny NeRF')
    parser.add_argument('--seed', type=int )
    parser.add_argument('--use-cpu', action='store_true')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scene-name', type=str)
    parser.add_argument('--experiment-name', type=str, default=get_timestamp())
    parser.add_argument('--num-iterations', type=int, default=20000)
    parser.add_argument('--steps-per-save', type=int, default=1000)
    parser.add_argument('--steps-per-eval', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=5e-3)
    args = parser.parse_args()
    print(args)

    seed = args.seed
    if(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
    device = 'cpu' if args.use_cpu else 'cuda:0'
    data_dir = args.data_dir
    output_dir = args.output_dir
    scene = args.scene_name or data_dir.name
    experiment_name = args.experiment_name
    num_iterations = args.num_iterations
    steps_per_save = args.steps_per_save
    steps_per_eval = args.steps_per_eval
    lr= args.lr
    assert steps_per_save < num_iterations, f'Steps per save should not be greater than number of iterations ({steps_per_save} > {num_iterations})'
    assert steps_per_eval < num_iterations, f'Steps per evaluation should not be greater than number of iterations ({steps_per_eval} > {num_iterations})'


    experiment_save_path = output_dir / scene / experiment_name
    model_save_path = experiment_save_path / 'models'
    log_save_path = experiment_save_path / 'logs'

    model_save_path.mkdir(parents=True, exist_ok=True)
    log_save_path.mkdir(parents=True, exist_ok=True)


    nerf = TinyNeRF(device)
    optimizer = optim.Adam(nerf.F_c.parameters(), lr=lr)
    criterion = nn.MSELoss()

    (images, poses, init_ds) = load_data(device)
    (test_ds, test_os, test_img, train_idxs) = set_up_test_data(
        images, device, poses, init_ds
    )
    images = torch.Tensor(images[train_idxs])
    poses = torch.Tensor(poses[train_idxs])

    psnrs = []
    iternums = []
    nerf.F_c.train()
    for i in tqdm(range(num_iterations)):
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


        if i % steps_per_save == 0:
            torch.save(nerf.F_c.state_dict(), model_save_path / f"model_{i}.pth")

            print("Saving")
        if i % steps_per_eval == 0:
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
            plt.savefig(log_save_path / f"{i}.jpg")
            plt.close()

            nerf.F_c.train()

    
    torch.save(nerf.F_c.state_dict(),model_save_path /  f"model_last.pth")
    print("Done!")