import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from torch import nn, optim

from nerfdiff.nerf.tiny_nerf import TinyNeRF
from nerfdiff.dataset.scene_dataset import SceneDataset
from nerfdiff.utils.base import get_timestamp

if __name__ == "__main__":

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
    lr = args.lr
    path = args.data_dir
    assert steps_per_save < num_iterations, f'Steps per save should not be greater than number of iterations ({steps_per_save} > {num_iterations})'
    assert steps_per_eval < num_iterations, f'Steps per evaluation should not be greater than number of iterations ({steps_per_eval} > {num_iterations})'

    experiment_save_path = output_dir / scene / experiment_name
    model_save_path = experiment_save_path / 'models'
    log_save_path = experiment_save_path / 'logs'
    model_save_path.mkdir(parents=True, exist_ok=True)
    log_save_path.mkdir(parents=True, exist_ok=True)

    nerf = TinyNeRF(device=device)
    dataset = SceneDataset(path=path, device=device)
    optimizer = optim.Adam(nerf.F_c.parameters(), lr=lr)
    loss_function = nn.MSELoss()

    test_image, test_ray_directions, test_ray_origins = dataset.get_test_data()

    psnrs = []
    iternums = []
    nerf.F_c.train()

    for i in tqdm(range(num_iterations)):
        image, ray_directions, ray_origins = dataset.get_next_training_data()
        rendered_image = nerf(ray_directions, ray_origins)
        loss = loss_function(rendered_image, image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % steps_per_save == 0:
            torch.save(nerf.F_c.state_dict(), model_save_path / f"model_{i}.pth")
            print("Saving")
        if i % steps_per_eval == 0:
            nerf.F_c.eval()
            with torch.no_grad():
                rendered_image = nerf(test_ray_directions, test_ray_origins)

            loss = loss_function(rendered_image, test_image)
            print(f"Loss: {loss.item()}")
            psnr = -10.0 * torch.log10(loss)

            psnrs.append(psnr.item())
            iternums.append(i)

            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.imshow(rendered_image.detach().cpu().numpy())
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