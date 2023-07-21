import torch
import numpy as np
import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

#! nerfloc imports
from nerfdiff.dataset.scene_dataset import SceneDataset
from nerfdiff.model.diff_model import DiffModel
from nerfdiff.diffusion.noise_scheduler import NoiseScheduler
import nerfdiff.utils.base as utils
import nerfdiff.utils.plot as plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    parser.add_argument("--dataset_path", type=str, default="/usr/dataset/dtu_scene6")
    parser.add_argument("--train_poses", type=str, default="poses.txt")
    parser.add_argument("--test_poses", type=str, default="poses.txt")
    # parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--hidden_layers", type=int, default=5)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])
    parser.add_argument("--save_images_step", type=int, default=1)
    config = parser.parse_args()

    #! Define Dataset
    train_dataset = SceneDataset(path=config.dataset_path, filename=config.train_poses)
    test_dataset = SceneDataset(path=config.dataset_path, filename=config.test_poses)


    #! Define Model
    model = DiffModel(
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        emb_size=config.embedding_size,
        time_emb=config.time_embedding,
        input_emb=config.input_embedding).to(config.device)

    model_path = f"data/diff_exps/{config.experiment_name}/model.pth"
    model.load_state_dict(torch.load(model_path))
    print(f"Model Loaded: {model_path}")
    model.eval()

    #! Define Noise Scheduler
    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule)

    #! Define Timesteps: 0 to 50
    timesteps = list(range(len(noise_scheduler)))[::-1]

    #! Define Inputs and Outputs
    op_dim = 7

    label_named = np.linspace(0,49, 49)

    unit_cam = plot.get_identity_cam(0.02)

    #! For each Label:
    # for label, exp_op in zip(labels, expected_op):
    for count, dataa in enumerate(test_dataset):
        label = dataa['image']
        exp_op = dataa['qctc']
        print("Data point:", dataa['name'], exp_op)
        # print("Labels and Expected Outputs are:", label.shape, exp_op)
        #! Get a random data sample
        sample = torch.randn(config.eval_batch_size, op_dim)
        # sample = normalize_qt(sample)
        initial_sample = sample.numpy()

        fake_y = np.random.rand(*initial_sample.shape)

        errors = []
        frames = []
        # correct_filter = []

        #! For each timestep:
        for i, t in enumerate(tqdm(timesteps)):
            t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long()
            l = []
            for _ in range(config.eval_batch_size):
                l.append(label)
            l = torch.from_numpy(np.stack(l)).float()
            # print(l.shape)
            # l = torch.from_numpy(np.repeat(label, config.eval_batch_size)).long()

            with torch.no_grad():
                sample = sample.to(config.device)
                t = t.to(config.device)
                l = l.to(config.device)

                # print('input shapes:')
                # print(sample.shape, t.shape, l.shape)

                residual = model(sample, t, l)

            sample = noise_scheduler.step(residual.cpu(), t[0].cpu(), sample.cpu())
            # sample = normalize_qt(sample)

            result = sample.numpy()

            #! Calculate Error
            error = utils.diff_qctc(result, exp_op.numpy())
            # print(mse, rot, trans)

            # error_thres = 0.1
            # correct_result = error < error_thres 
            # correct_filter.append(correct_result)

            #! Accumulate everything
            frames.append(result)
            errors.append(error)

        #! ---------------------------------------------------------------------

        # label 
        print(f"Saving reverse diffusion images for label {label_named[count]} ...")
        imgdir = f"data/diff_exps/{config.experiment_name}/rev_diffusion/label_{label_named[count]}"
        os.makedirs(imgdir, exist_ok=True)

        # Convert to Numpy
        frames = np.stack(frames)
        errors = np.array(errors)
        # correct_filter = np.stack(correct_filter)

        print(frames.shape)
        print(errors.shape)

        # print("Correct Filter: ")
        # print(correct_filter.shape)

        
        xmin, xmax = -6, 6
        ymin, ymax = -6, 6

        print("Writing ...")
        for i, frame in tqdm(enumerate(frames)):

            # print("This is frame")
            # print(frame.shape)
            # input()

            fig = plt.figure(figsize=(8, 8))

            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=-70, azim=90, roll=0)
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
            ax.set_zlim(-1,1)
            ax.view_init(elev=-70, azim=-90, roll=0)

            #! Normalize q
            frame = utils.normalize_qt(frame)


            #! Plot all 100 camera poses in this time step
            for frm in frame:
                cam = plot.get_cam_plot(frm[:4], frm[4:], unit_cam)
                ax.plot(cam[:,0], cam[:,1], cam[:,2], alpha=0.3)

            #! Plot Test GT pose
            exp_op_np = exp_op.numpy()
            gt_cam = plot.get_cam_plot(exp_op_np[:4], exp_op_np[4:], unit_cam)
            ax.plot(gt_cam[:,0], gt_cam[:,1], gt_cam[:,2], linewidth=4, color='green')
            
            #! Plot Train Dataset poses
            for td in train_dataset:
                t_qctc = td['qctc'].numpy()
                cam = plot.get_cam_plot(t_qctc[:4], t_qctc[4:], unit_cam)
                ax.plot(cam[:,0], cam[:,1], cam[:,2], alpha=1, color='red')

      

            #! Write errors to the image
            e = errors[i]
            trans_percent = (e[4]/2)*100
            text = f"MSE: {e[0]:.4f}  |  YPR: {e[1]:.2f}, {e[2]:.2f}, {e[3]:.2f}  |  TRANS: {e[4]:.3f} ({trans_percent:.1f}%)"
            fig.text(x=0.5, y=0, s=text, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)

            # fig.text(0.5, 0.5, "hahaha")

            plt.savefig(f"{imgdir}/3d_{i:04}.jpg")
            plt.close()   
            #! -------------------------------
            # n_bins = 30

            # fig, (hist_ax, conv_ax) = plt.subplots(2, 2, tight_layout=True, figsize=(10,5), gridspec_kw={'height_ratios': [2, 1]})
            # fig.suptitle(f"Input Label: {label_named[count]}")


            # #! Hist Plot 1
            # hist_ax[0].set_xlim(-3,3)
            # hist_ax[0].set_ylim(0, config.eval_batch_size)

            # hist_ax[0].hist(frame[:,4], bins=np.linspace(-3,3,100))

            # #! Hist Plot 2
            # hist_ax[1].set_xlim(-3,3)
            # hist_ax[1].set_ylim(0, config.eval_batch_size)

            # hist_ax[1].hist(frame[:,5], bins=np.linspace(-3,3,100))



            # #! For each output dim:
            # for dim in range(2):
            #     dim_data = initial_sample[:, dim+4]
            #     dim_fake_y = fake_y[:, dim+4]

            #     dim_fltr = correct_filter[i, :, dim+4]

            #     correct_points = dim_data[dim_fltr]
            #     incorrect_points = dim_data[np.logical_not(dim_fltr)]

            #     correct_rand_y = dim_fake_y[dim_fltr]
            #     incorrect_rand_y = dim_fake_y[np.logical_not(dim_fltr)]

            #     conv_ax[dim].set_xlim(-3,3)
            #     conv_ax[dim].set_ylim(0,1)


            #     conv_ax[dim].set_yticks([])

            #     conv_ax[dim].scatter(correct_points, correct_rand_y, color='green', alpha=0.5)
            #     conv_ax[dim].scatter(incorrect_points, incorrect_rand_y, color='red', alpha=0.5)

            #     conv_ax[dim].axvline(x = exp_op[dim+4], color = 'b', linewidth=4 , label = 'axvline - full height')

            # # plt.ylim(ymin, ymax)
            # plt.savefig(f"{imgdir}/{i:04}.png")
            # plt.close()            


        print("Done !")
        # input()