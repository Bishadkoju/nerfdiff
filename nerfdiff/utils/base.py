import numpy as np
import torch
from nerfdiff.utils.read_write_model import qvec2rotmat
from scipy.spatial.transform import Rotation as R


def normalize_qt(qt):
    """
    Parameters
    ----------
    qt: np.ndarray
        (N, 7)

    Returns
    -------
    qt: np.ndarray
        qt with normalized q (N,7)
    """
    q = qt[:, :4]
    q_norm = np.linalg.norm(q, axis=1)
    q = q/q_norm.reshape(-1,1)
    qt[:, :4] = q
    return qt 


def get_names_qctcs(poses_path):
    """
    Get image names and poses from the poses.txt file

    Parameters
    ----------
    poses_path: str
        Path to pose file

    Returns
    -------
    names: list
        List of image names
    poses: torch.Tensor
        (N,7) Poses
    """
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


def diff_qctc(qctc, qctc_gt):
    """
    Find mean difference between a set of qctc and a GT

    Parameters
    ---------
    qctc: np.ndarray
        (N,7) q might be un-normalized
    qctc2_gt: np.ndarray
        (7,) q is normalized, since it is GT

    Returns
    -------
    mse: float
        Mean Square error between two qctc
    yaw,pitch,roll: float
        Angle errors
    trans: float
        Translational error
    """
    # print(qctc.shape, qctc_gt.shape)
    #! Normalize qc (just to be sure)
    qctc = normalize_qt(qctc)

    #! MSE
    mse = np.sum(np.square(qctc-qctc_gt), axis=1)/qctc_gt.shape[0] # Error in each sample of N
    mean_mse = np.mean(mse)

    #! YPR error
    R_gt = qvec2rotmat(qctc_gt[:4])

    eul_diff = []
    for qt in qctc:
        r = qvec2rotmat(qt[:4])
        r_diff = R_gt.T@r
        r_diff_scipy = R.from_matrix(r_diff)
        eul = r_diff_scipy.as_euler('zxy', degrees=True)
        eul_diff.append(eul)
    eul_diff = np.array(eul_diff)
    eul_diff_mean = np.mean(np.abs(eul_diff), axis=0)

    #! Trans error
    trans_error = np.sqrt(np.sum(np.square(qctc[:,4:]-qctc_gt[4:]), axis=1))
    trans_error_mean = np.mean(trans_error)

    mean_mse = mean_mse.reshape(-1)
    trans_error_mean = trans_error_mean.reshape(-1)
    all = np.concatenate([mean_mse, eul_diff_mean, trans_error_mean])

    return all