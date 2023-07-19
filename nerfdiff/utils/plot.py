import numpy as np
import torch
from nerfdiff.utils.read_write_model import qvec2rotmat

def get_identity_cam(scale=1):
    """
    Gets an identity camera plot

    Parameters
    ----------
    scale: int
        Scale of the camera plot

    Returns
    -------
    cam_plot: np.ndarray
        (11, 3) Sequence of points to make camera plot
    """
    f = 10
    unit_cam = np.array([
        [0,0,0],
        [3,-2,f],
        [3,2,f],
        [-3,2,f],
        [-3,-2,f],
        [0,-4,f]
    ])

    seq = np.array([3,4,1,2,0,1,5,4,0,3,2])
    draw_cam = unit_cam[seq]

    return draw_cam*scale


def get_cam_plot(qvec, tvec, unit_cam):
    """
    Gets camera plot, pose defined by given quaternion and translation

    Parameters
    ----------
    qvec: np.ndarray
        (4,) Quaternion of Camera Pose
    tvec: np.ndarray
        (3,) Translation of Camera Pose
    unit_cam: np.ndarray
        (11,3) Ideal camera sequence

    Returns
    -------
    cam: np.ndarray
        (11,3) Sequence of corrdinates to draw given camera pose
    """
    R = qvec2rotmat(qvec)
    return (R@unit_cam.T + tvec.reshape(3,1)).T


def plot_cam_poses(poses: np.ndarray, ax, scale=0.02, alpha=1, color=None):
    unit_cam = get_identity_cam(scale)
    for pos in poses:
        cam = get_cam_plot(pos[:4], pos[4:], unit_cam)
        if color is None:
            ax.plot(cam[:,0], cam[:,1], cam[:,2], alpha=alpha)
        else:
            ax.plot(cam[:,0], cam[:,1], cam[:,2], alpha=alpha, color=color)