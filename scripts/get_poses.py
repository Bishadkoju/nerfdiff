import argparse
import numpy as np
from nerfdiff.utils.read_write_model import read_images_text, qvec2rotmat, rotmat2qvec
np.set_printoptions(suppress=True, precision=4)


def get_qctc_names(data):
    data_keys = list(data.keys())
    qctc = []
    image_names = []
    for k in data_keys:
        data_point = data[k]

        image_names.append(data_point.name)

        q = data_point.qvec
        t = data_point.tvec.reshape(-1,1)
        R = qvec2rotmat(q)

        Rc = R.T
        tc = -R.T@t

        qc = rotmat2qvec(Rc) #TODO Check if these rot2mat and mat2rot works fine !!
        qc = qc / np.linalg.norm(qc) # Normalize the quaternion

        out_data = np.concatenate((qc, tc.flatten()))

        qctc.append(out_data)

    qctc = np.array(qctc)

    #! Normalize t over the entire dataset
    x_min, x_max = np.min(qctc[:, 4]), np.max(qctc[:,4])
    y_min, y_max = np.min(qctc[:, 5]), np.max(qctc[:,5])
    z_min, z_max = np.min(qctc[:, 6]), np.max(qctc[:,6])

    qctc[:, 4] = (qctc[:, 4] - x_min) / (x_max - x_min)
    qctc[:, 5] = (qctc[:, 5] - y_min) / (y_max - y_min)
    qctc[:, 6] = (qctc[:, 6] - z_min) / (z_max - z_min)

    qctc[:, 4:] = (qctc[:, 4:] * 2) -1

    return image_names, qctc


if __name__=='__main__':

    #! Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help='Colmap format images.txt file')
    parser.add_argument("--output", type=str, required=True, help='Poses file')
    config = parser.parse_args()

    #! Read Input
    images = read_images_text(config.input)

    #! Convert
    names, qctcs = get_qctc_names(images)

    #! Output
    with open(config.output, "w") as out_file:
        for name,qctc in zip(names, qctcs):
            out_file.write(name)
            for i in range(7):
                out_file.write(f" {qctc[i]:.5f}")
            out_file.write('\n')

    print("Done!")