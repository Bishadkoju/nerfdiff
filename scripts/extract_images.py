import numpy as np
from pathlib import Path
from PIL import Image
from nerfdiff.utils.read_write_model import rotmat2qvec

"""
Simple script to extract data from cars.npz
"""

data_path = "/usr/dataset/cars.npz"
output_path = '/usr/dataset/cars'

data = np.load(data_path)
print("Available Keys:")
for val in data.keys():
    print(val)
print()

#! Extract images
images = data['images']
images_dir = f"{output_path}/images"
Path(images_dir).mkdir(parents=True, exist_ok=True)

for i, img in enumerate(images):
    pil_img = im = Image.fromarray(img)
    pil_img.save(f"{images_dir}/{i:04d}.jpg")

print(f"Images extracted to {images_dir}")

#! Extract poses
poses = data['poses']

qt = []
for pos in poses:
    R = pos[:3,:3]
    t = pos[:3,-1]
    q = rotmat2qvec(R)
    qt.append(np.concatenate((q,t)))
qt = np.array(qt)

#! Extract other data
with open(f"{output_path}/poses.txt", "w") as file:
    file.write(f"> focal:{data['focal']}\n")
    if 'camera_distance' in data.keys(): file.write(f"> camera_distance:{data['camera_distance']}\n") 

    for i,q in enumerate(qt):
        file.write(f"{i}.jpg {q[0]} {q[1]} {q[2]} {q[3]} {q[4]} {q[5]} {q[6]}\n")

print(f"Poses extracted to {output_path}/poses.txt")

print('Done :)')