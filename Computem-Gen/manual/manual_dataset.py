import torch
import random
import numpy as np
import logging
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# from img2struct.scale import init_scale
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize, RandomRotation,RandomVerticalFlip
import cv2

def plt_img(reachable_points, img_size, save_path=None):
    grid = np.zeros((img_size, img_size))
    radius = 1
    for point in reachable_points:
        x, y = point
        x_min, x_max = int(x - radius), int(x + radius) + 1
        y_min, y_max = int(y - radius), int(y + radius) + 1
        grid[x_min:x_max, y_min:y_max] = 1
    if save_path:
        plt.imsave(save_path, grid, cmap="gray")
    return grid

def resize(img, new_size):
    image = Image.fromarray(img)
    resized_image = image.resize(new_size, Image.ANTIALIAS)
    resized_image_np_array = np.array(resized_image)
    return resized_image_np_array

def parallelogram_coordinates(a, b, theta1):
    A = [0, 0]
    B = [a, 0]
    C = [b * math.cos(math.radians(theta1)), b * math.sin(math.radians(theta1))]
    D = [a + b * math.cos(math.radians(theta1)), b * math.sin(math.radians(theta1))]
    return np.array([A, B, C, D])

def get_repeat_vec(c, d, theta2):
    E = [c, 0]
    F = [d * math.cos(math.radians(theta2)), d * math.sin(math.radians(theta2))]
    return np.array([E, F])

def reachable_points_pre(n=50):
    x = np.linspace(-n, n, 2 * n + 1)
    y = np.linspace(-n, n, 2 * n + 1)
    xx, yy = np.meshgrid(x, y)
    reachable_points = np.vstack((xx.flatten(), yy.flatten())).T
    return reachable_points

def reachable_points(single_cell, repeat_vec, reachable_points, img_size):
    """
    single_cell shape[4, 2]
    repeat_vec shape[2, 2]
    reachable_points shape[n, 2]
    img_size int
    """
    new_points = single_cell[:, None, :] + reachable_points[None, :, :] @ repeat_vec
    new_points = new_points.reshape(-1, 2)
    new_points = new_points[
        (new_points[:, 0] >= 0)
        & (new_points[:, 0] < img_size)
        & (new_points[:, 1] >= 0)
        & (new_points[:, 1] < img_size)
    ]
    return new_points

def reorder(arr):
    sorted_index = arr[:, 0].argsort()
    arr = arr[sorted_index]

    sorted_index = arr[arr[:, 0] == arr[0, 0], 1].argsort()
    arr[arr[:, 0] == arr[0, 0]] = arr[arr[:, 0] == arr[0, 0]][sorted_index]

    sorted_index = arr[arr[:, 0] == arr[-1, 0], 1].argsort()
    arr[arr[:, 0] == arr[-1, 0]] = arr[arr[:, 0] == arr[-1, 0]][sorted_index]
    return arr

def get_rotation_vectors(scale_range, layer=1):
    degree = random.random() * 180
    angle = np.radians(degree)
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    scale_ = random.random()
    scale = scale_ * (scale_range[1] - scale_range[0]) + scale_range[0]
    shift = np.random.rand(layer, 2)
    shift = reorder(shift)
    shift = shift.reshape(1, -1)
    return rotation_matrix, shift, scale

def get_rotated_vectors(xy, rotation_matrix, shift, scale):
    xy = np.matmul(xy, rotation_matrix)
    xy = xy * scale + shift
    return xy

def add_noise(xy, noise_offset, noise=0.15):
    xy_noise = ((np.random.rand(*xy.shape) - 0.5) * (noise_offset * noise)).astype(
        np.int64
    )
    xy += xy_noise
    return xy, xy_noise

def get_min_distance(n, points):
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    coords = np.stack((x, y), axis=-1)
    diff = coords[..., np.newaxis, :] - points[np.newaxis, np.newaxis, :]
    distances = np.sqrt(np.sum(diff**2, axis=-1))
    min_distances = distances.min(axis=-1)
    return min_distances

def get_gray_image(image_size, xy_dic, bright_diff, damp, save_path=None, range=[2, 9]):
    dist = get_min_distance(image_size, xy_dic)
    img = np.random.rand(*dist.shape)
    out_mask = dist >= range[1]
    in_mask = dist < range[0]
    img[out_mask] = img[out_mask] * 255 - bright_diff
    img[np.logical_and(np.logical_not(in_mask), np.logical_not(out_mask))] = img[np.logical_and(np.logical_not(in_mask), np.logical_not(out_mask))]*255 - \
                                                                             (dist[np.logical_and(np.logical_not(in_mask), np.logical_not(out_mask))]-2)*damp
    # img[np.logical_and(np.logical_not(in_mask), np.logical_not(out_mask))] *= 180
    img[in_mask] = img[in_mask] * 200 + bright_diff
    img[img < 0] = 0
    img[img > 255] = 255
    img = np.rot90(np.flip(img, axis=1))
    if save_path:
        plt.imsave(save_path, img, cmap="gray")
    return img / 255

def get_repeat(
    img_size,
    noise_offset,
    reachable_points_pre,
    single_cell,
    repeat_vec,
    shift_scale,
    bright_diff,
    damp,
    rotation_shift_info,
):
    rotation_matrix, shift, scale = rotation_shift_info
    repeat_vec = get_rotated_vectors(
        repeat_vec, rotation_matrix=rotation_matrix, shift=0, scale=scale
    )
    shift_ = shift * scale * shift_scale
    single_cell = get_rotated_vectors(
        single_cell,
        rotation_matrix=rotation_matrix,
        shift=shift_ + np.array([[img_size / 2, img_size / 2]]),
        scale=scale,
    )
    all_points = reachable_points(
        single_cell, repeat_vec, reachable_points_pre, img_size
    )
    all_points_noise = all_points
    all_points_noise, _ = add_noise(all_points, noise_offset)
    # img = plt_img(all_points_noise, img_size, save_path=None)
    img = get_gray_image(img_size, all_points_noise,bright_diff, damp,save_path=None)
    return {
        "img": img,
    }

def return_pic(rotation_shift_info, layer):
    r = reachable_points_pre(n=50)
    a, b, theta1, c, d, theta2 = (2.828, 2.815, 57.68, 6.42, 6.585, 60.2)
    single_cell = parallelogram_coordinates(a, b, theta1)
    repeat_vec = get_repeat_vec(c, d, theta2)
    shift_scale = np.array([[repeat_vec[0, 0], repeat_vec[1, 1]]])
    rotation_matrix, shift_list, scale = rotation_shift_info
    bright_diff = random.randint(120,140)
    damp = random.randint(0,7)
    img_list = []
    blur = 5
    adw = 1
    sigma = 0.8
    for i in range(layer):
        img = get_repeat(
            img_size=256,
            noise_offset=7,
            reachable_points_pre=r,
            single_cell=single_cell,
            repeat_vec=repeat_vec,
            shift_scale=shift_scale,
            bright_diff=bright_diff,
            damp = damp,
            rotation_shift_info=(
                rotation_matrix,
                shift_list[:, 2 * i : 2 * i + 2],
                scale,
            ),
        )["img"]
        img_list.append(img)
    img_list = np.stack(img_list, axis=0).sum(axis=0)
    img_list = cv2.GaussianBlur(img_list, (blur, blur), sigma)
    img_list = cv2.addWeighted(img_list, adw, img_list, 0, 0.5)
    img_list = cv2.normalize(img_list, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img_list

class IMGDataset(Dataset):
    def __init__(
        self,
        image_size=256,
        resize_image=1,
        layer=2,
        noise_offset=7,
        cell_info=(2.828, 2.815, 57.68, 6.42, 6.585, 60.2),
        scale_range=[12, 16],
        length=100000,
    ):
        super().__init__()
        self.image_size = image_size
        self.resize_image = resize_image
        # self.resize_image = 1024
        self.layer = layer
        self.noise_offset = noise_offset
        self.cell_info = cell_info
        a, b, theta1, c, d, theta2 = cell_info
        self.single_cell = parallelogram_coordinates(a, b, theta1)
        self.repeat_vec = get_repeat_vec(c, d, theta2)
        self.shift_scale = np.array([[self.repeat_vec[0, 0], self.repeat_vec[1, 1]]])
        self.length = length
        self.reachable_points_pre = reachable_points_pre(n=50)
        self.scale_range = scale_range
        assert self.image_size >= 1
        assert self.resize_image >= 1
        assert self.layer >= 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # rotation_matrix, shift, scale, scale_
        (
            rotation_matrix,
            shift_list,
            scale,
        ) = get_rotation_vectors(self.scale_range, layer=self.layer)
        img_list = []
        # shift_list = np.array([[0.,0.,0.28,0.03]])
        bright_diff = random.randint(120,140)
        damp = random.randint(13,15)
        blur = random.randint(1, 5)*2 + 1
        adw = random.uniform(0.5,1.5)
        sigma = random.uniform(0.8,1.2)
        for i in range(self.layer):
            return_dict_tmp = get_repeat(
                self.image_size,
                self.noise_offset,
                reachable_points_pre=self.reachable_points_pre,
                single_cell=self.single_cell,
                repeat_vec=self.repeat_vec,
                shift_scale=self.shift_scale,
                bright_diff=bright_diff,
                damp=damp,
                rotation_shift_info=(
                    rotation_matrix,
                    shift_list[:, 2 * i : 2 * i + 2],
                    scale,
                ),
            )
            img_list.append(return_dict_tmp["img"])
        # img_list = np.stack(img_list, axis=0).sum(axis=0).clip(0, 1)
        img_list = np.stack(img_list, axis=0).sum(axis=0)
        for i in range(1):
            img_list = cv2.GaussianBlur(img_list, (blur, blur), sigma)
        img_list = cv2.addWeighted(img_list, adw, img_list, 0, 0.5)
        img_list = cv2.normalize(img_list, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return_dict = {
            "img": img_list,
            "rotation_matrix": rotation_matrix,
            "shift": shift_list,
            "scale": scale,
        }
        for k, v in return_dict.items():
            if isinstance(v, np.ndarray):
                return_dict[k] = torch.tensor(v).float()
        return return_dict


dataset = IMGDataset()
for i in range(5):
    a = dataset[0]
    plt.imsave(str(i)+'.png', a['img'], cmap="gray")
    print('rotation_matrix: ',a['rotation_matrix'])
    print('shift: ',a['shift'])
    print('scale: ',a['scale'])
