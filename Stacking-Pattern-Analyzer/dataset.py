import os
import torch
import random
import numpy as np
from torch.utils.data import  Dataset
from PIL import Image
from PIL import ImageFilter
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop,Compose,Resize,RandomCrop
from torchvision.transforms import InterpolationMode 

class ImageDataset(Dataset):
    def __init__(self, 
                 images_path='./data',
                 crop_size=512,
                 task = '',
                 augment=True,
                 is_train=True):
        self.images_path = images_path
        self.task = task
        self.noise_path = '../noises'
        self.crop_size = crop_size
        self.augment = augment
        self.is_train = is_train
        self.image_size = 1024
        self.images = sorted([item for item in os.listdir(images_path) if ((item.endswith('.png')) 
                                                                           & (float(item.split('_')[0]) > -3.2) 
                                                                           & (float(item.split('_')[0]) < 3.2) 
                                                                           & (float(item.split('_')[1]) < 2.8) 
                                                                           & (float(item.split('_')[1]) >= 0.0)
                                                                           )])
        self.noises = sorted([item for item in os.listdir(self.noise_path) if item.endswith('.png')])
        image_num = len(self.images)
        train_cut = int(0.99*image_num)
        self.mod_num = 1
        indices = random.sample(range(image_num),train_cut)
        if self.is_train:
            self.images = sorted([self.images[i] for i in indices])
        else:
            remain_indices = set(range(image_num)) - set(indices)
            self.images = sorted([self.images[i] for i in remain_indices])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.images_path, self.images[index])).convert('RGB')
        label = self.images[index//self.mod_num].split('_')
        
        # Data augmentation
        if self.augment:
            image = self._augment_data(image)
        
        r = random.uniform(-90, 90)
        image = self._random_offset_and_rotation(image,r)
        if self.task == 'slip':
            label = torch.tensor([float(label[0]), float(label[1])])
        elif self.task == '3layer-slip':
            if(float(label[1])<float(label[3]) or float(label[2])<float(label[4])):
                label = torch.tensor([float(label[3]), float(label[4]),float(label[1]), float(label[2])])
            else:
                label = torch.tensor([float(label[1]), float(label[2]),float(label[3]), float(label[4])])
        elif self.task == 'twist':
            label = torch.tensor([float(label[2])/60.])


        # Center crop
        if self.crop_size is not None:
            crop_size = random.randint(self.crop_size-100,720)
            crop = Compose([CenterCrop(crop_size),Resize((self.crop_size, self.crop_size), interpolation=InterpolationMode.BICUBIC)])
            image = crop(image)

        # Convert to tensor
        image = TF.to_tensor(image)
        image = self._random_noise(image)
        image = self._random_noise_2(image)
        image = self._random_crop(image)
        image = torch.where(image>1.0, 1.0, image)
        
        return image, label

    def _augment_data(self, image):
        # Random horizontal flip
        # if random.random() > 0.5:
        #     image = TF.hflip(image)

        # Random contrast adjustment
        contrast_factor = random.uniform(0.7, 1.5)
        image = TF.adjust_contrast(image, contrast_factor)

        # Random guass
        if random.random() > 0.8:
            radius = random.uniform(0.5,4)
            gaussian_blur = ImageFilter.GaussianBlur(radius=radius)
            image = image.filter(gaussian_blur) 

        # norm
        image = np.array(image) 
        image = (image - image.min()) / (image.max() - image.min())  
        image = Image.fromarray((image * 255).astype(np.uint8))

        # Random brightnes
        brightness_factor = random.uniform(0.8, 1.2)
        image = TF.adjust_brightness(image, brightness_factor)

        # Random crop
        r_crop = RandomCrop(int(0.9*1024))
        image = r_crop(image)

        return image

    def _random_offset_and_rotation(self, img, r):

        rotated_img = TF.rotate(img, r) 

        return rotated_img

    def _random_crop(self, img):
        if random.uniform(0,1)>0.4:
            return img
        
        x = random.randint(0, int(self.crop_size/4*3))
        y = random.randint(0, int(self.crop_size/4*3))
        dx = random.randint(int(self.crop_size/8), int(self.crop_size))
        dy = random.randint(int(self.crop_size/8), dx)

        padding = random.uniform(0,0.2)

        img[:, x:x+dx, y:y+dy] = padding

        return img
    
    def _random_noise(self, img):
        if random.uniform(0,1)>0.4:
            return img
        index = random.randint(0,len(self.noise_path)-1)
        crop = Compose([RandomCrop(random.randint(720,1024)),Resize((self.crop_size, self.crop_size), interpolation=InterpolationMode.BICUBIC)])
        noise = Image.open(os.path.join(self.noise_path, self.noises[index])).convert('L').convert('RGB')

        contrast_factor = random.uniform(1.0, 3.0)
        noise = TF.adjust_contrast(noise, contrast_factor)

        radius = random.uniform(0.5,4)
        gaussian_blur = ImageFilter.GaussianBlur(radius=radius)
        noise = noise.filter(gaussian_blur) 
        
        noise = crop(noise)
        noise = TF.to_tensor(noise)

        img = img + noise * random.uniform(0,1.3)

        return img
    
    def _random_noise_2(self, img):
        if random.uniform(0,1)>0.0:
            return img

        gauss_std = random.uniform(0.01, 0.5)
        contrast = random.uniform(0.8,5.0)
        
        img_noise = img.clone()
        
        for _ in range(1):
            gaussion_noise = torch.normal(0,gauss_std,(512,512))
            img_noise = img_noise + gaussion_noise

        img_noise = (img_noise - img_noise.min()) / (img_noise.max() - img_noise.min()) 
        img_noise = Image.fromarray((np.asarray(img_noise)[0] * 255).astype(np.uint8))
        contrast_factor = contrast
        img_noise = TF.adjust_contrast(img_noise, contrast_factor)
        img_noise = TF.to_tensor(img_noise).repeat(3,1,1)

        img_noise = (img_noise - img_noise.min()) / (img_noise.max() - img_noise.min())  
        
        return img_noise