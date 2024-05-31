import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize, RandomRotation,RandomVerticalFlip
import random
import torchvision.transforms.functional as TF

class dataset_label(data.Dataset):
  def __init__(self, opts, setname, input_dim):
    self.dataroot = opts.dataroot
    images = os.listdir(os.path.join(self.dataroot, opts.phase + setname))
    self.img = [os.path.join(self.dataroot, opts.phase + setname, x) for x in images]
    self.size = len(self.img)
    self.input_dim = input_dim
    self.crop_size = opts.crop_size
    self.phase = opts.phase

    # setup image transformation
    transforms = []
    # transforms.append(CenterCrop(opts.crop_size))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=(0.5), std=(0.5)))
    self.transforms = Compose(transforms)
    print('%s: %d images'%(setname, self.size))
    return

  def __getitem__(self, index):
    image = self.load_img(self.img[index], self.input_dim)
    filename = os.path.basename(self.img[index])
    label = self.load_img(os.path.join(self.dataroot, 'label', filename), 3)
    if self.crop_size is not None:
            i, j, h, w = self._get_random_crop_params(image, (self.crop_size, self.crop_size))
            image = TF.crop(image, i, j, h, w)
            label = TF.crop(label, i, j, h, w)
    image = self.trans_img(image, self.input_dim)
    label = self.trans_img(label, 3)
    return image, label, os.path.splitext(filename)[0]

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    return img
  
  def trans_img(self,img,input_dim):
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img
  
  def _get_random_crop_params(self, img, crop_size):
        w, h = img.size
        crop_h, crop_w = crop_size
        i = random.randint(0, h - crop_h)
        j = random.randint(0, w - crop_w)
        return i, j, crop_h, crop_w

  def __len__(self):
    return self.size
  
class dataset_output(data.Dataset):
  def __init__(self, opts, setname, input_dim):
    self.dataroot = opts.dataroot
    images = os.listdir(os.path.join(self.dataroot, opts.phase + setname))
    self.img = [os.path.join(self.dataroot, opts.phase + setname, x) for x in images]
    self.size = len(self.img)
    self.input_dim = input_dim

    # setup image transformation
    transforms = []
    # transforms.append(CenterCrop(opts.crop_size))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=(0.5), std=(0.5)))
    self.transforms = Compose(transforms)
    print('%s: %d images'%(setname, self.size))
    return

  def __getitem__(self, index):
    data = self.load_img(self.img[index], self.input_dim)
    filename = os.path.basename(self.img[index])
    return data, os.path.splitext(filename)[0]

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.size
  
class dataset_single(data.Dataset):
  def __init__(self, opts, setname, input_dim):
    self.dataroot = opts.dataroot
    images = os.listdir(os.path.join(self.dataroot, opts.phase + setname))
    self.img = [os.path.join(self.dataroot, opts.phase + setname, x) for x in images]
    self.size = len(self.img)
    self.input_dim = input_dim

    # setup image transformation
    transforms = []
    # transforms.append(CenterCrop(opts.crop_size))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=(0.5), std=(0.5)))
    self.transforms = Compose(transforms)
    print('%s: %d images'%(setname, self.size))
    return

  def __getitem__(self, index):
    data = self.load_img(self.img[index], self.input_dim)
    return data

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = img.resize((1024,1024))
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.size


class dataset_unpair(data.Dataset):
  def __init__(self, opts):
    self.dataroot = opts.dataroot

    # A
    images_A = os.listdir(os.path.join(self.dataroot, opts.phase + 'A'))
    self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]

    # B
    images_B = os.listdir(os.path.join(self.dataroot, opts.phase + 'B'))
    self.B = [os.path.join(self.dataroot, opts.phase + 'B', x) for x in images_B]

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b

    self.crop_size = opts.crop_size
    transforms = []
    
    transforms.append(RandomHorizontalFlip())
    transforms.append(RandomVerticalFlip())
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=(0.5), std=(0.5)))

    self.transforms = Compose(transforms)
    print('A: %d, B: %d images'%(self.A_size, self.B_size))
    return

  def __getitem__(self, index):
    if self.dataset_size == self.A_size:
      data_A = self.load_img(self.A[index], self.input_dim_A)
      data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
    else:
      data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A)
      data_B = self.load_img(self.B[index], self.input_dim_B)
    return data_A, data_B

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    pre_trans = Compose([
      RandomRotation(45),
      Resize((1024, 1024), Image.BICUBIC),
      CenterCrop(self.crop_size + random.randint(-50,100)),
      Resize((self.crop_size, self.crop_size), Image.BICUBIC)
    ])
    img = pre_trans(img)
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size
