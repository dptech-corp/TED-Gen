import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.models as models
from model import StructNet

# Model parameter path
weight_path = '1.pth'
# Image path
image_path = '1.png'
# Task type
task = 'slip'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if task == 'slip':
    model = StructNet(output_dim=3)
elif task == 'twist':
    model = StructNet(output_dim=1)
elif task == '3layer-slip':
    model = StructNet(output_dim=4)
model.to(device)

model.load_state_dict(torch.load(weight_path))
model.eval()

image = Image.open(image_path).convert('RGB')
image = image.resize((1024,1024))
image = TF.hflip(image)
image_list = []

for i in range(16):
    crop_size = 512 + i * 10
    image_c = TF.crop(image, 0, 0, crop_size, crop_size)
    image_c = image_c.resize([512,512])
    data = TF.to_tensor(image_c)
    # test_data = data[i]
    image_list.append(data)
data = torch.stack(image_list,dim = 0).to(device)
output = model(data)
print(output.mean(dim=0)[:2])