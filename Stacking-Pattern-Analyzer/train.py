import torch
from transformers.optimization import get_linear_schedule_with_warmup
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ImageDataset
from model import StructNet
from loss import re_loss, mo_loss, re3_loss

## slip or twist or 3layer-slip
task = 'slip'
dataset_path='./dataset/'
batch_size = 32
learning_rate = 1e-3
num_epochs = 3000
device_ids = [0]

train_dataset = ImageDataset(images_path=dataset_path,is_train=True , task=task)
train_dataloader = DataLoader(train_dataset,num_workers=12 * len(device_ids),batch_size=batch_size*len(device_ids), shuffle=True)

if task == 'slip':
    model = StructNet(output_dim=2)
    criterion = re_loss
elif task == 'twist':
    model = StructNet(output_dim=1)
    criterion = mo_loss
elif task == '3layer-slip':
    model = StructNet(output_dim=4)
    criterion = re3_loss

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=num_epochs*len(train_dataloader)*0.05, 
                                            num_training_steps=num_epochs*len(train_dataloader))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

for epoch in range(num_epochs):
    train_losses = []
    shift_losses = []

    for batch_idx, (data, target) in enumerate(train_dataloader):
        data = data.to(device)
        target = target.to(device).float()
        optimizer.zero_grad()
        output = model(data)
        loss, shift_loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_losses.append(loss.item())
        shift_losses.append(shift_loss)
        
    lr = optimizer.param_groups[0]['lr']

    if epoch % 100 == 0:
        params = model.state_dict()
        if epoch != 0:
            torch.save(params, f'./model/model_params-{epoch}.pth')

        
    print(f'Epoch {epoch}, Total Loss {sum(train_losses) / len(train_losses)}, Shift Loss {sum(shift_losses) / len(shift_losses)}, lr {lr}')
    