import torch
import torchvision
from options import TestOptions
from dataset import dataset_single,dataset_output,dataset_label
from model import DRIT
from saver import save_my_imgs
import os
import sys

def main():
  # parse options
  parser = TestOptions()
  opts = parser.parse()

  # data loader
  print('\n--- load dataset ---')
  datasetA = dataset_label(opts, 'A', opts.input_dim_a)
  datasetB = dataset_output(opts, 'B', opts.input_dim_b)

  if opts.a2b:
    loader = torch.utils.data.DataLoader(datasetA, batch_size=1, num_workers=opts.nThreads)
    loader_attr = torch.utils.data.DataLoader(datasetB, batch_size=1, num_workers=opts.nThreads, shuffle=True)
  else:
    loader = torch.utils.data.DataLoader(datasetB, batch_size=1, num_workers=opts.nThreads)
    loader_attr = torch.utils.data.DataLoader(datasetA, batch_size=1, num_workers=opts.nThreads, shuffle=True)

  rept_num = 1
  
  opts.name = '111'
  opts.resume = '1.pth'

  # model
  print('\n--- load model ---')
  model = DRIT(opts)
  model.train()
  model.setgpu(opts.gpu)
  model.resume(opts.resume, train=True)

  # directory
  result_dir = opts.result_dir
  if not os.path.exists(result_dir):
    os.mkdir(result_dir)

  # test
  print('\n--- testing ---')
  for i in range(rept_num):
    for idx1, (img1, label, name1) in enumerate(loader):
      # if idx1 > 10:break
      print('{}/{}'.format(idx1, len(loader)))
      img1 = img1.cuda()
      imgs = []
      labels = []
      names = []
      for idx2, (img2, name2) in enumerate(loader_attr):
        if idx2 == opts.num:
          break
        img2 = img2.cuda()
        with torch.no_grad():
          if opts.a2b:
            img = model.test_forward_transfer(img1, img2, a2b=True)
          else:
            img = model.test_forward_transfer(img2, img1, a2b=False)
        imgs.append(img)
        labels.append(label)
        names.append(name1[0]+'_'+str(i)+str(idx2))


      save_my_imgs(imgs, names, result_dir + '/imgs', opts.name)
      save_my_imgs(labels, names, result_dir + '/labels', opts.name, True)

  return

if __name__ == '__main__':
  main()
