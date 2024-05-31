import torch
import torchvision
from options import TestOptions
from dataset import dataset_single,dataset_output,dataset_label
from model import DRIT
from saver import save_my_imgs
import os


def main():
  # parse options
  parser = TestOptions()
  opts = parser.parse()

  # data loader
  print('\n--- load dataset ---')
  datasetA = dataset_label(opts, 'A', opts.input_dim_a)
  datasetB = dataset_output(opts, 'B', opts.input_dim_b)
  # datasetA = dataset_single(opts, 'A', opts.input_dim_a)
  # datasetB = dataset_single(opts, 'B', opts.input_dim_b)
  if opts.a2b:
    loader = torch.utils.data.DataLoader(datasetA, batch_size=1, num_workers=opts.nThreads)
    loader_attr = torch.utils.data.DataLoader(datasetB, batch_size=1, num_workers=opts.nThreads, shuffle=True)
  else:
    loader = torch.utils.data.DataLoader(datasetB, batch_size=1, num_workers=opts.nThreads)
    loader_attr = torch.utils.data.DataLoader(datasetA, batch_size=1, num_workers=opts.nThreads, shuffle=True)

  # list = [19,29,49,59,109,119,269,369,399,439,449,479,489,499,509,539,609,699]
  # list = [539,639]
  list = [279,329,349,599,679,709]
  rept_num = 1
  # list = []
  
  for id in list:
    print('start generate pth ' + str(id))
    opts.name = str(id)
    # opts.resume = '/mnt/vepfs/users/ycjin/moore/results/2023-08-04-09/'+str(id).zfill(5)+'.pth'
    # opts.resume = '/mnt/vepfs/users/ycjin/moore/DRIT/results/2023-08-09-02/'+str(id).zfill(5)+'.pth'
    opts.resume = '/mnt/vepfs/users/ycjin/moore/DRIT/results/2023-09-12-05_300/'+str(id).zfill(5)+'.pth'

    # model
    print('\n--- load model ---')
    model = DRIT(opts)
    model.train()
    model.setgpu(opts.gpu)
    model.resume(opts.resume, train=True)

    # directory
    # result_dir = os.path.join(opts.result_dir, opts.name)
    result_dir = opts.result_dir
    if not os.path.exists(result_dir):
      os.mkdir(result_dir)

    # test
    print('\n--- testing ---')
    for i in range(rept_num):
      for idx1, (img1, label, name1) in enumerate(loader):
        print('{}/{}'.format(idx1, len(loader)))
        img1 = img1.cuda()
        # imgs = [img1]
        # names = ['input']
        imgs = []
        labels = []
        names = []
        for idx2, (img2, name2) in enumerate(loader_attr):
          if idx2 == opts.num:
            break
          img2 = img2.cuda()
          with torch.no_grad():
            if opts.a2b:
              ca,cb,aa,ab,img = model.test_forward_step(img1, img2, a2b=True)
              # model.init_gen(img1,img2)
            else:
              img = model.test_forward_step(img2, img1, a2b=False)
          imgs.append(img)
          labels.append(label)
          # names.append('output_{}'.format(idx2))
          names.append(name1[0]+'_'+str(i)+str(idx2))
        # save_imgs(imgs, names, os.path.join(result_dir, '{}'.format(idx1)))


        save_my_imgs(imgs, names, result_dir + '/imgs', opts.name)
        save_my_imgs(labels, names, result_dir + '/labels', opts.name, True)
        # model.update_D(img1, img2)

        # assembled_images = model.aim_output()
        # img_filename = os.path.join(result_dir, name1[0] + opts.name + '.png' )
        # torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)

  return

if __name__ == '__main__':
  main()
