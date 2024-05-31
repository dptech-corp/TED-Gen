from ase.io import read, write
from ase.build import molecule,rotate,cut
from ase.visualize import view
from ase import Atoms
import torchvision.transforms.functional as TF
from collections import Counter
import subprocess
import os
import cv2
import random
import numpy as np
from PIL import Image

def to_database(database_path = '', dat_path = ''):

    if not os.path.exists(database_path+'/data'):
        os.makedirs(database_path+'/data')
    if not os.path.exists(database_path+'/label'):
        os.makedirs(database_path+'/label')

    with open(dat_path, 'r') as data_paths:
        data_path = data_paths.readlines()
        for i in range(len(data_path)):
            label_n = Image.open(data_path[i][:-1] + '/ReSe2.png')
            for img in os.listdir(data_path[i][:-1]+'/image'):
                image = Image.open(data_path[i][:-1]+'/image/'+img)
                image = TF.crop(image, image.size[0]//2 - 800, image.size[1]//2 - 300, 1024, 1024)
                label = TF.crop(label_n.copy(), label_n.size[0]//2 - 800, label_n.size[1]//2 - 300, 1024, 1024)
                image.save(database_path + '/data/'+os.path.splitext(img)[0]+'.png','PNG')
                label.save(database_path + '/label/'+os.path.splitext(img)[0]+'.png','PNG')

def write_xyz(structure, file_name='', no_show_s = True):
    file = ''
    symbols = structure.get_chemical_symbols()
    atom_counts = Counter(symbols)
    for atom, count in atom_counts.items():
        file += atom + str(count)
    file += '\t' + str(len(structure)) + '\n'

    cell = structure.get_cell()
    file += str(cell[0][0]) + '\t' + str(cell[1][1]) + '\t' + str(cell[2][2]) + '\n' 

    write('temp.xyz',structure)
    with open('temp.xyz', 'r') as p:
        content = p.read().splitlines()
        i=2
        for line in content:
            if i >= 0:
                i-=1
                continue

            if ('S' in line) and no_show_s:
                continue

            line = line.replace('S','16')
            line = line.replace('Re','75')
            a = [value for value in line.split(' ') if value is not None and value != ""]
            if float(a[2])<0 or float(a[2])>69.82:
                continue

            for item in a:
                file += item + '\t'
            file += '1' + '\t'
            file += '0' + '\t' +'\n'
    
        file += '-1'

    with open(file_name, 'w') as f:
        f.write(file)

def save_label(size,structure,file_name,p_size = 0):
    img = np.zeros([int(structure.cell[1][1]/(structure.cell[0][0]/size)),size],np.uint8)

    for atom in structure:
        if atom.symbol =='Re':
            p = atom.position[:2]/(structure.cell[0][0]/size)
            p =p.astype(int)
            cv2.circle(img, p, p_size, (255, 255, 255), -1)

    cv2.imwrite(file_name+'.png',img)

def save_train_label(size,layer1,layer2,file_name,p_size = 10):

    if layer1.cell[1][1] > layer1.cell[0][0]:
        img = np.zeros([int(layer1.cell[1][1]/(layer1.cell[0][0]/size)),size],np.uint8)
    else:
        img = np.zeros([size,int(layer1.cell[0][0]/(layer1.cell[1][1]/size))],np.uint8)


    img1 = img.copy()
    img2 = img.copy()

    for atom in layer1:
        if atom.symbol =='Re':
            if layer1.cell[1][1] > layer1.cell[0][0]:
                p = atom.position[:2]/(layer1.cell[0][0]/size)
            else:
                p = atom.position[:2]/(layer1.cell[1][1]/size)
            p =p.astype(int)+1
            if p_size == 0:
                cv2.circle(img1, p, p_size, (255, 255, 255), -1)
            else:
                cv2.circle(img1, p, p_size, (255, 255, 255), -1)

    for atom in layer2:
        if atom.symbol =='Re':
            if layer1.cell[1][1] > layer1.cell[0][0]:
                p = atom.position[:2]/(layer2.cell[0][0]/size)
            else:
                p = atom.position[:2]/(layer2.cell[1][1]/size)
            p =p.astype(int)+1
            if p_size == 0:
                cv2.circle(img2, p, p_size, (255, 255, 255), -1)
            else:
                cv2.circle(img2, p, p_size, (255, 255, 255), -1)

    merged_array = np.concatenate(([img1], [img2], [img]), axis=0)
    label = np.transpose(merged_array, (1, 2, 0))

    cv2.imwrite(file_name+'.png',label)

def run_computem(incostem_path,param_path):

    for param in os.listdir(param_path):
        process = subprocess.Popen(incostem_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
        with open(param_path + '/' + param,'r') as p:
            lines = p.readlines()
            for line in lines:
                process.stdin.write(line)
                process.stdin.flush()
        
        output = process.communicate()[0]

def generate_param(path = '',image_size = 4096,params = ['','','',''], d_num = 5, s_num = 5):
    if not os.path.exists(path+'/param'):
        os.makedirs(path+'/param')
    if not os.path.exists(path+'/image'):
        os.makedirs(path+'/image')

    A1_param_mean             = 1*10**-6  #mean of C12 (#mm)
    A1_param_std              = 1*10**-6  #std of C12 (#mm)
    B2_param_mean             = 10*10**-6 #mean of 1/3*C21 (#mm)
    B2_param_std              = 20*10**-6 #std of 1/3*C21 (#mm)
    A2_param_mean             = 25*10**-6 #mean of C23 (#mm)
    A2_param_std              = 50*10**-6 #std of C23 (#mm)

    C12a = 0
    C12b = 0
    C21a = 0
    C21b = 0
    C23a = 0
    C23b = 0

    pd = 10.0/d_num 
    ps = 0.1/s_num

    for i in range(d_num):
      for j in range(s_num):
        Defocus = 35 + i * pd
        Source_size_at_specimen = 0.6 + j * ps

        idx = random.uniform(10000,20000)
        # A1 = np.random.normal(A1_param_mean, A1_param_std )
        # B2 = np.random.normal(B2_param_mean, B2_param_std )
        # A2 = np.random.normal(A2_param_mean, A2_param_std )
        # A1_angle = 2*np.pi*np.random.random()
        # B2_angle = 2*np.pi*np.random.random()
        # A2_angle = np.random.normal(np.pi/2, np.pi/24) #Mean at 90deg, std= 7.5deg
        # C12a = A1 * np.cos(A1_angle)
        # C12b = A1 * np.sin(A1_angle)
        # C21a = 3*B2 * np.cos(B2_angle)
        # C21b = 3*B2 * np.sin(B2_angle)
        # C23a = A2 * np.cos(A2_angle)
        # C23b = A2 * np.sin(A2_angle)
        
        with open(path+'/param/'+str(i)+str(j)+'.param','w') as param:
            param.write(path+'/ReSe2.xyz\n')
            #各维图像数量
            param.write('1 1 1\n')
            #图像存储地址
            param.write(path+'/image/'+params[0][:4]+'_'+params[1][:4]+'_'+params[2][:4]+'_'+params[3][:4]+'_'+str(Defocus)[:4]+'_'+str(Source_size_at_specimen)[:4]+'_'+str(idx)[:4]+'.tif\n')
            #存储的图像大小
            param.write(str(image_size)+' '+str(image_size)+'\n')
            #STEM probe parameters
            param.write('300 0 0 '+str(Defocus)+' 21.3\n')
            #ADF detector angles thetamin, thetamax (in mard)
            param.write('39 200\n')
            #type higher order aber
            param.write('C12a '+str(C12a)+' C12b '+str(C12b)+' C21a '+str(C21a)+' C21b '+str(C21b)+' C23a '+str(C23a)+' C23b '+str(C23b)+' END\n')
            #Source size at specimen(FWHM in Ang.)
            param.write(str(Source_size_at_specimen)+'\n')
            #Defocus spread(FWHM in Ang.)
            param.write('0 \n')
            #Add noise
            param.write('n\n')
            #Type total probe current and dwell time 
            param.write('-1\n')

def generate_data(structure_path='',DAT_path = '',move_list = None,incostem_path = './incostem',image_size = 4096,image_num = 1, no_show_s = True):

    atoms = read(structure_path)

    ReSe2_layer1 = atoms.copy()
    ReSe2_layer2 = atoms.copy()

    ReSe2_layer2.positions[:,2] += 7.

    if random.uniform(0,1)>2.0:
        r = random.uniform(0.,180.)
    else: r = 0
    center = atoms.get_cell().sum(axis=0) / 2.0
    ReSe2_layer2.rotate(r,'z',center)

    # if random.uniform(0,1)>0.1:
    #     move_x = random.uniform(0.3,2.4)
    # else:
    #     move_x = random.uniform(3.1,3.15)

    # if random.uniform(0,1)>0.2:
    #     move_y = random.uniform(0.3,2.1)
    # else:
    #     move_y = random.uniform(2.6,2.7)

    move_x = move_list[0]
    move_y = move_list[1]
    move_z = random.uniform(-0.5,0.5)

    ReSe2_layer2.positions[:,0] += move_x
    ReSe2_layer2.positions[:,1] += move_y
    ReSe2_layer2.positions[:,2] += move_z

    if random.uniform(0,1)>2.0:
        if_flip = '1'
        ReSe2_layer2.rotate(180,'y',center)
    else:if_flip = '0'

    for i in range(image_num):
        ReSe2_layer1.pop(random.randint(0,len(ReSe2_layer1)-10))
        ReSe2_layer2.pop(random.randint(0,len(ReSe2_layer2)-10))

    ReSe2 = ReSe2_layer1 +ReSe2_layer2

    ReSe2_layer2 = ReSe2[len(ReSe2_layer1):]
    ReSe2_layer1 = ReSe2[:len(ReSe2_layer1)]

    path = './data/ReSe2_'+str(move_x)[:6]+'_'+str(move_y)[:6]+'_'+str(r)[:4]+'_'+if_flip
    os.makedirs(path)

    if not os.path.exists('./dat'):
        os.makedirs('./dat')

    write_xyz(ReSe2,file_name=path+'/ReSe2.xyz', no_show_s = no_show_s)
    # save_label(image_size,ReSe2,file_name=path+'/ReSe2')
    with open(DAT_path,'a') as dat:
        dat.write(path+'\n')
        

    ReSe2_layer1 = ReSe2[:len(ReSe2_layer1)]
    write_xyz(ReSe2_layer1,file_name=path+'/ReSe2_layer1.xyz')
    # save_label(image_size,ReSe2_layer1,file_name=path+'/ReSe2_layer1')

    ReSe2_layer2 = ReSe2[len(ReSe2_layer1):]
    write_xyz(ReSe2_layer2,file_name=path+'/ReSe2_layer2.xyz')
    # save_label(image_size,ReSe2_layer2,file_name=path+'/ReSe2_layer2')

    save_train_label(image_size,ReSe2_layer1,ReSe2_layer2,file_name=path+'/ReSe2')

    generate_param(path,image_size,params = [str(move_x),str(move_y),str(r),if_flip], d_num = 1, s_num = 1)

    run_computem(incostem_path,path+'/param')

for x in range(-33,33,1):
    for y in range(0,28,1):
            generate_data(structure_path='./structure/ReSe2/one_layer.xyz',
                          DAT_path = './dat/ReSe2.dat',
                          move_list=[x/10.,y/10.],
                          image_size = 2048,
                          image_num=1, 
                          no_show_s = False)

to_database(database_path='./database',dat_path='./dat/ReSe2.dat')