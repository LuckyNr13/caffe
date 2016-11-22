# ===========================================
# 
# To get started we need to load a few modules and set some paths.

import cPickle as pickle
import glob
import numpy as np
import matplotlib as mpl
if 1:
    mpl.use('Agg')
import matplotlib.pyplot as plt
import random
import h5py
# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import pandas as pd
import caffe
import pdb

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
LOAD_IMAGES = 1
loadP=1
films = ['120602PH5','130218PH8','140206PH8'] #20 40k (I think)
#films = ['120602PH5','130218PH8'] #20 40k (I think)
#films = ['130218PH8','140206PH8'] #20 40k (I think)
#films = ['140206PH8','120602PH5'] #20 40k (I think)
#films = ['130218PH8'] #20 70k
#films = ['120602PH5'] #20 40k (I think)
#films = ['140206PH8']
#films = ['160223ME8']


is_02s = [0,1]

input_image_test = list()
input_image_name_test = list()
im_name_list = list()
label_test = list()
mov_test = list()

is_02 = 0

for film in films:
#film = films[0]
#for is_02 in is_02s:
    if is_02==1:
        zwo=str(2)
    else:
        zwo=str(0)
    #images_test00_Cen1Std1_OLD/
    IMAGE_TEST_DIR = '/home/florian_buettner/projects/scRNN/data/dataMatrices_150518/images_test0'+zwo+'_Cen1Std12/'+film+'/'
    IMAGE_TEST_FILE = './test'+film+'_0'+zwo+'.pickle'
    data_path = '/home/florian_buettner/caffe/data/single_cells/data_150518/'
    if is_02<3:
        IMAGE_LIST = pd.read_table(data_path+'test'+film+'_0'+zwo+'2.txt', sep=' ', header=None)
    else:
        IMAGE_LIST = pd.read_table(data_path+'test'+film+'_0'+zwo+'.txt', sep=' ', header=None)
    
    IMAGE_LIST = np.array(IMAGE_LIST.iloc[:,0])
    
    print  'Loading files for testing the model'
    #Get valid inds and labels
    if is_02<3:
        datah5 = h5py.File('/home/florian_buettner/caffe/data/single_cells/data_150518/test'+film+'_0'+zwo+'2.hdf5')
        mov = np.reshape(datah5['data'][:],(len(datah5['data'][:]),1))#[:, np.array([0])]
    else:
        datah5 = h5py.File('/home/florian_buettner/caffe/data/single_cells/data_150518/test'+film+'_0'+zwo+'.hdf5')
        mov = np.reshape(datah5['data'][:][:, np.array([0])],(len(datah5['data'][:][:, np.array([0])]),1))#[:, np.array([0])]
    print mov.shape
    print mov.shape
    all_files_test = IMAGE_LIST#[0:10]
    inds_val = [int(fp.split('_')[5][4:]) for fp in all_files_test]
    inds_array = np.asanyarray(inds_val)
    inds_unique_test = pd.unique(inds_array)

    if loadP ==1:
        fpickle = open('/home/florian_buettner/caffe/data/single_cells/data_150518/IMtest'+film+'_0'+zwo+'22.pickle', 'r')
        input_image_test.extend(pickle.load(fpickle))
        label_test.extend(pickle.load(fpickle))
        mov_test.extend(pickle.load(fpickle))
        im_name_list.extend(pickle.load(fpickle))
        fpickle.close()
    else:
        mov_idx = 0
        for imnum in inds_unique_test:
            is_imagelist_test = np.array([fnmatch.fnmatch(file_t, '*file'+str(imnum)+'_*') for file_t in all_files_test])
            imagelist_test = all_files_test[is_imagelist_test]
            label_test.append(int(imagelist_test[0].split('_')[6]))

            mov_test.append(mov[mov_idx:(mov_idx+len(imagelist_test)),:])
            mov_idx+=len(imagelist_test)
            im_name_list.append(imagelist_test)
            input_image_test.append([caffe.io.load_image(IMAGE_TEST_DIR+im,color=False)/0.00390625 for im in imagelist_test])
            #input_image_test.append([caffe.io.load_image(IMAGE_TEST_DIR+im,color=False) for im in imagelist_test])
            print imnum, 'cells of',len(inds_unique_test), 'processed'
