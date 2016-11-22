# coding: utf-8
# Classifying with LeNet
import numpy as np
import glob
import pdb
import matplotlib as mpl
if 1:
	mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
import random

caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import pickle
import caffe
import SCloadextract_FB as SCloadextract_FB


def testNet(film_tr,num_iter):
        #PRETRAINED = '../snapshots/'+film_tr+'A/FBalexNM2_BNS_iter_'+str(num_iter)+'.caffemodel'
        PRETRAINED = '../snapshots/'+film_tr+'/FBalex_BNS_iter_'+str(num_iter)+'.caffemodel'
        #MODEL_FILE = 'single_cells/FBalexNM_deploy.prototxt'
        MODEL_FILE = 'single_cells/FBalex_deploy.prototxt'
        net = caffe.Classifier(MODEL_FILE, PRETRAINED)
        caffe.set_mode_gpu()
        net.transformer.set_input_scale('data0', 0.00390625)	

	input_image_test = SCloadextract_FB.input_image_test
	label_test = SCloadextract_FB.label_test
	print 'Balance test set: ', np.mean(np.array(label_test))
	
	preds_all = list()
	numFeats = 51
	feats = list()
	for numim in range(len(input_image_test)):
		numCells = len(input_image_test[numim])
		featMat = np.zeros((numCells,numFeats))
		pred_ = np.zeros((numCells,1))
		for num_cell in range(numCells):
                        pred_[num_cell]= net.predictFB([input_image_test[numim][num_cell]],np.reshape(SCloadextract_FB.mov_test[numim][num_cell],(1,1)), oversample=False)[:,1]#.mean()
                        #pred_[num_cell]= net.predict([input_image_test[numim][num_cell]], oversample=False)[:,1]#.mean()
			featMat[num_cell,:50] = net.blobs['fc7_BN'].data.ravel()
			featMat[num_cell,50] = SCloadextract_FB.mov_test[numim][num_cell]#net.blobs['hdfdata'].data.ravel()[0] 

		feats.append(featMat)  # predict takes any number of images, and formats them for the Caffe net automatically
		preds_all.append(pred_)
		print numim

	res = dict()
	res['feats'] = feats
	res['pred_all'] = preds_all
	res['lab'] = label_test
	return res

if __name__ =="__main__":
	film_tr = sys.argv[1]
	film_tst = SCloadextract_FB.film
	num_iter = sys.argv[2]	
	res = testNet(film_tr,num_iter)
	label_test = res['lab']
	res['im_names'] = SCloadextract_FB.im_name_list
	feats = res['feats']
	fp = open('./single_cells/extract/featAlexBN_'+film_tr+'_2_test_'+film_tst+'latent_'+str(SCloadextract_FB.is_02)+'_'+str(num_iter)+'_51.pickle', 'w')
	pickle.dump(res,fp)
	fp.close()
	fm = './single_cells/extract/featAlexBN_'+film_tr+'_2_test_'+film_tst+'latent_'+str(SCloadextract_FB.is_02)+'_'+str(num_iter)+'_51.mat'
	import scipy as SP
	import scipy.io
	SP.io.savemat(fm, res, appendmat=True, format='5', long_field_names=True, do_compression=False, oned_as='row')
