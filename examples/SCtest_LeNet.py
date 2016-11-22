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
import scipy as SP
import scipy.io
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import pickle
import caffe
import SCloadtest_FB as SCloadtest_FB
import time

def testNet(film_tr,num_iter):
    #PRETRAINED = '../snapshots/'+film_tr+'A/FBalex_BNAdm_iter_'+str(num_iter)+'.caffemodel'
    PRETRAINED = '../snapshots/'+film_tr+'/FBalexNM_BNS_iter_'+str(num_iter)+'.caffemodel'
    #MODEL_FILE = 'single_cells/FBalex_deploy.prototxt'
    MODEL_FILE = 'single_cells/FBalexNM_deploy.prototxt'
    print PRETRAINED
    net = caffe.Classifier(MODEL_FILE, PRETRAINED)
    caffe.set_mode_gpu()
    net.transformer.set_input_scale('data0', 0.00390625)
    
    input_image_test = SCloadtest_FB.input_image_test
    label_test = SCloadtest_FB.label_test
    print 'Balance test set: ', np.mean(np.array(label_test))
    
    preds = list()
    preds_all = list()
    for numim in range(len(input_image_test)):
        #prediction = net.predictFB(input_image_test[numim],SCloadtest_FB.mov_test[numim], oversample=False)[:,1]#.mean()
        prediction = net.predict(input_image_test[numim],oversample=False)[:,1]#.mean()
        preds.append(prediction.mean())  # predict takes any number of images, and formats them for the Caffe net automatically
        preds_all.append(prediction)  # predict takes any number of images, and formats them for the Caffe net automatically
        print numim, 'of ', len(input_image_test), ' predicted'
    res = dict()
    res['pred'] = preds
    res['pred_all'] = preds_all
    res['lab'] = label_test
    res['im_name'] = SCloadtest_FB.im_name_list
    return res

if __name__ =="__main__":
    film_tr = sys.argv[1]
    film_tst = SCloadtest_FB.film
    num_iter = sys.argv[2]    
    res = testNet(film_tr,num_iter)
    pred = res['pred']
    if SCloadtest_FB.is_02==3:
        label_test = (SP.array(res['lab'])>0)*1.0
    else:
          label_test = res['lab']
    fpr, tpr, thresholds = metrics.roc_curve(label_test, pred)
    auc = metrics.auc(fpr, tpr)
    print 'AUC = ', auc
    predictions = (np.array(pred)>0.5)*1
    print 'F1 = ',metrics.f1_score(label_test, predictions, pos_label=None, average='macro')
    f1macro = metrics.f1_score(label_test, predictions, pos_label=None, average='macro')
    f1_i = metrics.f1_score(label_test, predictions)
    f1_i_0 = metrics.f1_score(label_test, predictions, pos_label = 0)
    SP.io.savemat('./single_cells/res/prediAlexNM_test_'+film_tst+'latent_'+str(SCloadtest_FB.is_02)+'_iter_'+str(num_iter)+'_'+str(SP.round_(auc,3))+\
        '_F1_'+str(f1macro)+'.mat', res, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')    
