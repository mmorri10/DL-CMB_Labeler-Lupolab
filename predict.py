'''
Predict CMB labels for new subjects.

Eason Chen
03/07/2019
'''
from datetime import datetime
import numpy as np
import os
import time
import json
import argparse
from scipy.io import loadmat, savemat

from keras.models import load_model
from util import *


MODEL_PATH = '../final_models/7T_swi+3T_btumor_77'
N_MODELS = 5
THRESHOLD = 0.1 # cmb classification threshold
REQUIRED_FIELDS = ['centroids', 'swi']

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # don't print the tf debugging information


def parse_arg():
    '''
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser(description='Run Deep CMB networks on new subjects.')
    parser.add_argument('--input', dest='data_filepath', action='store', default=None,
                        help='Input .mat file to process.')
    args = parser.parse_args()
    return args


def is_datafile_valid(data_filepath):
    '''
    Check if input data filepath and .mat is valid.
    '''
    if not os.path.isfile(data_filepath):
        print('File {} not existed.'.format(data_filepath))
        return False
    if not data_filepath.endswith('.mat'):
        print('File {} not .mat format.'.format(data_filepath))
        return False
    try:
        temp = loadmat(data_filepath)
        for k in REQUIRED_FIELDS:
            _ = temp[k]
    except:
        print('Error happened when checking {}. Check file integrity or missing fields'.format(data_filepath))
        return False
    print('==>Data file check finished.')
    return True
        


def predict(data_filepath):
    '''
    Load data, load networks and predict labels
    
    Input:
        data_filepath: filepath of the preprocessed .mat file.
            * Fields required in .mat:
                See REQUIRED_FIELDS
                
    Output:
        None. But adds labels to the .mat file
    '''
    # load mat data
    print('==>Start running Deep CMB networks...')
    start_time = time.time()
    
    all_yp = []
    mat = load_scan_mat(data_filepath)
    print('==>Total number of networks: {}'.format(N_MODELS))
    
    for i in range(N_MODELS):

        print('    ==>Running MODEL {}'.format(i+1))
        # load model
        model_filepath = os.path.join(MODEL_PATH, 'model_{}_best.hdf5'.format(i))
        assert os.path.exists(model_filepath)
        model = load_model(model_filepath)
        
        X, y = load_data_predict(data_filepath)
        sds = SwiDataSequence(X, y, batch_size=16, 
                              shape=model.input.shape.as_list()[1:4], 
                              augmenter=None)
        
        yp = model.predict_generator(sds).ravel()
        all_yp.append(yp)

    raw_pred = np.mean(np.stack(all_yp, axis=1), axis=1)
        
    cmb_label = (raw_pred > THRESHOLD).astype(int)
    end_time = time.time()
    
    mat['raw_pred'] = raw_pred
    mat['cmb_label'] = cmb_label
    mat['elapsed_time'] = end_time - start_time
    mat['timestamp'] = datetime.now().isoformat(timespec='minutes')
    mat['threshold'] = THRESHOLD
    savemat(data_filepath, mat)
    print('==>All done. Saved to the original .mat file.')

    
if __name__ == '__main__':
    args = parse_arg()
    data_filepath = args.data_filepath
    if is_datafile_valid(data_filepath):
        predict(data_filepath)

        
