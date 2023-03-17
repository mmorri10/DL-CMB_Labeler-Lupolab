import glob, os
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import rotate

from keras.utils import Sequence


def load_scan_mat(matfile):
    '''
    Load single .mat file, returns data in dictionary format.
    '''
    try:
        mat = loadmat(matfile)
    except: # check file exists
        print('File {} not exists!'.format(matfile))
        return None
    return mat


def load_data_predict(filelist, shape=[24, 24, 12], contrasts=['swi']):
    '''
    Load batch of .mat data for predicting new subjects.

    Input:
        filelist: list of strings, specifies .mat filepath
        shape: list of int, 3D patch shape. [X, Y, Z]
        contrasts: list of strings, contrasts to use, 
                    can be extended to multi-channel(contrast) input
    Output:
        patches: NxWxHxZxC patches
        dummy_labels: length N null array
    '''
    # convert to string
    if isinstance(filelist, str):
        filelist = [filelist]

    all_patches = []
    for matfile in filelist:

        mat = load_scan_mat(matfile)
        if mat:
            centroids = mat['centroids']
            # extract patches of contrasts
            patches = [extract_patches(mat[con], shape, centroids) for con in contrasts]
            # channel last format
            patches = np.moveaxis(np.array(patches), 0, -1)
            all_patches.append(np.array(patches))

    all_patches = np.concatenate(all_patches)
    dummy_labels = [None] * len(all_patches)
        
    return all_patches, dummy_labels


def load_data(filelist, shape=[24, 24, 12], contrasts=['swi']):
    '''
    Load batch of .mat data.

    Input:
        filelist: list of strings, specifies .mat filepath
        shape: list of int, 3D patch shape. [X, Y, Z]
        contrasts: list of strings, contrasts to use, 
                    can be extended to multi-channel(contrast) input
    Output:
        patches: NxWxHxZxC patches
        labels: length N binary array (1-cmb, 0-fp)
    '''
    # convert to string
    if isinstance(filelist, str):
        filelist = [filelist]

    all_cmb_patches, all_fp_patches = [], []
    for matfile in filelist:

        mat = load_scan_mat(matfile)
        if mat:
            centroids_cmb = mat['centroids_cmb']
            centroids_fp = mat['centroids_fp']
            # extract patches of contrasts
            cmb_patches = [extract_patches(mat[con], shape, centroids_cmb) for con in contrasts]
            fp_patches = [extract_patches(mat[con], shape, centroids_fp) for con in contrasts]
            # channel last format
            cmb_patches = np.moveaxis(np.array(cmb_patches), 0, -1)
            fp_patches = np.moveaxis(np.array(fp_patches), 0, -1)

            all_cmb_patches.append(np.array(cmb_patches))
            all_fp_patches.append(np.array(fp_patches))

    all_cmb_patches = np.concatenate(all_cmb_patches)
    all_fp_patches = np.concatenate(all_fp_patches)

    patches = np.concatenate((all_cmb_patches, all_fp_patches))

    labels = np.concatenate((np.ones(len(all_cmb_patches)),
                             np.zeros(len(all_fp_patches))))
        
    return patches, labels            


def extract_patches(img, shape, centroids):
    '''
    Extract 3D patches centered at centroids with specified shape

    Input: 
        img: 3D image volume to extract patch
        shape: 3D shape, [X, Y, Z]
        centroids: Nx3 matrix of center coordinates
    Output:
        patches: NxWxHxZ matrix of N x 3D patches.
    '''
    def extract_single_patch(img_pad, shape, coord):
        X, Y, Z = shape[0], shape[1], shape[2]
        X1, X2 = int(coord[1])+X-int(X/2), int(coord[1])+X-int(X/2)+X
        Y1, Y2 = int(coord[0])+Y-int(Y/2), int(coord[0])+Y-int(Y/2)+Y
        Z1, Z2 = int(coord[2])+Z-int(Z/2), int(coord[2])+Z-int(Z/2)+Z
        return img_pad[X1:X2, Y1:Y2, Z1:Z2]

    X, Y, Z = shape[0], shape[1], shape[2]
    img = np.lib.pad(img, ((X,X),(Y,Y),(Z,Z)), 'edge')
    patches = [extract_single_patch(img, shape, c) for c in centroids]
    # catch error when len(centroids)==0
    patches = np.array(patches) if patches else np.zeros([0] + shape, dtype=np.uint8)
    return patches


def crop_patch(img, shape):
    shape0 = img.shape
    wx, wy, wz = int(shape[0]/2), int(shape[1]/2), int(shape[2]/2)
    x, y, z = int(shape0[1]/2), int(shape0[2]/2), int(shape0[3]/2)
    return img[:, x-wx:x+wx, y-wy:y+wy, z-wz:z+wz]


class SwiDataSequence(Sequence):
    '''
    Keras data sequence to load data
    '''
    def __init__(self, X, y, 
                 batch_size, shape=[16,16,8], augmenter=None, shuffle=False):
        self.X = X
        self.X = self.X / 255.0 # normalize SWI data
        self.y = y
        self.batch_size = batch_size
        self.shape = shape
        self.augmenter = augmenter
        assert len(X) == len(y)
        self.shuffle = shuffle
        if self.shuffle:
            self._shuffle(seed=123)


    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))


    def __getitem__(self, idx):
        batch_X = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        # data augmentation
        if self.augmenter:
            batch_X = np.array([self.augmenter(x) for x in batch_X])

        batch_X = crop_patch(batch_X, self.shape)

        return batch_X, batch_y


    def _shuffle(self, seed=None):
        if seed:
            np.random.seed(seed)
        randomize = np.arange(len(self.X))
        np.random.shuffle(randomize)
        self.X = self.X[randomize]
        self.y = self.y[randomize]

    def on_epoch_end(self):
        '''Shuffle data'''
        if self.shuffle:
            self._shuffle(seed=None)


def augment_img(img, rot_range, shift_range, flip_flag=True, 
                    noise_amp=0., random_const=0):
    # v2: shift, and the rotate
    # v3: shift->rotate, add random constant, add noise
    X, Y, Z, _ = img.shape

    flip_x = np.random.choice([0, 1], 1)
    flip_y = np.random.choice([0, 1], 1)
    flip_z = np.random.choice([0, 1], 1)
    angle = np.random.choice(np.arange(-rot_range, rot_range+1), 1)
    shift_x = np.random.choice([0,-shift_range], 1)
    shift_y = np.random.choice([0,-shift_range], 1)
    shift_z = np.random.choice([0], 1)

    # random flip
    if flip_x and flip_flag:
        img = img[::-1,:,:]
    if flip_y and flip_flag:
        img = img[:,::-1,:]
    if flip_z and flip_flag:
        img = img[:,:,::-1]
    
    # random shift
    img = np.lib.pad(img, ((shift_range,shift_range),(shift_range,shift_range),(shift_range,shift_range),(0,0)), 'edge')
    img = img[int(shift_range+shift_x):int(X+shift_x+1), int(shift_range+shift_y):int(Y+shift_y+1), \
                int(shift_range+shift_z):int(Z+shift_z+1), :]
    
    # random rotate
    img = rotate(img, angle=angle, reshape=False)

    # random constant
    const = np.random.uniform(low=-random_const, high=random_const)
    
    # random noice
    noise = np.random.uniform(low=-noise_amp, high=noise_amp, size=img.shape)
    img = img + const + noise

    return img


def load_input_filelist(filename):
    '''Load txt input file in ../input directory'''
    with open(filename, 'r') as f:
        filelist = f.readlines()
    filelist = [f.strip() for f in filelist] # remove '\n'
    return filelist


# demo/debug
if __name__ == '__main__':

    aug_params = {'rot_range'   : 180,
                  'shift_range' : 1  ,
                  'flip_flag'   : True, 
                  'noise_amp'   : 0, 
                  'random_const': 0}

    augmenter = lambda x: augment_img(x, **aug_params)

    filelist = glob.glob('../data/3T_CCM/*_?.mat')[:5]
    print(filelist)

    X, y = load_data(filelist, contrasts=['swi'])
    sds = SwiDataSequence(X, y, batch_size=32, augmenter=None, shuffle=False)
    for i in range(10):
        X, y = sds[i]
        print(X.shape, y.shape, X.mean(), X.max(), X.min())
