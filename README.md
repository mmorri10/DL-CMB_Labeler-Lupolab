# DL-CMB_Labeler
Code and trained models for deep learning-based CMB detection.

**Overview:** This is a 3D deep learning tool for automated false positive removal after running our Lupo-Lab/CMB_Labeler tool. To read more about the network architecture and training processes please refer to: https://link.springer.com/article/10.1007%2Fs10278-018-0146-z

**Directories:**
python_code: deep CNN implementation, training, testing and prediction code.
matlab_code: essential matlab code to prepare data for training, testing and prediction.
final_models: trained model weights and demo data.

**Usage:**

1. Run Lupo-Lab/CMB_Labeler with 'semion' (user-guided GUI for FP reduction) or 'semioff'.
2. Run subjectlist.m to create datadir.mat file (this stores the paths of all the subjects you want to run)
3. Run create_mat_file.m to generate the .mat data for deep network python script (see sample.mat for an example of correct output)
4. Run predict.py MAT_FILE_PATH (in virtualenv/conda):

python predict.py MAT_FILE_PATH

The result will be added to the original .mat file. See requirements.txt to set up the correct environment.


For support please contact: janine.lupo at ucsf.edu
