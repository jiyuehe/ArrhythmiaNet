# Copyright 2026 Jiyue He
# Contact: jiyuehe@alumni.upenn.edu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#%%
import os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
import modules
import torch
import numpy as np

torch.cuda.empty_cache() # release unused cached CUDA memory

#%%
# settings
train_predict_flag = 0 # 1: train; 0: predict
testing_data_flag = 1 # 0: simulation data; 1: clinical data
arrhythmia_flag = 1 # 0: focal; 1: flutter
continue_training_flag = 0 # 0: train from scratch; 1: load best_unet_model.pth and continue training

parameters = {}
parameters['batch_size'] = 10 # number of training samples. 1 sample is an electrograms-activation_maps pairs of 1 simulation on 1 geometry
parameters['learning_rate'] = 1e-4 # too small or too big are both bad
parameters['epochs'] = 30 # maximum training epochs. may stop earlier with early stopping
parameters['early_stopping_patience'] = 10 # stop training if no improvement for this many epochs
if arrhythmia_flag == 0:
   parameters['data_folder_simulation'] = Path('/home/j/Desktop/hdd/share_folder/simulation_results/one_focal')
   parameters['result_folder'] = Path(script_dir) / 'result' / 'focal'
   parameters['t_start'] = 0
   parameters['time_step'] = 1
   parameters['n_timepoints'] = 500
   parameters['t_end'] = parameters['t_start'] + parameters['n_timepoints'] * parameters['time_step']
if arrhythmia_flag == 1:
   parameters['data_folder_simulation'] = Path('/home/j/Desktop/hdd/share_folder/simulation_results/flutter')
   parameters['result_folder'] = Path(script_dir) / 'result' / 'flutter'
   parameters['t_start'] = 500
   parameters['time_step'] = 1
   parameters['n_timepoints'] = 500
   parameters['t_end'] = parameters['t_start'] + parameters['n_timepoints'] * parameters['time_step']
parameters['data_folder_mesh'] = Path('/home/j/Desktop/ssd/carto3_data/processed')
parameters['data_folder_patient'] = Path('/home/j/Documents/box/carto3_files/data_npz/good')
parameters['data_folder_processed_clinical'] = Path('/home/j/Desktop/ssd/git/ArrhythmiaNet/result/processed_clinical_data')
parameters['result_folder'].mkdir(exist_ok=True)

#%%
# categorize the data files into training, validation, and test sets
file_names_train, file_names_validation, file_names_test = modules.utility.categorize_files_into_train_validation_test(parameters['data_folder_simulation'])
parameters['file_names_train'] = file_names_train
parameters['file_names_validation'] = file_names_validation
parameters['file_names_test'] = file_names_test

print(f'n_train: {len(file_names_train)}, n_validation: {len(file_names_validation)}, n_test: {len(file_names_test)}')

#%%
# create the U-Net model
parameters['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
   import MinkowskiEngine as ME # https://nvidia.github.io/MinkowskiEngine/overview.html

   if arrhythmia_flag == 0: # focal
      parameters['model'] = modules.unet_minkowski.MinkowskiUNet(in_channels=parameters['n_timepoints']+1, out_channels=1, D=3).to(parameters['device']) 
      # D is the dimension of the input data
      # in_channels is n_timepoints channels for the electrogram, plus 1 channel as an indicator for electrode nodes: 1 for electrode nodes, 0 for non-electrode nodes
      # out_channels is 1 for activation time map
   if arrhythmia_flag == 1: # flutter
      parameters['model'] = modules.unet_minkowski.MinkowskiUNet(in_channels=parameters['n_timepoints']+1, out_channels=3, D=3, out_activation='sin_cos_mask').to(parameters['device'])
      # out_channels is 3 for of phase map: sin(phase), cos(phase), a mask for non-conductive region due to pulmonary vein isolation

   print(f'model created with {sum(p.numel() for p in parameters["model"].parameters())} parameters')
   # print(parameters['model']) # use print to show model architecture
except ImportError:
   print('MinkowskiEngine is not installed.')

# train the model
if train_predict_flag == 1:
   print('train model')

   if continue_training_flag == 1: # load pre-trained model to continue training
      model_path = parameters['result_folder'] / 'best_unet_model.pth'
      print(f'loading pre-trained model from {model_path}')
      parameters['model'].load_state_dict(torch.load(model_path, map_location=parameters['device']))

   # train the model
   modules.train_predict.train_model(parameters, arrhythmia_flag)

   # plot loss history
   modules.result_analysis.plot_loss_history(parameters['result_folder'])

# predict using the trained model
if train_predict_flag == 0:
   print('model prediction')

   parameters['model'].load_state_dict(torch.load(parameters['result_folder'] / 'best_unet_model.pth', map_location=parameters['device'])) # load the best model

   if testing_data_flag == 0: # simulation
      N = 10
      file_names_test = np.array(parameters['file_names_test'])
      if N < len(file_names_test):
         indices = np.linspace(0, len(file_names_test) - 1, N, dtype=int) # uniformly sample N files
         file_names_test = file_names_test[indices]
   elif testing_data_flag == 1: # clinical
      file_names_test_focal = [
         # '10_2-LA FAM',
         # '16_2-LA FAM',
         # '40_3-la',
         # '60_2-LA_ACTIVATION',
         # '89_1-LAFAM SNS',
         # '90_1-ablation',
         # '91_2-LA sinus voltage baseline',
         # '91_2-LA sinus with ablation lesions',
         # '92_1-la volt',
         # '94_1-1-ReLA voltage_sinus',
         # '95_1-SINUS CARTO FINDER',
         '99_2-LaFAM_cartofinder_data',
         # '100_1-LA FAM1',
         # '101_1-lagood',
         # '112_6-LA CL 300',
      ]

      file_names_test_flutter = [
         # '19_1-x',
         # '102_2-1-1-ReLA FAM AT',
         # '102_2-LA FAM AFL1edit',
         '107_3-LA CL 270',
         # '114_1-LA fam afl 250',
         # '114_1-LA fam afl 250_pre abl',
      ]

      # add '_clinical_data.npz' to file name ends
      file_names_test_focal = [name + '_clinical_data.npz' for name in file_names_test_focal]
      file_names_test_flutter = [name + '_clinical_data.npz' for name in file_names_test_flutter]

      if arrhythmia_flag == 0:
         file_names_test = file_names_test_focal
      elif arrhythmia_flag == 1:
         file_names_test = file_names_test_flutter

   predicted_data, truth_data = modules.train_predict.predict(parameters, file_names_test, testing_data_flag, arrhythmia_flag)

   # convert all elements to numpy arrays if they are tensors
   predicted_data = [x.numpy() if hasattr(x, 'numpy') else x for x in predicted_data]
   truth_data = [x.numpy() if hasattr(x, 'numpy') else x for x in truth_data]

   print('prediction done')

   # plot the predicted and true activation time maps
   modules.result_analysis.plot_truth_and_predicted_map(truth_data, predicted_data, file_names_test, parameters, testing_data_flag, arrhythmia_flag)

print('done')

#%%
