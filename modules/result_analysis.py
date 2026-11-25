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

import math
import os
import sys
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
os.chdir(script_dir) # change the working directory
script_dir = Path(script_dir)

from platform import node
import matplotlib.pyplot as plt 
import numpy as np
import plotly.graph_objects as go

COMMON_DIR = Path('/home/j/Documents/box/codes/common')
sys.path.insert(0, str(COMMON_DIR))
import common

def plot_loss_history(result_folder):
    loss_history_path = result_folder / 'loss_history.txt'
    train_loss_history = []
    val_loss_history = []
    with open(loss_history_path, 'r') as f:
        for line in f:
            if line != 'train_loss\tval_loss\n':
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    train_loss_history.append(float(parts[0]))
                    val_loss_history.append(float(parts[1]))

    plt.figure()
    plt.plot(train_loss_history, 'r-', label='Training Loss')
    plt.plot(val_loss_history, 'b-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'r: train loss ({min(train_loss_history)*10**3:.2f}e-3), b: valid loss ({min(val_loss_history)*10**3:.2f}e-3)')
    plt.legend()
    plt.grid(True)
    plt.savefig(result_folder / 'loss_history.png', dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()

def voxelize_nodes(node):
    min_coords = np.floor(node.min(axis=0)).astype(int) # node.min(axis=0) returns a 1D array containing the minimum x, y, and z values across all points
    max_coords = np.ceil(node.max(axis=0)).astype(int) # node.max(axis=0) returns a 1D array containing the maximum x, y, and z values across all points
    grid_shape = max_coords - min_coords + 1
    grid_indices = (node - min_coords).astype(int)
    
    voxels = np.zeros(grid_shape, dtype=bool)
    voxels[grid_indices[:,0], grid_indices[:,1], grid_indices[:,2]] = True

    return voxels, grid_indices

def plot_truth_and_predicted_map(truth_data, predicted_data, file_names_test, parameters, train_predict_flag, arrhythmia_flag):
    n_samples = len(predicted_data)

    for sample_id in np.arange(n_samples):
        print(f'plotting sample {sample_id+1}/{n_samples}')

        # load geometry data
        if train_predict_flag == 0: # simulation
            name_prefix = file_names_test[sample_id].split('_simulation_results_')[0]
        elif train_predict_flag == 1: # clinical
            name_prefix = file_names_test[sample_id].split('_clinical_data')[0]

        mesh_data = np.load(parameters['data_folder_patient'] / f'{name_prefix}_mesh.npz', allow_pickle=True)
        node = mesh_data['voxel']
        node = node - np.round(node.mean(axis=0)).astype(int) # shape (n_nodes, 3)
        n_nodes = node.shape[0]

        # electrode voxel ids
        if train_predict_flag == 0: # simulation
            pass
            # # randomly make some electrodes bad
            # bad_electrode_ratio = 0
            # n_bad_electrode = int(bad_electrode_ratio * n_nodes)
            # non_e_id = np.random.choice(n_nodes, n_bad_electrode, replace=False)
        elif train_predict_flag == 1: # clinical
            data = np.load(parameters['data_folder_patient'] / f'{name_prefix}_clinical_data.npz', allow_pickle=True)
            map_data = {k: data[k] for k in data.files}

            # find the good electrode nodes that have good signals
            act = map_data['clinical_activation_uni']
            good_e_id = [i for i, x in enumerate(act) if x != 0]
            non_e_id = np.setdiff1d(np.arange(n_nodes), good_e_id)

        if arrhythmia_flag == 0:
            data_truth = truth_data[sample_id].flatten()
            data_predicted = predicted_data[sample_id].flatten()
            color_tpye = 'red-blue'
        if arrhythmia_flag == 1:
            truth_channels = truth_data[sample_id]
            predicted_channels = predicted_data[sample_id]
            if hasattr(truth_channels, 'detach'):
                truth_channels = truth_channels.detach().cpu().numpy()
            if hasattr(predicted_channels, 'detach'):
                predicted_channels = predicted_channels.detach().cpu().numpy()

            truth_sin = truth_channels[0]
            truth_cos = truth_channels[1]
            truth_no_value = truth_channels[2] > 0.5
            data_truth = np.mod(np.arctan2(truth_sin, truth_cos), 2 * np.pi) / (2 * np.pi)
            data_truth[truth_no_value] = np.nan

            predicted_sin = predicted_channels[0]
            predicted_cos = predicted_channels[1]
            predicted_no_value = predicted_channels[2] > 0.5
            data_predicted = np.mod(np.arctan2(predicted_sin, predicted_cos), 2 * np.pi) / (2 * np.pi)
            data_predicted[predicted_no_value] = np.nan
            
            color_tpye = 'red-magenta'

        # plot truth map
        fig_name = parameters['result_folder'] / f'{file_names_test[sample_id][:-4]}_map_truth.png'
        common.electroanatomical_map_on_voxel_cube(node, data_truth, color_tpye, fig_name)

        # plot predicted map
        error_mae = np.nanmean(np.abs(data_predicted - data_truth)) # mean absolute error on normalized activation time
        fig_name = parameters['result_folder'] / f'{file_names_test[sample_id][:-4]}_map_predict_MAE_{error_mae:.4f}.png'
        common.electroanatomical_map_on_voxel_cube(node, data_predicted, color_tpye, fig_name)
