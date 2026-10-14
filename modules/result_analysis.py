# Copyright 2026 Jiyue He
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

import os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
os.chdir(script_dir) # change the working directory
script_dir = Path(script_dir)

# add the workspace root to Python path
from platform import node
import modules.common as common
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.pyplot as plt 
import numpy as np

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

def scatter_or_voxel_plot(plot_scatter_voxel_flag, sparse_electrode_flag, node, e_id, non_e_id, voxels, grid_indices, converted_color, fig, ax):
    if plot_scatter_voxel_flag == 1:
        dpi = fig.dpi
        points_per_mm = dpi / 25.4
        if sparse_electrode_flag == 0:
            node_size = points_per_mm ** 2 * 2 * 2
        elif sparse_electrode_flag == 1:
            node_size = points_per_mm ** 2 * 2

        ax.scatter(node[non_e_id, 0], node[non_e_id, 1], node[non_e_id, 2], c='grey', s=1, edgecolor='none', linewidth=0, alpha=0.3)
        ax.scatter(node[e_id, 0], node[e_id, 1], node[e_id, 2], c=converted_color[e_id,:], edgecolor='none', linewidth=0, marker='s', s=node_size, depthshade=True)
    elif plot_scatter_voxel_flag == 0:
        voxel_color = np.zeros((*voxels.shape, 3))
        voxel_color[grid_indices[:,0], grid_indices[:,1], grid_indices[:,2]] = converted_color
        ax.voxels(voxels, facecolors=voxel_color, edgecolor=None, shade=False)

def plot_truth_and_predicted_map(truth_data, predicted_data, file_names_test, parameters, data_type):
    n_samples = len(predicted_data)

    for sample_id in np.arange(n_samples):
        print(f'plotting sample {sample_id+1}/{n_samples}')

        # load geometry data
        if data_type == 'simulation':
            name_prefix = file_names_test[sample_id].split('_simulation_results_')[0]
        elif data_type == 'clinical':
            name_prefix = file_names_test[sample_id].split('_clinical_data')[0]

        mesh_data = np.load(parameters['data_folder_patient'] / f'{name_prefix}_mesh.npz', allow_pickle=True)
        # node = mesh_data['voxel3mm_1mm_spacing']
        node = mesh_data['voxel']

        node = node - np.round(node.mean(axis=0)).astype(int) # shape (n_nodes, 3)
        n_nodes = node.shape[0]

        # electrode voxel ids
        if data_type == 'simulation':
            # randomly make some electrodes bad
            bad_electrode_ratio = 0
            n_bad_electrode = int(bad_electrode_ratio * n_nodes)
            non_e_id = np.random.choice(n_nodes, n_bad_electrode, replace=False)
        elif data_type == 'clinical':
            data = np.load(parameters['data_folder_patient'] / f'{name_prefix}_clinical_data.npz', allow_pickle=True)
            map_data = {k: data[k] for k in data.files}

            # find the good electrode nodes that have good signals
            act = map_data['clinical_activation_uni']
            good_e_id = [i for i, x in enumerate(act) if x != 0]
            non_e_id = np.setdiff1d(np.arange(n_nodes), good_e_id)

        e_id = np.arange(node.shape[0], dtype=np.int64)
        voxels, grid_indices = voxelize_nodes(node)

        data_truth = truth_data[sample_id].flatten()
        data_predicted = predicted_data[sample_id].flatten()

        # compute error
        error_mae = np.nanmean(np.abs(data_predicted - data_truth)) # mean absolute error on normalized activation time

        # plot truth map
        voxel_id_of_vertex1mm = mesh_data['voxel_id_of_vertex1mm']
        face = mesh_data['face1mm']
        vertex = mesh_data['vertex1mm']

        value_vertex = data_truth[voxel_id_of_vertex1mm]
        data = np.asarray(value_vertex, dtype=float)
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        data_threshold = data_min - 0.1

        vertex_color = common.convert_value_to_red_magenta(
            data, data_min, data_max, data_threshold
        )
        face_color = np.nanmean(vertex_color[face[:, :3]], axis=1)

        plt.figure()
        ax = plt.axes(projection='3d')
        mesh = Poly3DCollection(
            vertex[face],
            facecolors=face_color,
            edgecolors='gray',
            linewidths=0.1,
            alpha=1,
        )
        ax.add_collection3d(mesh)
        ax.auto_scale_xyz(vertex[:, 0], vertex[:, 1], vertex[:, 2])
        plt.axis('off')

        ax.view_init(elev=70, azim=-70)

        common.set_axes_equal(ax)
        plt.tight_layout()

        image_file_name = parameters['result_folder'] / f'{file_names_test[sample_id][:-4]}_map_truth.png'
        plt.savefig(image_file_name, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()
        common.crop_image(image_file_name)

        # plot predicted map
        value_vertex = data_predicted[voxel_id_of_vertex1mm]
        data = np.asarray(value_vertex, dtype=float)
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        data_threshold = data_min - 0.1

        vertex_color = common.convert_value_to_red_magenta(
            data, data_min, data_max, data_threshold
        )
        face_color = np.nanmean(vertex_color[face[:, :3]], axis=1)

        plt.figure()
        ax = plt.axes(projection='3d')

        # triangular mesh plot
        mesh = Poly3DCollection(
            vertex[face],
            facecolors=face_color,
            edgecolors='gray',
            linewidths=0.1,
            alpha=1,
        )
        ax.add_collection3d(mesh)
        ax.auto_scale_xyz(vertex[:, 0], vertex[:, 1], vertex[:, 2])

        # # scatter plot
        # ax.scatter(
        #     vertex[:, 0],
        #     vertex[:, 1],
        #     vertex[:, 2],
        #     c=vertex_color,
        #     s=2,
        #     linewidths=0
        # )

        plt.axis('off')

        ax.view_init(elev=70, azim=-70)

        common.set_axes_equal(ax)
        plt.tight_layout()

        image_file_name = parameters['result_folder'] / f'{file_names_test[sample_id][:-4]}_map_predict_MAE_{error_mae:.4f}.png'
        plt.savefig(image_file_name, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()
        common.crop_image(image_file_name)
