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

def save_interactive_mesh(voxel, lat_voxel, fig_name):
    data = lat_voxel
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    data_threshold = data_min-0.1 # a little small than data_min, so that places with value of data_min will have color
    color = common.convert_value_to_red_blue(data, data_min, data_max, data_threshold)
    
    # Find the grid spacing so adjacent cubes meet without gaps.
    axis_steps = [
        np.diff(np.unique(voxel[:, axis]))
        for axis in range(3)
    ]
    positive_steps = [steps[steps > 0] for steps in axis_steps if steps.size]
    spacing = min(np.min(steps) for steps in positive_steps) if positive_steps else 1.0

    origin = np.min(voxel, axis=0)
    voxel_index = np.rint((voxel - origin) / spacing).astype(int)
    grid_shape = tuple(np.max(voxel_index, axis=0) + 1)

    # One collection is much cheaper to draw than the separate collection
    # ax.voxels creates for each cube. Include only faces with no neighbour.
    filled = np.zeros(np.asarray(grid_shape) + 2, dtype=bool)
    padded_index = voxel_index + 1
    filled[tuple(padded_index.T)] = True
    face_offsets = np.array([
        [[-.5, -.5, -.5], [-.5, -.5, .5], [-.5, .5, .5], [-.5, .5, -.5]],
        [[.5, -.5, -.5], [.5, .5, -.5], [.5, .5, .5], [.5, -.5, .5]],
        [[-.5, -.5, -.5], [.5, -.5, -.5], [.5, -.5, .5], [-.5, -.5, .5]],
        [[-.5, .5, -.5], [-.5, .5, .5], [.5, .5, .5], [.5, .5, -.5]],
        [[-.5, -.5, -.5], [-.5, .5, -.5], [.5, .5, -.5], [.5, -.5, -.5]],
        [[-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]],
    ])
    neighbours = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0],
                           [0, 1, 0], [0, 0, -1], [0, 0, 1]])
    faces = []
    face_colors = []
    for offset, corners in zip(neighbours, face_offsets):
        adjacent = padded_index + offset
        visible = ~filled[tuple(adjacent.T)]
        if np.any(visible):
            centers = origin + voxel_index[visible] * spacing
            faces.append(centers[:, None, :] + corners[None, :, :] * spacing)
            face_colors.append(color[visible])

    surface_faces = np.concatenate(faces)
    surface_colors = np.concatenate(face_colors)

    vertices = surface_faces.reshape(-1, 3)
    quad_indices = np.arange(len(surface_faces) * 4).reshape(-1, 4)
    triangle_colors = np.repeat(surface_colors, 2, axis=0)
    triangle_colors = [
        f'rgb({r}, {g}, {b})'
        for r, g, b in np.rint(triangle_colors * 255).astype(int)
    ]
    fig = go.Figure(data=go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=np.column_stack((quad_indices[:, 0], quad_indices[:, 0])).ravel(),
        j=np.column_stack((quad_indices[:, 1], quad_indices[:, 2])).ravel(),
        k=np.column_stack((quad_indices[:, 2], quad_indices[:, 3])).ravel(),
        facecolor=triangle_colors,
        flatshading=True,
        showscale=False,
    ))
    lower = origin - spacing / 2
    upper = origin + (np.asarray(grid_shape) - .5) * spacing

    elev, azim = 70, -70

    elev = np.deg2rad(elev)
    azim = np.deg2rad(azim)
    camera_eye = dict(
        x=2.2 * np.cos(elev) * np.cos(azim),
        y=2.2 * np.cos(elev) * np.sin(azim),
        z=2.2 * np.sin(elev),
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[lower[0], upper[0]]),
            yaxis=dict(visible=False, range=[lower[1], upper[1]]),
            zaxis=dict(visible=False, range=[lower[2], upper[2]]),
            aspectmode='data',
            dragmode='orbit',
            camera=dict(
                eye=camera_eye,
                up=dict(x=0, y=0, z=1),
            ),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    output_path = Path(fig_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path.with_suffix(".html"), include_plotlyjs=True)
    fig.write_image(output_path, scale=2)

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
        node = mesh_data['voxel']
        node = node - np.round(node.mean(axis=0)).astype(int) # shape (n_nodes, 3)
        n_nodes = node.shape[0]

        # electrode voxel ids
        if data_type == 'simulation':
            pass
            # # randomly make some electrodes bad
            # bad_electrode_ratio = 0
            # n_bad_electrode = int(bad_electrode_ratio * n_nodes)
            # non_e_id = np.random.choice(n_nodes, n_bad_electrode, replace=False)
        elif data_type == 'clinical':
            data = np.load(parameters['data_folder_patient'] / f'{name_prefix}_clinical_data.npz', allow_pickle=True)
            map_data = {k: data[k] for k in data.files}

            # find the good electrode nodes that have good signals
            act = map_data['clinical_activation_uni']
            good_e_id = [i for i, x in enumerate(act) if x != 0]
            non_e_id = np.setdiff1d(np.arange(n_nodes), good_e_id)

        data_truth = truth_data[sample_id].flatten()
        data_predicted = predicted_data[sample_id].flatten()

        # plot truth map
        fig_name = parameters['result_folder'] / f'{file_names_test[sample_id][:-4]}_map_truth.png'
        save_interactive_mesh(node, data_truth, fig_name)

        # plot predicted map
        error_mae = np.nanmean(np.abs(data_predicted - data_truth)) # mean absolute error on normalized activation time
        fig_name = parameters['result_folder'] / f'{file_names_test[sample_id][:-4]}_map_predict_MAE_{error_mae:.4f}.png'
        save_interactive_mesh(node, data_predicted, fig_name)
