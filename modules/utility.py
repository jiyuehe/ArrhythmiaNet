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

import numpy as np
import torch
import sys

try:
    import MinkowskiEngine as ME
except ImportError:
    ME = None

def load_npz_with_numpy2_pickle_compatibility(file_path, allow_pickle=True):
    # load npz and materialize arrays with NumPy 2.x -> 1.x pickle compatibility
    try:
        data = np.load(file_path, allow_pickle=allow_pickle)
        return {k: data[k] for k in data.files}
    except ModuleNotFoundError as e:
        if "numpy._core" not in str(e):
            raise

        # NumPy 2.x pickles may reference numpy._core.*; old NumPy exposes numpy.core.*
        sys.modules.setdefault('numpy._core', np.core)
        sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
        sys.modules.setdefault('numpy._core._multiarray_umath', np.core._multiarray_umath)

        data = np.load(file_path, allow_pickle=allow_pickle)
        return {k: data[k] for k in data.files}

def categorize_files_into_train_validation_test(data_folder_simulation):
    train_validation_test_file_names = 'train_validation_test_file_names.txt'
    split_file_path = data_folder_simulation / train_validation_test_file_names

    if not split_file_path.exists():
        # grab file names of the simulation results
        simulation_files = list(data_folder_simulation.glob('*_simulation_results_*.npz'))

        patient_files = {}
        for simulation_file in simulation_files:
            patient_id = simulation_file.name.split('_', 1)[0]
            patient_files.setdefault(patient_id, []).append(simulation_file.name)

        patient_ids = list(patient_files)
        n_patients = len(patient_ids)

        # randomly split patients so every simulation for a patient stays together
        perm = np.random.permutation(n_patients)

        n_train = int(0.8 * n_patients)
        n_val = int(0.15 * n_patients)

        patient_indices_train = perm[:n_train]
        patient_indices_validation = perm[n_train:n_train + n_val]
        patient_indices_test = perm[n_train + n_val:]

        # write a text file to save the file names of the training, validation and test data 
        with open(split_file_path, 'w') as f:
            f.write('[train]\n')
            for idx in patient_indices_train:
                for file_name in patient_files[patient_ids[idx]]:
                    f.write(f'{file_name}\n')
            f.write('[validation]\n')
            for idx in patient_indices_validation:
                for file_name in patient_files[patient_ids[idx]]:
                    f.write(f'{file_name}\n')
            f.write('[test]\n')
            for idx in patient_indices_test:
                for file_name in patient_files[patient_ids[idx]]:
                    f.write(f'{file_name}\n')

    # load the file names of the training, validation and test data from the text file
    with open(data_folder_simulation / train_validation_test_file_names, 'r') as f:
        lines = f.readlines()
        file_names_train = []
        file_names_validation = []
        file_names_test = []
        current_section = None
        for line in lines:
            line = line.strip()
            if line == '[train]':
                current_section = 'train'
            elif line == '[validation]':
                current_section = 'validation'
            elif line == '[test]':
                current_section = 'test'
            elif line and current_section is not None:
                if current_section == 'train':
                    file_names_train.append(line)
                elif current_section == 'validation':
                    file_names_validation.append(line)
                elif current_section == 'test':
                    file_names_test.append(line)

    return file_names_train, file_names_validation, file_names_test

def normalize_to_unit_interval(values):
    values = np.asarray(values)

    if values.ndim == 1:
        min_value = np.nanmin(values)
        max_value = np.nanmax(values)
        range_value = max_value - min_value

        if range_value == 0:
            return np.zeros_like(values, dtype=np.float32)

        return ((values - min_value) / range_value).astype(np.float32)

    if values.ndim == 2:
        # Normalize each node's time sequence independently (axis 0 = time).
        min_value = np.nanmin(values, axis=0, keepdims=True)
        max_value = np.nanmax(values, axis=0, keepdims=True)
        range_value = max_value - min_value

        zero_range_mask = (range_value == 0)
        safe_range = np.where(zero_range_mask, 1, range_value)
        normalized = (values - min_value) / safe_range
        normalized[:, zero_range_mask.squeeze(0)] = 0

        return normalized.astype(np.float32)

def load_input_and_target(start_idx, end_idx, file_names, name_prefixes, parameters, data_type, arrhythmia_flag):
    data_folder_simulation = parameters['data_folder_simulation'] 
    data_folder_processed_clinical = parameters['data_folder_processed_clinical']
    data_folder_mesh = parameters['data_folder_mesh']

    x_temp = [] # input. electrograms of all voxels
    nodes_temp = [] # input. xyz coordinates of the x_temp
    y_temp = [] # target
    for i in range(start_idx, end_idx):
        # load electrode coordinates
        # ------------------------------
        mesh_file_name = name_prefixes[i] + '_mesh.npz'
        mesh_data = load_npz_with_numpy2_pickle_compatibility(data_folder_mesh / mesh_file_name, allow_pickle=True)

        voxel = mesh_data['voxel']
        node = voxel - np.round(voxel.mean(axis=0)).astype(int) # center the coordinates at the origin and convert to integers. shape (n_nodes, 3)
        n_nodes = node.shape[0]

        b = i - start_idx # batch index, 1 batch corresponds to 1 simulation on 1 geometry
        batch_indices = torch.full((n_nodes, 1), b, dtype=torch.int32)
        temp = torch.cat([batch_indices, torch.from_numpy(node).int()], dim=1) # shape (n_nodes, 4) = [batch_indices | x | y | z]
        nodes_temp.append(temp)

        # load electrograms
        # ------------------------------
        if data_type == 0: # simulation
            simulation_results = dict(np.load(data_folder_simulation / file_names[i], allow_pickle=False))
        
            x_electrode = simulation_results['electrogram_unipolar'][parameters['t_start']:parameters['t_end']:parameters['time_step'], :] # shape (t, n_node)
            x_electrode = normalize_to_unit_interval(x_electrode)

            # none electrode voxel ids
            all_voxel_ids = np.arange(0,n_nodes)
            electrode_id_on_voxel = simulation_results['electrode_id_on_voxel']
            non_e_id = np.setdiff1d(all_voxel_ids, electrode_id_on_voxel)

            # electrograms on all nodes
            x = np.zeros((x_electrode.shape[0], n_nodes))
            x[:,electrode_id_on_voxel] = x_electrode
        elif data_type == 1: # clinical
            # load data
            clinical_data_file_name = name_prefixes[i] + '_clinical_data.npz'
            clinical_data = load_npz_with_numpy2_pickle_compatibility(data_folder_processed_clinical / clinical_data_file_name, allow_pickle=True)

            egm = clinical_data['electrogram_on_voxel'] # shape (n_node, t)
            egm = egm.T # shape (t, n_node)
            electrode_mask = np.isfinite(egm).all(axis=0)
            non_e_id = np.flatnonzero(~electrode_mask)
            x = np.zeros(egm.shape, dtype=np.float32)
            x[:, electrode_mask] = normalize_to_unit_interval(egm[:, electrode_mask])

        # add a binary row to indicate non-electrode nodes as a mask
        new_row = np.ones((1, x.shape[1]), dtype=np.float32)
        new_row[0, non_e_id] = 0
        x = np.concatenate((x, new_row), axis=0)
        
        x_temp.append(x)
        
        # load target data
        # ------------------------------
        if data_type == 0: # simulation
            if arrhythmia_flag == 0: # focal
                y = simulation_results['lat_voxel'].astype(np.float32)
                y = normalize_to_unit_interval(y)
            elif arrhythmia_flag == 1: # flutter
                y = simulation_results['lat_voxel_woi'].astype(np.float32)
                no_value_mask = ~np.isfinite(y)
                y_normalized = normalize_to_unit_interval(y)
                phase = 2 * np.pi * y_normalized
                y1 = np.sin(phase)
                y2 = np.cos(phase)
                y1 = np.where(no_value_mask, np.nan, y1)
                y2 = np.where(no_value_mask, np.nan, y2)
                y3 = no_value_mask.astype(np.float32)
                y = np.stack((y1, y2, y3), axis=1)
        elif data_type == 1: # clinical
            if arrhythmia_flag == 0: # focal
                y = clinical_data['activation_time_on_voxel']
                y = normalize_to_unit_interval(y)
            elif arrhythmia_flag == 1: # flutter
                y = clinical_data['activation_time_on_voxel']
                no_value_mask = ~np.isfinite(y)
                y_normalized = normalize_to_unit_interval(y)
                phase = 2 * np.pi * y_normalized
                y1 = np.sin(phase)
                y2 = np.cos(phase)
                y1 = np.where(no_value_mask, np.nan, y1)
                y2 = np.where(no_value_mask, np.nan, y2)
                y3 = no_value_mask.astype(np.float32)
                y = np.stack((y1, y2, y3), axis=1)

        y_temp.append(y)
            
    nodes_batch = torch.cat(nodes_temp, dim=0).to(parameters['device'])  # (batch * n_nodes, 4) = [batch_indices | x | y | z]

    # build feats_batch: (batch * n_nodes, t)
    feats_list = [torch.from_numpy(x).float().T for x in x_temp]  # each: (n_nodes, t)
    feats_batch = torch.cat(feats_list, dim=0).to(parameters['device'])

    # build targets_batch: (batch * n_nodes, n_output_channels)
    targets_list = [
        torch.from_numpy(y).float() if y.ndim == 2 else torch.from_numpy(y).float().reshape(-1, 1)
        for y in y_temp
    ]
    targets_batch = torch.cat(targets_list, dim=0).to(parameters['device'])

    # create MinkowskiEngine sparse tensor
    neural_network_input = ME.SparseTensor(features=feats_batch, coordinates=nodes_batch, device=parameters['device'])
    neural_network_target = ME.SparseTensor(features=targets_batch, coordinates=nodes_batch, device=parameters['device'])

    return neural_network_input, neural_network_target
