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
from pathlib import Path
import numpy as np
import modules

#%%
parameters = {}
parameters['data_folder_patient'] = Path('/home/j/Documents/box/carto3_files/data_npz/good')
parameters['result'] = Path(__file__).resolve().parent / 'result'
parameters['mesh_npz'] = Path('/home/j/Desktop/ssd/carto3_data/processed')

clinical_file_paths = list(parameters['data_folder_patient'].glob('*_carto_catheter_processed.npz'))

# sort file names according to the patient id in front of the first '_'
clinical_file_paths.sort(key=lambda file: int(file.name.split('_', 1)[0]))

for data_id in range(len(clinical_file_paths)):
    file_path = clinical_file_paths[data_id]
    clinical_data_processed = modules.utility.load_npz_with_numpy2_pickle_compatibility(file_path, allow_pickle=True)

    file_path_2 = file_path.with_name(file_path.name.replace('_processed', ''))
    clinical_data_original = modules.utility.load_npz_with_numpy2_pickle_compatibility(file_path_2, allow_pickle=True)

    # electrogram window of interest
    woi_start = clinical_data_processed['clinical_electrogram_woi_start'] # a number
    woi_end = woi_start + 500 # unit: ms

    # electrogram 
    electrogram_qrs_subtracted = clinical_data_processed['mapping_electrogram_unipolar_qrs_subtracted'][:,woi_start:woi_end,:] # (n_segment, woi_start:woi_end, n_electrode)

    electrogram = clinical_data_original['mapping_electrogram_unipolar'][:,woi_start:woi_end,:]

    # activation times
    activation_times = np.asarray(
        clinical_data_processed['mapping_electrogram_unipolar_activation_within_woi'].tolist()
    ).reshape(-1) - woi_start # (total_electrodes,) 
    activation_times = np.where(activation_times < 0, np.nan, activation_times)

    # electrode positions
    electrode_positions = clinical_data_processed['electrode_positions'] # (n_segment, n_electrode, 3)

    n_segment, n_time, n_electrode = electrogram_qrs_subtracted.shape
    electrogram_qrs_subtracted = electrogram_qrs_subtracted.transpose(0, 2, 1).reshape(-1, n_time) # (total_electrodes, n_time)
    electrogram = electrogram.transpose(0, 2, 1).reshape(-1, n_time) # (total_electrodes, n_time)
    electrode_positions = electrode_positions.reshape(-1, 3) # (total_electrodes, 3)

    debug_plot = 0
    if debug_plot == 1:
        # plot some electrograms with their activation markers
        modules.clinical_data_plots.electrogram_and_activation(activation_times, electrogram, n_time, electrogram_qrs_subtracted, file_path, parameters)

        # plot electrode positions colored by activation time
        modules.clinical_data_plots.electrode_positions_and_activation_color(electrode_positions, activation_times, file_path, parameters)

    # assign electrode data to atrial voxels
    name_prefix = file_path.stem.replace('_carto_catheter_processed', '')
    file_path_3 = parameters['mesh_npz'] / f'{name_prefix}_mesh.npz'
    geometry = modules.utility.load_npz_with_numpy2_pickle_compatibility(file_path_3, allow_pickle=True)

    voxel = geometry['voxel']
    

#%%
print('done')
