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
import plotly.graph_objects as go
import modules

#%%
parameters = {}
parameters['data_folder_patient'] = Path('/home/j/Documents/box/carto3_files/data_npz/good')
parameters['result'] = Path(__file__).resolve().parent / 'result'
parameters['mesh_npz'] = Path('/home/j/Desktop/ssd/carto3_data/processed')

clinical_file_paths = list(parameters['data_folder_patient'].glob('*_carto_catheter_processed.npz'))

# sort file names according to the patient id in front of the first '_'
clinical_file_paths.sort(key=lambda file: int(file.name.split('_', 1)[0]))

for data_id in [0]: #range(len(clinical_file_paths)):
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

    # remove those nan activation_times, and the associated electrogram, electrogram_qrs_subtracted, electrode_positions
    valid_activation_times = ~np.isnan(activation_times)
    activation_times = activation_times[valid_activation_times]
    electrogram = electrogram[valid_activation_times]
    electrogram_qrs_subtracted = electrogram_qrs_subtracted[valid_activation_times]
    electrode_positions = electrode_positions[valid_activation_times]

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

    # for each voxel, find an electrode only if there is electrode(s) along the "geometry's surface normal of that voxel", and find the closest such electrode if there are many.
    n_voxel = voxel.shape[0]
    electrode_id_on_voxel = np.full(n_voxel, -1, dtype=np.int64)
    pair_distance_squared_on_voxel = np.full(n_voxel, np.inf, dtype=np.float64)
    activation_time_on_voxel = np.full(n_voxel, np.nan, dtype=np.float32)
    electrogram_on_voxel = np.full((n_voxel, n_time), np.nan, dtype=np.float32)
    electrogram_qrs_subtracted_on_voxel = np.full((n_voxel, n_time), np.nan, dtype=np.float32)

    # Use face normals from the refined geometry when available.  Each voxel
    # is matched to the refined vertex stored in vertex_id_of_voxel.
    voxel_normals = np.full((n_voxel, 3), np.nan, dtype=np.float64)
    # if 'vertex' in geometry and 'face' in geometry and 'vertex_id_of_voxel' in geometry:
    vertex = np.asarray(geometry['vertex'], dtype=np.float64)
    face = np.asarray(geometry['face'], dtype=np.int64)
    vertex_id_of_voxel = np.asarray(geometry['vertex_id_of_voxel'], dtype=np.int64)

    face_vertices = vertex[face]
    face_normals = np.cross(
        face_vertices[:, 1] - face_vertices[:, 0],
        face_vertices[:, 2] - face_vertices[:, 0],
    )

    vertex_normals = np.zeros_like(vertex)
    np.add.at(vertex_normals, face[:, 0], face_normals)
    np.add.at(vertex_normals, face[:, 1], face_normals)
    np.add.at(vertex_normals, face[:, 2], face_normals)

    normal_lengths = np.linalg.norm(vertex_normals, axis=1)
    valid_normals = normal_lengths > 0
    vertex_normals[valid_normals] /= normal_lengths[valid_normals, None]

    valid_voxel_vertices = (
        (vertex_id_of_voxel >= 0)
        & (vertex_id_of_voxel < vertex.shape[0])
    )
    voxel_normals[valid_voxel_vertices] = (
        vertex_normals[vertex_id_of_voxel[valid_voxel_vertices]]
    )

    voxel_points = np.asarray(
        [voxel[id] for id in range(n_voxel)],
        dtype=np.float64,
    )
    # if 'vertex' in geometry and 'vertex_id_of_voxel' in geometry:
    vertex = np.asarray(geometry['vertex'], dtype=np.float64)
    vertex_id_of_voxel = np.asarray(geometry['vertex_id_of_voxel'], dtype=np.int64)
    valid_voxel_vertices = (
        (vertex_id_of_voxel >= 0)
        & (vertex_id_of_voxel < vertex.shape[0])
    )
    voxel_points[valid_voxel_vertices] = (
        vertex[vertex_id_of_voxel[valid_voxel_vertices]]
    )

    spacing = float(np.asarray(geometry.get('Delta', 1.0)).reshape(-1)[0])
    ray_tolerance = max(spacing, 1.0)
    normal_distance_threshold = 15.0
    electrode_positions = np.asarray(electrode_positions, dtype=np.float64)

    for voxel_id, (point, normal) in enumerate(zip(voxel_points, voxel_normals)):
        if not np.isfinite(normal).all():
            continue

        normal_length = np.linalg.norm(normal)
        if normal_length == 0:
            continue
        normal = normal / normal_length

        displacement = electrode_positions - point
        longitudinal = displacement @ normal
        perpendicular = np.linalg.norm(
            displacement - longitudinal[:, None] * normal[None, :],
            axis=1,
        )

        candidate_ids = np.flatnonzero(
            (perpendicular <= ray_tolerance)
            & (np.abs(longitudinal) <= normal_distance_threshold)
        )
        if candidate_ids.size == 0:
            continue

        closest_id = candidate_ids[
            np.argmin(perpendicular[candidate_ids] ** 2 + longitudinal[candidate_ids] ** 2)
        ]
        electrode_id_on_voxel[voxel_id] = closest_id
        pair_distance_squared_on_voxel[voxel_id] = (
            perpendicular[closest_id] ** 2 + longitudinal[closest_id] ** 2
        )
        activation_time_on_voxel[voxel_id] = activation_times[closest_id]
        electrogram_on_voxel[voxel_id] = electrogram[closest_id]
        electrogram_qrs_subtracted_on_voxel[voxel_id] = (
            electrogram_qrs_subtracted[closest_id]
        )

    assigned_voxels = np.flatnonzero(electrode_id_on_voxel >= 0)
    ordered_voxels = assigned_voxels[
        np.lexsort((
            assigned_voxels,
            pair_distance_squared_on_voxel[assigned_voxels],
            electrode_id_on_voxel[assigned_voxels],
        ))
    ]
    ordered_electrode_ids = electrode_id_on_voxel[ordered_voxels]
    duplicate_electrode_assignments = np.zeros(ordered_voxels.size, dtype=bool)
    duplicate_electrode_assignments[1:] = (
        ordered_electrode_ids[1:] == ordered_electrode_ids[:-1]
    )
    voxels_to_unassign = ordered_voxels[duplicate_electrode_assignments]
    electrode_id_on_voxel[voxels_to_unassign] = -1
    pair_distance_squared_on_voxel[voxels_to_unassign] = np.inf
    activation_time_on_voxel[voxels_to_unassign] = np.nan
    electrogram_on_voxel[voxels_to_unassign] = np.nan
    electrogram_qrs_subtracted_on_voxel[voxels_to_unassign] = np.nan
#%%
    debug_plot = 0
    if debug_plot == 1:
        # show electrodes in gray, the associated voxels in black, and draw a line linking the electrode-voxel pairs
        modules.clinical_data_plots.electrode_voxel_mapping(
            electrode_id_on_voxel,
            voxel_points,
            electrode_positions,
            parameters,
            file_path,
            geometry,
        )

#%%
print('done')
