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
from scipy.interpolate import griddata
from scipy.spatial import QhullError

#%%
parameters = {}
parameters['data_folder_patient'] = Path('/home/j/Documents/box/carto3_files/data_npz/good')
parameters['result'] = Path(__file__).resolve().parent / 'result'
parameters['mesh_npz'] = Path('/home/j/Desktop/ssd/carto3_data/processed')
parameters['processed_clinical_data'] = parameters['result'] / 'processed_clinical_data'
parameters['processed_clinical_data'].mkdir(exist_ok=True)

clinical_file_paths = list(parameters['data_folder_patient'].glob('*_carto_catheter_processed.npz'))

# sort file names according to the patient id in front of the first '_'
clinical_file_paths.sort(key=lambda file: int(file.name.split('_', 1)[0]))

focal_patient_id = [99] # [10, 16, 40, 60, 89, 90, 91, 92, 94, 95, 99, 100, 101, 112]
flutter_patient_id = [107] # [19, 102, 107, 112, 114]

for data_id in range(len(clinical_file_paths)):
    file_path = clinical_file_paths[data_id]
    name_prefix = file_path.stem.replace('_carto_catheter_processed', '')
    patient_id = int(name_prefix.split('_', 1)[0])

    if patient_id in focal_patient_id:
        map_type = 'focal'
    elif patient_id in flutter_patient_id:
        map_type = 'flutter'

    output_path = parameters['processed_clinical_data'] / f'{name_prefix}_clinical_data.npz'

    if not output_path.exists() and ((patient_id in focal_patient_id) or (patient_id in flutter_patient_id)):
        print(f'{name_prefix}, {data_id} of [0, {len(clinical_file_paths)}]')
    else:
        continue

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

    # keep 50 ms centered on activation, with a 25 ms cosine taper on each side.
    distance_from_activation = np.abs(np.arange(n_time)[None, :] - activation_times[:, None])
    taper_fraction = np.clip((distance_from_activation - 25) / 25, 0, 1)
    window = 0.5 * (1 + np.cos(np.pi * taper_fraction))
    electrogram_qrs_subtracted_cleaned = electrogram_qrs_subtracted * window

    debug_plot = 1
    if debug_plot == 1:
        # plot some electrograms with their activation markers
        modules.clinical_data_plots.electrogram_and_activation(activation_times, electrogram, n_time, electrogram_qrs_subtracted_cleaned, file_path, parameters)

    debug_plot = 1
    if debug_plot == 1:
        # plot electrodes colored by activation time
        modules.clinical_data_plots.electrode_positions_and_activation_color(
            electrode_positions,
            activation_times,
            file_path,
            parameters,
            'electrode',
            map_type
        )

    # assign electrode data to atrial voxels
    file_path_3 = parameters['mesh_npz'] / f'{name_prefix}_mesh.npz'
    geometry = modules.utility.load_npz_with_numpy2_pickle_compatibility(file_path_3, allow_pickle=True)

    voxel = geometry['voxel']

    # for each voxel, find an electrode only if there is electrode(s) along the "geometry's surface normal of that voxel", and find the closest such electrode if there are many.
    n_voxel = voxel.shape[0]
    electrode_id_on_voxel = np.full(n_voxel, -1, dtype=np.int64)
    pair_distance_squared_on_voxel = np.full(n_voxel, np.inf, dtype=np.float64)
    activation_time_on_voxel = np.full(n_voxel, np.nan, dtype=np.float32)
    activation_time_on_voxel_interpolated = np.full(n_voxel, np.nan, dtype=np.float32)
    electrogram_on_voxel = np.full((n_voxel, n_time), np.nan, dtype=np.float32)
    electrogram_qrs_subtracted_cleaned_on_voxel = np.full((n_voxel, n_time), np.nan, dtype=np.float32)

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

    spacing = float(np.asarray(geometry.get('Delta', 1.0)).reshape(-1)[0])
    ray_tolerance = max(spacing, 1.0)
    normal_distance_threshold = 15.0
    electrode_positions = np.asarray(electrode_positions, dtype=np.float64)

    for voxel_id, (point, normal) in enumerate(zip(voxel, voxel_normals)):
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
        electrogram_qrs_subtracted_cleaned_on_voxel[voxel_id] = (
            electrogram_qrs_subtracted_cleaned[closest_id]
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
    electrogram_qrs_subtracted_cleaned_on_voxel[voxels_to_unassign] = np.nan

    debug_plot = 0
    if debug_plot == 1:
        # show electrodes in gray, the associated voxels in black, and draw a line linking the electrode-voxel pairs
        modules.clinical_data_plots.electrode_voxel_mapping(
            electrode_id_on_voxel,
            voxel,
            electrode_positions,
            parameters,
            file_path,
            geometry,
        )

    debug_plot = 0
    if debug_plot == 1:
        # show all voxel: electrodes in colors
        modules.clinical_data_plots.electrode_positions_and_activation_color(
            voxel,
            activation_time_on_voxel,
            file_path,
            parameters,
            'voxel',
            map_type
        )

    # interpolate from the valid electrode voxels to all voxels
    def interpolate_voxel_field(values):
        valid_source_voxels = np.flatnonzero(electrode_id_on_voxel >= 0)
        if valid_source_voxels.size == 0:
            return values

        source_values = values[valid_source_voxels]
        source_values = np.asarray(source_values)
        if source_values.ndim == 1:
            finite_source = np.isfinite(source_values)
        else:
            finite_source = np.isfinite(source_values).all(axis=1)
        valid_source_voxels = valid_source_voxels[finite_source]
        if valid_source_voxels.size == 0:
            return values

        source_points = voxel[valid_source_voxels]
        source_values = values[valid_source_voxels]

        try:
            linear_values = griddata(
                source_points,
                source_values,
                voxel,
                method='linear',
            )
        except (QhullError, ValueError):
            linear_values = None

        if valid_source_voxels.size < 4 or linear_values is None:
            return griddata(
                source_points,
                source_values,
                voxel,
                method='nearest',
            )

        linear_values = np.asarray(linear_values)
        if source_values.ndim == 1:
            missing = ~np.isfinite(linear_values)
            if np.any(missing):
                linear_values[missing] = griddata(
                    source_points,
                    source_values,
                    voxel[missing],
                    method='nearest',
                )
        else:
            missing = ~np.isfinite(linear_values).all(axis=1)
            if np.any(missing):
                linear_values[missing] = griddata(
                    source_points,
                    source_values,
                    voxel[missing],
                    method='nearest',
                )
        return linear_values

    activation_time_on_voxel_interpolated[:] = interpolate_voxel_field(
        activation_time_on_voxel
    )

    debug_plot = 1
    if debug_plot == 1:
        # show interpolated map on all voxels
        modules.clinical_data_plots.electrode_positions_and_activation_color(
            voxel,
            activation_time_on_voxel_interpolated,
            file_path,
            parameters,
            'voxel_interpolated',
            map_type
        )

    # save the variable
    processed_clinical_data = {}
    processed_clinical_data['voxel'] = voxel
    processed_clinical_data['activation_time_on_voxel'] = activation_time_on_voxel
    processed_clinical_data['activation_time_on_voxel_interpolated'] = activation_time_on_voxel_interpolated
    processed_clinical_data['electrogram_on_voxel'] = electrogram_on_voxel
    processed_clinical_data['electrogram_qrs_subtracted_cleaned_on_voxel'] = electrogram_qrs_subtracted_cleaned_on_voxel
    np.savez(output_path, **processed_clinical_data)

#%%
print('done')
