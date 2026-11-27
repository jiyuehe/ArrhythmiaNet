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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
from pathlib import Path

COMMON_DIR = Path('/home/j/Documents/box/codes/common')
sys.path.insert(0, str(COMMON_DIR))
import common

CAMERA_ELEV = np.deg2rad(70)
CAMERA_AZIM = np.deg2rad(-70)
CAMERA_EYE = {
    'x': 2.2 * np.cos(CAMERA_ELEV) * np.cos(CAMERA_AZIM),
    'y': 2.2 * np.cos(CAMERA_ELEV) * np.sin(CAMERA_AZIM),
    'z': 2.2 * np.sin(CAMERA_ELEV),
}

#%%
def electrogram_and_activation(activation_times, electrogram, n_time, electrogram_qrs_subtracted, file_path, parameters):
    # plot some electrograms with their activation markers
    valid_ids = np.flatnonzero(np.isfinite(activation_times))
    if valid_ids.size:
        selected_ids = valid_ids[np.linspace(0, valid_ids.size - 1, 3, dtype=int)]
    else:
        selected_ids = np.linspace(0, electrogram.shape[0] - 1, 3, dtype=int)

    electrogram_fig = make_subplots(
        rows=len(selected_ids),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f'Electrode {electrode_id}' for electrode_id in selected_ids],
        vertical_spacing=0.08,
    )
    time = np.arange(n_time)
    for row, electrode_id in enumerate(selected_ids):
        activation_time = activation_times[electrode_id]
        electrogram_fig.add_trace(
            go.Scatter(
                x=time,
                y=electrogram[electrode_id],
                name='Original',
                line={'color': 'royalblue'},
                legendgroup='original',
                showlegend=row == 0,
            ),
            row=row + 1,
            col=1,
        )
        electrogram_fig.add_trace(
            go.Scatter(
                x=time,
                y=electrogram_qrs_subtracted[electrode_id],
                name='QRS subtracted',
                line={'color': 'darkorange'},
                legendgroup='qrs_subtracted',
                showlegend=row == 0,
            ),
            row=row + 1,
            col=1,
        )
        if np.isfinite(activation_time):
            electrogram_fig.add_vline(
                x=activation_time,
                line={'color': 'red', 'dash': 'dash', 'width': 1},
                row=row + 1,
                col=1,
            )
        electrogram_fig.update_yaxes(title_text='Amplitude', row=row + 1, col=1)

    electrogram_fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            name='Activation',
            line={'color': 'red', 'dash': 'dash', 'width': 1},
        ),
        row=1,
        col=1,
    )
    electrogram_fig.update_xaxes(title_text='Time (ms)', row=len(selected_ids), col=1)
    electrogram_fig.update_xaxes(showgrid=True)
    electrogram_fig.update_yaxes(showgrid=True)
    electrogram_fig.update_layout(
        title=file_path.name,
        height=300 * len(selected_ids),
        template='plotly_white',
    )

    figure_stem = file_path.stem.replace('_processed', '')
    electrogram_fig.write_html(
        parameters['processed_clinical_data'] / f'{figure_stem}_electrograms.html'
    )
    electrogram_fig.write_image(
        parameters['processed_clinical_data'] / f'{figure_stem}_electrograms.png',
        width=1200,
        height=300 * len(selected_ids),
    )

def electrode_positions_and_activation_color(
    electrode_positions,
    activation_times,
    file_path,
    parameters,
    suffix,
    map_type
):
    # plot electrode positions colored by activation time
    valid_positions = np.isfinite(electrode_positions).all(axis=1)
    valid_activation = np.isfinite(activation_times) & valid_positions
    activation_fig = go.Figure()
    voxel_ids = np.flatnonzero(valid_positions)
    voxel_colors = np.full(voxel_ids.size, 'lightgray', dtype=object)
    voxel_text = [
        f'Electrode {electrode_id}<br>No activation time'
        for electrode_id in voxel_ids
    ]
    if valid_activation.any():
        activation_values = activation_times[valid_activation]
        data_min = activation_values.min()
        data_max = activation_values.max()
        color_max = data_max
        data_threshold = data_min - 0.01

        if map_type == 'focal':
            mapped_colors = common.convert_value_to_red_blue(
                activation_times,
                data_min,
                color_max,
                data_threshold,
            )
        elif map_type == 'flutter':
            mapped_colors = common.convert_value_to_red_magenta(
                activation_times,
                data_min,
                color_max,
                data_threshold,
            )

        marker_colors = [
            f'rgb({red},{green},{blue})'
            for red, green, blue in np.rint(
                mapped_colors[valid_activation] * 255
            ).astype(int)
        ]

        activation_ids = np.flatnonzero(valid_activation)
        positions_in_voxels = np.searchsorted(voxel_ids, activation_ids)
        voxel_colors[positions_in_voxels] = marker_colors
        for position, electrode_id, activation_time in zip(
            positions_in_voxels,
            activation_ids,
            activation_values,
        ):
            voxel_text[position] = (
                f'Electrode {electrode_id}<br>Activation: '
                f'{activation_time:.2f} ms'
            )

    if voxel_ids.size:
        # Plotly marker sizes are in screen pixels, so construct 1 mm cubes in data space.
        cube_corners = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ])
        cube_triangles = np.array([
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ])
        voxel_vertices = (
            electrode_positions[voxel_ids, None, :] + cube_corners[None, :, :]
        ).reshape(-1, 3)
        voxel_faces = (
            cube_triangles[None, :, :] + 8 * np.arange(voxel_ids.size)[:, None, None]
        ).reshape(-1, 3)
        activation_fig.add_trace(
            go.Mesh3d(
                x=voxel_vertices[:, 0],
                y=voxel_vertices[:, 1],
                z=voxel_vertices[:, 2],
                i=voxel_faces[:, 0],
                j=voxel_faces[:, 1],
                k=voxel_faces[:, 2],
                facecolor=np.repeat(voxel_colors, cube_triangles.shape[0]),
                flatshading=True,
                name='Electrode voxels',
                text=np.repeat(voxel_text, cube_corners.shape[0]),
                hovertemplate='%{text}<extra></extra>',
            )
        )

    activation_fig.update_layout(
        dragmode='orbit',
        showlegend=False,
        scene={
            'aspectmode': 'data',
            'camera': {'eye': CAMERA_EYE},
            'xaxis': {'visible': True, 'showgrid': True},
            'yaxis': {'visible': True, 'showgrid': True},
            'zaxis': {'visible': True, 'showgrid': True},
        },
        template='plotly_white',
    )
    figure_stem = file_path.stem.replace('_carto_catheter_processed', '')
    activation_fig.write_html(
        parameters['processed_clinical_data'] / f'{figure_stem}_activation_map_{suffix}.html'
    )
    activation_fig.write_image(
        parameters['processed_clinical_data'] / f'{figure_stem}_activation_map_{suffix}.png',
        width=1200,
        height=900,
    )
    common.crop_image(parameters['processed_clinical_data'] / f'{figure_stem}_activation_map_{suffix}.png')

def electrode_voxel_mapping(
    electrode_id_on_voxel,
    voxel_points,
    electrode_positions,
    parameters,
    file_path,
    geometry,
):
    # show electrodes in gray, the associated voxels in black, and draw a line linking the electrode-voxel pairs
    paired_voxel_ids = np.flatnonzero(electrode_id_on_voxel >= 0)
    paired_electrode_ids = electrode_id_on_voxel[paired_voxel_ids]

    valid_pairs = (
        np.isfinite(voxel_points[paired_voxel_ids]).all(axis=1)
        & np.isfinite(electrode_positions[paired_electrode_ids]).all(axis=1)
    )
    paired_voxel_ids = paired_voxel_ids[valid_pairs]
    paired_electrode_ids = paired_electrode_ids[valid_pairs]

    voxel_pair_points = voxel_points[paired_voxel_ids]
    electrode_pair_points = electrode_positions[paired_electrode_ids]

    mapping_fig = go.Figure()
    mesh_vertices = np.asarray(geometry['vertex'])
    mesh_faces = np.asarray(geometry['face'])
    mapping_fig.add_trace(
        go.Mesh3d(
            x=mesh_vertices[:, 0],
            y=mesh_vertices[:, 1],
            z=mesh_vertices[:, 2],
            i=mesh_faces[:, 0],
            j=mesh_faces[:, 1],
            k=mesh_faces[:, 2],
            name='Refined mesh',
            color='lightgray',
            opacity=0.3,
            hoverinfo='skip',
        )
    )

    valid_electrodes = np.isfinite(electrode_positions).all(axis=1)
    mapping_fig.add_trace(
        go.Scatter3d(
            x=electrode_positions[valid_electrodes, 0],
            y=electrode_positions[valid_electrodes, 1],
            z=electrode_positions[valid_electrodes, 2],
            mode='markers',
            name='Electrodes',
            marker={'size': 3, 'color': 'lightgray'},
            text=[f'Electrode {electrode_id}' for electrode_id in np.flatnonzero(valid_electrodes)],
            hovertemplate='%{text}<extra></extra>',
        )
    )

    if paired_voxel_ids.size:
        mapping_fig.add_trace(
            go.Scatter3d(
                x=voxel_pair_points[:, 0],
                y=voxel_pair_points[:, 1],
                z=voxel_pair_points[:, 2],
                mode='markers',
                name='Associated voxels',
                marker={'size': 3, 'color': 'black'},
                text=[
                    f'Voxel {voxel_id}<br>Electrode {electrode_id}'
                    for voxel_id, electrode_id in zip(
                        paired_voxel_ids,
                        paired_electrode_ids,
                    )
                ],
                hovertemplate='%{text}<extra></extra>',
            )
        )

        line_x = []
        line_y = []
        line_z = []
        for electrode_point, voxel_point in zip(
            electrode_pair_points,
            voxel_pair_points,
        ):
            line_x.extend([electrode_point[0], voxel_point[0], None])
            line_y.extend([electrode_point[1], voxel_point[1], None])
            line_z.extend([electrode_point[2], voxel_point[2], None])

        mapping_fig.add_trace(
            go.Scatter3d(
                x=line_x,
                y=line_y,
                z=line_z,
                mode='lines',
                name='Electrode-voxel links',
                line={'color': 'steelblue', 'width': 2},
                hoverinfo='skip',
            )
        )

    mapping_fig.update_layout(
        dragmode='orbit',
        showlegend=False,
        scene={
            'aspectmode': 'data',
            'camera': {'eye': CAMERA_EYE},
            'xaxis': {'visible': True, 'showgrid': True},
            'yaxis': {'visible': True, 'showgrid': True},
            'zaxis': {'visible': True, 'showgrid': True},
        },
        template='plotly_white',
    )

    figure_stem = file_path.stem.replace('_carto_catheter_processed', '')
    mapping_fig.write_html(
        parameters['processed_clinical_data'] / f'{figure_stem}_electrode_voxel_mapping.html'
    )
    mapping_fig.write_image(
        parameters['processed_clinical_data'] / f'{figure_stem}_electrode_voxel_mapping.png',
        width=1200,
        height=900,
    )
    common.crop_image(parameters['processed_clinical_data'] / f'{figure_stem}_electrode_voxel_mapping.png')
