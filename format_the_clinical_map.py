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
from plotly.subplots import make_subplots
import modules

#%%
parameters = {}
parameters['data_folder_patient'] = Path('/home/j/Documents/box/carto3_files/data_npz/good')
parameters['result_folder'] = Path(__file__).resolve().parent / 'result'
parameters['result_folder'].mkdir(parents=True, exist_ok=True)

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
        # Plot three electrograms with their activation markers.
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
            parameters['result_folder'] / f'{figure_stem}_electrograms.html'
        )
        electrogram_fig.write_image(
            parameters['result_folder'] / f'{figure_stem}_electrograms.png',
            width=1200,
            height=300 * len(selected_ids),
        )

        # plot electrode positions colored by activation time
        valid_positions = np.isfinite(electrode_positions).all(axis=1)
        valid_activation = np.isfinite(activation_times) & valid_positions
        missing_activation = ~np.isfinite(activation_times) & valid_positions
        activation_fig = go.Figure()
        if valid_activation.any():
            activation_values = activation_times[valid_activation]
            data_min = activation_values.min()
            data_max = activation_values.max()

            activation_fig.add_trace(
                go.Scatter3d(
                    x=electrode_positions[valid_activation, 0],
                    y=electrode_positions[valid_activation, 1],
                    z=electrode_positions[valid_activation, 2],
                    mode='markers',
                    name='Activated electrode',
                    marker={
                        'size': 4,
                        'color': activation_values,
                        'colorscale': 'RdBu_r',
                        'cmin': data_min,
                        'cmax': data_max,
                        'colorbar': {'title': 'Activation (ms)'},
                    },
                    text=[f'Electrode {i}' for i in np.flatnonzero(valid_activation)],
                    hovertemplate='%{text}<br>Activation: %{marker.color:.2f} ms<extra></extra>',
                )
            )
        if missing_activation.any():
            activation_fig.add_trace(
                go.Scatter3d(
                    x=electrode_positions[missing_activation, 0],
                    y=electrode_positions[missing_activation, 1],
                    z=electrode_positions[missing_activation, 2],
                    mode='markers',
                    name='No activation time',
                    marker={'size': 4, 'color': 'lightgray'},
                    text=[f'Electrode {i}' for i in np.flatnonzero(missing_activation)],
                    hovertemplate='%{text}<br>No activation time<extra></extra>',
                )
            )

        activation_fig.update_layout(
            title=f'{file_path.name}: activation map',
            scene={
                'xaxis_title': 'X',
                'yaxis_title': 'Y',
                'zaxis_title': 'Z',
                'aspectmode': 'data',
            },
            template='plotly_white',
        )
        activation_fig.write_html(
            parameters['result_folder'] / f'{figure_stem}_activation_map.html'
        )
        activation_fig.write_image(
            parameters['result_folder'] / f'{figure_stem}_activation_map.png',
            width=1200,
            height=900,
        )


#%%
print('done')
