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
        parameters['result'] / f'{figure_stem}_electrograms.html'
    )
    electrogram_fig.write_image(
        parameters['result'] / f'{figure_stem}_electrograms.png',
        width=1200,
        height=300 * len(selected_ids),
    )

def electrode_positions_and_activation_color(electrode_positions, activation_times, file_path, parameters):
    # plot electrode positions colored by activation time
    valid_positions = np.isfinite(electrode_positions).all(axis=1)
    valid_activation = np.isfinite(activation_times) & valid_positions
    missing_activation = ~np.isfinite(activation_times) & valid_positions
    activation_fig = go.Figure()
    if valid_activation.any():
        activation_values = activation_times[valid_activation]
        data_min = activation_values.min()
        data_max = activation_values.max()
        color_max = data_max if data_max > data_min else data_min + 1
        data_threshold = data_min - 0.01

        mapped_colors = common.convert_value_to_red_blue(
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

        activation_fig.add_trace(
            go.Scatter3d(
                x=electrode_positions[valid_activation, 0],
                y=electrode_positions[valid_activation, 1],
                z=electrode_positions[valid_activation, 2],
                mode='markers',
                name='Activated electrode',
                marker={
                    'size': 4,
                    'color': marker_colors,
                },
                text=[f'Electrode {i}' for i in np.flatnonzero(valid_activation)],
                customdata=activation_values,
                hovertemplate='%{text}<br>Activation: %{customdata:.2f} ms<extra></extra>',
            )
        )

        colorbar_values = np.linspace(data_min, color_max, 256)
        colorbar_rgb = common.convert_value_to_red_blue(
            colorbar_values,
            data_min,
            color_max,
            data_threshold,
        )
        colorscale = [
            [
                index / (len(colorbar_rgb) - 1),
                f'rgb({red},{green},{blue})',
            ]
            for index, (red, green, blue) in enumerate(
                np.rint(colorbar_rgb * 255).astype(int)
            )
        ]
        activation_fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode='markers',
                showlegend=False,
                hoverinfo='skip',
                marker={
                    'size': 0,
                    'color': [data_min],
                    'colorscale': colorscale,
                    'cmin': data_min,
                    'cmax': color_max,
                    'showscale': True,
                    'colorbar': {'title': 'Activation (ms)'},
                },
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
                marker={'size': 2, 'color': 'lightgray'},
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
    figure_stem = file_path.stem.replace('_processed', '')
    activation_fig.write_html(
        parameters['result'] / f'{figure_stem}_activation_map.html'
    )
    activation_fig.write_image(
        parameters['result'] / f'{figure_stem}_activation_map.png',
        width=1200,
        height=900,
    )

