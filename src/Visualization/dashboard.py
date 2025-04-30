import base64
import io

import dash
import numpy as np
import torch
from dash import html, dcc, Output, Input
from matplotlib import pyplot as plt

from src.Framework.utils import normalize
from src.fcd.utils import load_data


def main(file: str):
    # Load and prepare data
    item = load_data(file, device=torch.device('cpu'))

    image_options = {
        't1w': normalize(item['t1w'])[0].rot90(k=3, dims=(1, 2)).numpy(),
        'FLAIR': normalize(item['FLAIR'])[0].rot90(k=3, dims=(1, 2)).numpy()
    }

    default_image_name = 't1w'
    default_image = image_options[default_image_name]
    max_slice = default_image.shape[0] - 1

    change_options = {
        't1w': np.random.randn(*image_options['t1w'].shape),
        'FLAIR': np.random.randn(*image_options['FLAIR'].shape),
    }

    target = normalize(item['roi'])[0].rot90(k=3, dims=(1, 2)).numpy()

    def get_encoded_slice(image_key: str, slice_index: int, show_change: bool, show_target: bool):
        image = image_options[image_key][slice_index, :, :]
        change = change_options[image_key][slice_index, :, :]
        target_slice = target[slice_index, :, :]

        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray', origin='lower')
        if show_change:
            ax.imshow(change, cmap='Reds', origin='lower', alpha=0.5)
        if show_target:
            ax.imshow(target_slice, cmap='Greens', origin='lower', alpha=0.5 * (target_slice > 0) )

        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"

    # Create Dash app
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("MRI Log Viewer"),

        html.Div([
            # Left column: Image
            html.Div([
                html.Img(id='mri-image', style={'width': '768px'})
            ], style={'flex': '1', 'padding': '10px'}),

            # Right column: Controls
            html.Div([
                html.H3("Select sequence:"),
                dcc.Dropdown(
                    id='image-dropdown',
                    options=[{'label': k, 'value': k} for k in image_options.keys()],
                    value=default_image_name,
                    clearable=False
                ),
                html.H3("Select slice:"),
                dcc.Slider(
                    id='slice-slider',
                    min=0,
                    max=max_slice,
                    step=1,
                    value=max_slice // 2,
                    marks={0: '0', max_slice: str(max_slice)},
                    tooltip={'placement': 'bottom'}
                ),
                html.Div([
                    dcc.Checklist(
                        id='checklist',
                        options=[{'label': 'change', 'value': 'change'},
                                 {'label': 'target', 'value': 'target'}],
                        value=[],  # empty = unchecked, ['change'] = checked
                        inline=False
                    )
                ], style={'margin-top': '20px'})
            ], style={'flex': '1', 'padding': '10px'})
        ], style={'display': 'flex'})
    ])

    # Unified callback for dropdown and slider
    @app.callback(
        Output('mri-image', 'src'),
        Input('image-dropdown', 'value'),
        Input('slice-slider', 'value'),
        Input('checklist', 'value'),
    )
    def update_image(image_key, slice_idx, checklist):
        return get_encoded_slice(image_key, slice_idx, 'change' in checklist, 'target' in checklist)

    app.run(debug=True)


if __name__ == '__main__':
    main("nnUNet/nnUNet_raw/Dataset101_fcd/sub-00003")
