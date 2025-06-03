import base64
import glob
import io
import json
import os.path
from typing import Dict

import dash
import numpy as np
from dash import html, dcc, Output, Input, ctx
from matplotlib import pyplot as plt
from monai.transforms import LoadImaged, Compose, ResampleToMatchd

from src.Architecture.CustomTransforms import AddMissingd


def load(path: str):
    with open(f'{path}/logs.json') as f:
        item = json.load(f)

    loader = Compose([
        LoadImaged(keys=['t1w', 'FLAIR', 'target'],
                   reader="NibabelReader",
                   ensure_channel_first=True,
                   allow_missing_keys=True),
        AddMissingd(keys=['t1w', 'FLAIR', 'target'], key_add='target', ref='FLAIR'),
        ResampleToMatchd(keys=['t1w', 'FLAIR'], key_dst='target')])

    item = loader(item)

    images: Dict[str, np.ndarray] = {key: value[0].numpy() for key, value
                                       in item.items() if key != 'target'}
    target: np.ndarray = item['target'][0].numpy()

    changes: Dict[int, Dict[str, np.ndarray]] = {}
    for file in glob.glob(f'{path}/*.npz'):
        nr = int(os.path.basename(file)[7:-4])
        changes[nr] = np.load(file)

    preds: Dict[int, np.ndarray] = {}
    for file in glob.glob(f'{path}/*.npy'):
        nr = int(os.path.basename(file)[5:-4])
        preds[nr] = np.load(file)

    return images, target, changes, preds


def main(file: str):
    # Load and prepare data
    image_options, target, change_options, predictions = load(file)

    default_image_name = 't1w'
    default_image = image_options[default_image_name]
    max_slice = default_image.shape[0] - 1
    max_step = max(change_options.keys())

    orientation_to_dim: Dict[str, int] = {
        'sagittal-btn': 0,
        'axial-btn': 2,
        'coronal-btn': 1
    }

    rotation: Dict[str, int] = {
        'sagittal-btn': 3,
        'axial-btn': 1,
        'coronal-btn': 3
    }

    def get_encoded_slice(image_key: str,
                          slice_index: int,
                          orientation: str,
                          step: int,
                          show_change: bool,
                          show_target: bool,
                          show_pred: bool,
                          apply_change: bool):
        dim = orientation_to_dim.get(orientation, 0)
        rot = rotation.get(orientation, 0)
        image = np.rot90(image_options[image_key].take([slice_index], axis=dim).squeeze(axis=dim), k=rot)

        fig, ax = plt.subplots()

        if apply_change:
            ax.imshow(image + np.rot90(change_options[step][image_key].take([slice_index], axis=dim).squeeze(axis=dim), k=rot),
                      cmap='gray',
                      origin='lower')
        else:
            ax.imshow(image, cmap='gray', origin='lower')

        if show_change:
            change = np.rot90(change_options[step][image_key].take([slice_index], axis=dim).squeeze(axis=dim), k=rot)
            ax.imshow(change, cmap='RdBu', origin='lower', alpha=0.5)

        if show_target:
            target_slice = np.rot90(target.take([slice_index], axis=dim).squeeze(axis=dim), k=rot)
            ax.imshow(target_slice, cmap='Greens', origin='lower', alpha=0.5 * (target_slice > 0))

        if show_pred:
            pred_slice = np.rot90(predictions[step].take([slice_index], axis=dim).squeeze(axis=dim), k=rot)
            ax.imshow(pred_slice, cmap='Reds', origin='lower', alpha=0.5 * (pred_slice > 0))

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
                html.Div([
                    html.Button("axial", id='axial-btn', n_clicks=0, style={'margin': '10px'}),
                    html.Button("sagittal", id='sagittal-btn', n_clicks=0, style={'margin': '10px'}),
                    html.Button("coronal", id='coronal-btn', n_clicks=0, style={'margin': '10px'}),
                ], style={'display': 'flex', 'gap': '10px', 'marginTop': '10px'}),
                dcc.Store(id='orientation'),
                html.Img(id='mri-image', style={'height': '1024px'})
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
                html.H3("Select step:"),
                dcc.Slider(
                    id='step-slider',
                    min=0,
                    max=max_step,
                    step=1,
                    value=max_step,
                    marks={0: '0', max_step: str(max_step)},
                    tooltip={'placement': 'bottom'}
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
                                 {'label': 'apply change', 'value': 'apply'},
                                 {'label': 'target', 'value': 'target'},
                                 {'label': 'prediction', 'value': 'pred'},],
                        value=[],  # empty = unchecked, ['change'] = checked
                        inline=False
                    )
                ], style={'margin-top': '20px'})
            ], style={'flex': '1', 'padding': '10px'})
        ], style={'display': 'flex'})
    ])

    @app.callback(
        Output('orientation', 'data'),
        Output('sagittal-btn', 'style'),
        Output('axial-btn', 'style'),
        Output('coronal-btn', 'style'),
        Input('axial-btn', 'n_clicks'),
        Input('sagittal-btn', 'n_clicks'),
        Input('coronal-btn', 'n_clicks'),
        prevent_initial_call='initial_duplicate'
    )
    def store_last_clicked(n1, n2, n3):
        triggered_id = ctx.triggered_id if ctx.triggered_id is not None else 'axial-btn'

        def style(btn_id):
            if triggered_id == btn_id:
                return {'backgroundColor': 'red', 'margin': '5px'}
            return {'backgroundColor': 'white', 'margin': '5px'}

        return triggered_id, style('sagittal-btn'), style('axial-btn'), style('coronal-btn')

    @app.callback(
        Output('mri-image', 'src'),
        Input('orientation', 'data'),
        Input('image-dropdown', 'value'),
        Input('step-slider', 'value'),
        Input('slice-slider', 'value'),
        Input('checklist', 'value'),
    )
    def update_image(orientation, image_key, step, slice_idx, checklist):
        return get_encoded_slice(
            image_key=image_key,
            slice_index=slice_idx,
            orientation=orientation,
            step=step,
            show_change='change' in checklist,
            show_target='target' in checklist,
            show_pred='pred' in checklist,
            apply_change='apply' in checklist
        )

    app.run(debug=True)


if __name__ == '__main__':
    main("logs/test")
