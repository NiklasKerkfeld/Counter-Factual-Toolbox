import base64
import io
import os
from typing import Dict

import dash
from dash import html, Input, Output, callback, ALL, dcc, ctx, MATCH, State
from matplotlib import pyplot as plt

from src.Visualization.Dashboard.load import images, dataset_folders

dash.register_page(__name__, path='/src/Visualization/Dashboard/pages/image')

# Create image blocks
images_blocks = [
    html.Div([
        html.H3(folder),
        html.Div([
            html.Div([
                html.Div([
                    html.Button("axial",
                                id={'type': 'orientation-btn', 'index': folder,
                                    'view': 'axial'},
                                style={'backgroundColor': 'gray', 'margin': '5px'}),
                    html.Button("sagittal",
                                id={'type': 'orientation-btn', 'index': folder,
                                    'view': 'sagittal'},
                                style={'backgroundColor': 'gray', 'margin': '5px'}),
                    html.Button("coronal",
                                id={'type': 'orientation-btn', 'index': folder,
                                    'view': 'coronal'},
                                style={'backgroundColor': 'gray', 'margin': '5px'}),
                ], style={'display': 'flex', 'gap': '10px', 'marginTop': '10px'}),
                html.Img(id={'type': 'mri-image', 'index': folder},
                         style={'width': '512px'}),
                dcc.Store(id={'type': 'orientation-store', 'index': folder},
                          data='axial')
            ], style={'padding': '10px'}),

            html.Div([
                html.H3("Select sequence:"),
                dcc.Dropdown(
                    id={'type': 'image-dropdown', 'index': folder},
                    options=[{'label': k, 'value': k} for k in imgs.keys() if k != 'target'],
                    value=list(imgs.keys())[0],
                    clearable=False
                ),
                html.H3("Select step:"),
                dcc.Slider(
                    id={'type': 'step-slider', 'index': folder},
                    min=0,
                    max=10,
                    step=1,
                    value=10,
                    marks={0: '0', 10: str(10)},
                    tooltip={'placement': 'bottom'}
                ),
                html.H3("Select slice:"),
                dcc.Slider(
                    id={'type': 'slice-slider', 'index': folder},
                    min=0,
                    max=160,
                    step=1,
                    value=160 // 2,
                    marks={0: '0', 160: str(160)},
                    tooltip={'placement': 'bottom'}
                ),
                html.Div([
                    dcc.Checklist(
                        id='checklist',
                        options=[{'label': 'change', 'value': 'change'},
                                 {'label': 'apply change', 'value': 'apply'},
                                 {'label': 'target', 'value': 'target'},
                                 {'label': 'prediction', 'value': 'pred'}, ],
                        value=[],  # empty = unchecked, ['change'] = checked
                        inline=False
                    )
                ], style={'margin-top': '20px'})
            ], style={'flex': '1', 'padding': '10px'})
        ], style={'display': 'flex', 'gap': '40px'})
    ],
        id={'type': 'image-block', 'index': folder},
        style={'display': 'none'}
    ) for folder, imgs in images.items()
]

layout = html.Div([
    html.H1("Images"),
    html.Div([
        *images_blocks]
    )
])


@callback(
    Output({'type': 'orientation-store', 'index': MATCH}, 'data'),
    Output({'type': 'orientation-btn', 'index': MATCH, 'view': ALL}, 'style'),
    Input({'type': 'orientation-btn', 'index': MATCH, 'view': ALL}, 'n_clicks'),
    State({'type': 'orientation-btn', 'index': MATCH, 'view': ALL}, 'id'),
    prevent_initial_call=True
)
def orientation_buttons(n_clicks, btn_ids):
    if not any(n_clicks):
        raise dash.exceptions.PreventUpdate

    triggered = ctx.triggered_id
    selected_view = triggered['view'] if triggered else 'axial'

    styles = []
    for btn_id in btn_ids:
        styles.append({
            'backgroundColor': 'red' if btn_id['view'] == selected_view else 'gray',
            'margin': '5px'
        })

    return selected_view, styles


# image selection
@callback(
    Output({'type': 'image-block', 'index': ALL}, 'style'),
    Input('shared-data-selection', 'data'),
    prevent_initial_call=True
)
def image_visibility(data):
    selected = data.get('datasets', []) if data else []
    return [
        {'display': 'block'} if dataset in selected else {'display': 'none'}
        for dataset in dataset_folders
    ]


# get image
@callback(
    Output({'type': 'mri-image', 'index': ALL}, 'src'),
    State('dataset-store', 'data'),
    Input('shared-data-selection', 'data'),
    Input({'type': 'image-dropdown', 'index': ALL}, 'value'),
    Input({'type': 'slice-slider', 'index': ALL}, 'value'),
    Input({'type': 'orientation-store', 'index': ALL}, 'data'),
    prevent_initial_call=True
)
def show_selected_images(dataset_last_selected, data, sequences, steps, orientations):
    if ctx.triggered_id == 'shared-data-selection':
        update = dataset_last_selected
    else:
        update = ctx.triggered_id['index']

    updated_srcs = []
    for dataset_name, sequence, orientation, step in zip(dataset_folders, sequences, orientations, steps):
        if dataset_name == update:
            updated_srcs.append(get_image(dataset_name,
                                          sequence=sequence,
                                          orientation=orientation,
                                          slice=step))
        else:
            updated_srcs.append(dash.no_update)

    return updated_srcs


orientation_to_dim: Dict[str, int] = {
    'sagittal': 0,
    'axial': 2,
    'coronal': 1
}


def get_image(dataset: str, sequence: str = 't1w', orientation: str = 'axial', slice: int = 80):
    dim = orientation_to_dim.get(orientation, 0)
    image = images[dataset][sequence].take([slice], axis=dim).squeeze(axis=dim)

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray', origin='lower')

    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"
