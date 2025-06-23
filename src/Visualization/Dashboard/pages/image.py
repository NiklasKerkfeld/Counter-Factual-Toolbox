import base64
import io
from typing import Dict, Tuple

import dash
import numpy as np
from PIL import Image
from dash import html, Input, Output, callback, ALL, dcc, ctx, MATCH, State

import src.utils
from src.Visualization.Dashboard.load import loader
from src.Visualization.Dashboard.utils import blend_segmentation, pad, blend_change

dash.register_page(__name__, path='/src/Visualization/Dashboard/pages/image')

image_blocks = []

layout = html.Div([
    html.H1("Images"),
    html.Div(id='image-container')
])

orientation_to_dim: Dict[str, int] = {
    'sagittal': 0,
    'axial': 2,
    'coronal': 1
}


@callback(
    Output('image-container', 'children', allow_duplicate=True),
    Input('shared-data-selection', 'data'),
    State('image-container', 'children'),
    prevent_initial_call=True
)
def update_image_blocks(selected_datasets, current_children):
    selected_datasets = selected_datasets['datasets']
    if not isinstance(current_children, list):
        current_children = []

    # Get currently rendered dataset IDs from existing blocks
    current_ids = {child['props']['id']['index'] for child in current_children if
                   'props' in child and isinstance(child['props'].get('id'), dict)}

    selected_datasets = set(selected_datasets or [])

    # Compute additions and removals
    to_add = selected_datasets - current_ids
    to_keep = selected_datasets & current_ids

    updated_children = [child for child in current_children if
                        child['props']['id']['index'] in to_keep]

    for dataset in to_add:
        updated_children.append(
            build_image_block(dataset))  # 👈 helper function builds a single block

    return updated_children


def build_image_block(dataset):
    image_dict = src.utils.get_image(dataset)
    init_seq = list(image_dict.keys())[0]
    min_val = int(image_dict[init_seq].min())
    max_val = int(image_dict[init_seq].max() + 1)
    steps = loader.get_steps(dataset)

    image_cells = []
    for orientation in ['axial', 'sagittal', 'coronal']:
        slices = image_dict[init_seq].shape[orientation_to_dim[orientation]]
        image_cells.append(
            html.Div([
                html.H4(orientation),
                html.Div([
                    dcc.RangeSlider(min=min_val, max=max_val, step=0.1,
                                    value=[min_val, max_val],
                                    marks={min_val: str(min_val), max_val: str(max_val)},
                                    vertical=True,
                                    id={'type': 'range-slider', 'index': dataset,
                                        'orientation': orientation}),
                    html.Img(id={'type': 'mri-image', 'index': dataset, 'orientation': orientation},
                             style={'height': '475px'})
                ], style={'display': 'flex', 'alignItems': 'center'}),
                dcc.Slider(
                    id={'type': 'slice-slider', 'index': dataset, 'orientation': orientation},
                    min=0,
                    max=slices,
                    step=1,
                    value=slices // 2,
                    marks={0: '0', slices: str(slices)},
                    tooltip={'placement': 'bottom'}),
                dcc.Store(
                    id={'type': 'sequence-store', 'index': dataset, 'orientation': orientation},
                    data=init_seq),
                dcc.Store(
                    id={'type': 'checklist-store', 'index': dataset, 'orientation': orientation},
                    data=[]),
                dcc.Store(
                    id={'type': 'step-store', 'index': dataset, 'orientation': orientation},
                    data=max(steps + [0]))
            ])
        )

    return html.Div([
        html.Div([
            html.H2(dataset),
            dcc.Dropdown(
                id={'type': 'image-dropdown', 'index': dataset},
                options=[{'label': k, 'value': k} for k in image_dict.keys() if k != 'target'],
                value=init_seq,
                clearable=False),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}),
        html.Div([
            dcc.Checklist(
                id={'type': 'checklist', 'index': dataset},
                options=[{'label': 'show change', 'value': 'change'},
                         {'label': 'show target', 'value': 'target'},
                         {'label': 'show prediction', 'value': 'pred'},
                         {'label': 'apply change', 'value': 'apply'},
                         ],
                value=[],  # empty = unchecked, ['change'] = checked
                inline=True
            ),
            html.Div([
                html.H4("step:"),
                html.Div(
                    dcc.Slider(
                        id={'type': 'step-slider', 'index': dataset},
                        min=0,
                        max=max(steps + [0]),
                        step=None,  # disables arbitrary steps
                        marks={s: str(s) for s in steps},
                        value=max(steps + [0]),
                        tooltip={'placement': 'bottom'}
                    ), style={'flex': '1'}
                )
            ], style={'display': 'flex', 'alignItems': 'center', 'width': '100%'})
        ]),
        html.Div(image_cells, style={'display': 'flex', 'alignItems': 'center'})
    ], id={'type': 'image-block', 'index': dataset})


@callback(
    Output({'type': 'mri-image', 'index': MATCH, 'orientation': MATCH}, 'src'),
    Input({'type': 'range-slider', 'index': MATCH, 'orientation': MATCH}, 'value'),
    Input({'type': 'slice-slider', 'index': MATCH, 'orientation': MATCH}, 'value'),
    Input({'type': 'sequence-store', 'index': MATCH, 'orientation': MATCH}, 'data'),
    Input({'type': 'checklist-store', 'index': MATCH, 'orientation': MATCH}, 'data'),
    Input({'type': 'step-store', 'index': MATCH, 'orientation': MATCH}, 'data'),
    prevent_initial_call=True
)
def update_image(value_range, selected_slice, sequence, checklist, step):
    triggered_id = ctx.triggered_id
    dataset = triggered_id['index']
    orientation = triggered_id['orientation']
    show_target = 'target' in checklist
    show_change = 'change' in checklist
    show_pred = 'pred' in checklist
    apply_change = 'apply' in checklist
    return get_image(dataset, value_range, sequence, orientation, selected_slice, step,
                     show_target, show_change, show_pred, apply_change)


def get_image(dataset: str,
              value_range: Tuple[int, int],
              sequence: str = 't1w',
              orientation: str = 'axial',
              slice: int = 80,
              step: int = 0,
              show_target: bool = False,
              show_change: bool = False,
              show_pred: bool = False,
              apply_change: bool = False):
    dim = orientation_to_dim.get(orientation, 0)
    image = src.utils.get_image(dataset)[sequence].take([slice], axis=dim).squeeze(axis=dim).clip(
        value_range[0],
        value_range[1])
    image -= value_range[0]
    image /= (value_range[1] + 1e-9)

    if apply_change:
        print("applying change")
        image = loader.get_changed_image(dataset, step, sequence).take([slice], axis=dim).squeeze(
            axis=dim)
        image -= value_range[0]
        image /= (value_range[1] + 1e-9)

    image = (image * 255).astype(np.uint8)  # Normalize to 0-255
    img_pil = Image.fromarray(image)

    if show_change:
        print("showing change")
        change = loader.get_change(dataset, step, sequence).take([slice], axis=dim).squeeze(
            axis=dim)
        img_pil = blend_change(img_pil, change, cmap='RdBu')

    if show_pred:
        pred = loader.get_pred(dataset, step).take([slice], axis=dim).squeeze(axis=dim)
        img_pil = blend_segmentation(img_pil, pred, cmap='Reds')

    if show_target:
        target = src.utils.get_image(dataset)['target'].take([slice], axis=dim).squeeze(axis=dim)
        img_pil = blend_segmentation(img_pil, target)


    img_pil = pad(img_pil)
    img_pil.rotate(90)

    buf = io.BytesIO()
    src.utils.save(buf, format='PNG')
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


@callback(
    Output({'type': 'range-slider', 'index': MATCH, 'orientation': ALL}, 'max'),
    Output({'type': 'range-slider', 'index': MATCH, 'orientation': ALL}, 'min'),
    Input({'type': 'sequence-store', 'index': MATCH, 'orientation': ALL}, 'data'),
    prevent_initial_call=True
)
def sequence_select(sequences):
    dataset = ctx.triggered_id['index']
    image = src.utils.get_image(dataset)[sequences[0]]
    max_value, min_value = image.max(), image.min()

    return [max_value, max_value, max_value], [min_value, min_value, min_value]


@callback(
    Output({'type': 'sequence-store', 'index': MATCH, 'orientation': ALL}, 'data'),
    Input({'type': 'image-dropdown', 'index': MATCH}, 'value'),
    prevent_initial_call=True
)
def sequence_select(sequence):
    return [sequence, sequence, sequence]


@callback(
    Output({'type': 'checklist-store', 'index': MATCH, 'orientation': ALL}, 'data'),
    Input({'type': 'checklist', 'index': MATCH}, 'value')
)
def checklist_select(checklist):
    return [checklist, checklist, checklist]


@callback(
    Output({'type': 'step-store', 'index': MATCH, 'orientation': ALL}, 'data'),
    Input({'type': 'step-slider', 'index': MATCH}, 'value')
)
def step_select(step):
    return [step, step, step]
