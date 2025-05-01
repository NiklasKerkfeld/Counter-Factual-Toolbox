import os

from dash import html, dcc, State
import dash
from app import app
from dash import Input, Output, ctx, ALL

from src.Visualization.Dashboard.load import dataset_folders


# Create one button per dataset
dataset_buttons = [
    html.Button(
        folder,
        id={'type': 'dataset-button', 'index': folder},
        n_clicks=0,
        style={
            "display": "block", "margin": "10px", "background": "none",
            "border": "none", "color": "blue", "textAlign": "left", "cursor": "pointer"
        }
    ) for folder in dataset_folders
]

# Layout
app.layout = html.Div([
    # Shared store for data selection
    dcc.Store(id='shared-data-selection', data={'datasets': []}),

    # Left Sidebar (10%) with buttons instead of dcc.Link
    html.Div([
        html.Div([
            html.H4("Select Dataset"),
            dcc.Store(id='dataset-store', data='none'),
            *dataset_buttons  # Unpack list of buttons
        ], style={'width': '10%', 'float': 'left', 'padding': '20px',
                  'backgroundColor': '#f9f9f9'}),

        # Right side (90%)
        html.Div([
            # Top navigation bar
            html.Div([
                dcc.Link('Values', href='/src/Visualization/Dashboard/pages/values',
                         style={'margin': '10px'}),
                dcc.Link('Image', href='/src/Visualization/Dashboard/pages/image',
                         style={'margin': '10px'}),
            ], style={'backgroundColor': '#e0e0e0', 'padding': '10px'}),

            # Page content area
            html.Div([
                dash.page_container
            ], style={'padding': '20px'})
        ], style={'width': '85%', 'float': 'right'})
    ])
])


@app.callback(
    Output('shared-data-selection', 'data'),
    Output({'type': 'dataset-button', 'index': ALL}, 'style'),
    Output('dataset-store', 'data'),
    Input({'type': 'dataset-button', 'index': ALL}, 'n_clicks'),
    State('shared-data-selection', 'data'),
    prevent_initial_call=True
)
def update_selected_datasets(n_clicks_list, current_data):
    triggered = ctx.triggered_id
    if not triggered:
        return dash.no_update, dash.no_update, dash.no_update

    # Get currently selected datasets
    current_selected = current_data.get('datasets', []) if current_data else []

    clicked_dataset = triggered['index']

    # Toggle logic
    if clicked_dataset in current_selected:
        current_selected.remove(clicked_dataset)
    else:
        current_selected.append(clicked_dataset)

    # Update styles: red for selected, blue otherwise
    def get_style(n_clicks):
        return {
            'color': 'red' if n_clicks % 2 == 1 else 'blue',
            'backgroundColor': 'transparent',
            'display': 'block',
            'margin': '10px',
            'border': 'none',
            'textAlign': 'left',
            'cursor': 'pointer'
        }

    # Match button IDs to dataset names
    styles = [get_style(n_clicks) for n_clicks in n_clicks_list]

    return {'datasets': current_selected}, styles, clicked_dataset


if __name__ == '__main__':
    app.run(debug=True)
