import dash
from dash import html, Input, Output, callback, dcc, ALL
import plotly.express as px
from src.Visualization.Dashboard.load import value_data

dash.register_page(__name__, path='/src/Visualization/Dashboard/pages/values')

value_plots = [
    html.Div([
        html.H2(value),
        html.Div(id='loss-page-output'),
        dcc.Graph(id={'type': 'value_plot', 'index': value}),
    ]) for value in set(value_data['key'])
]

layout = html.Div([
    html.H1("Values"),
    *value_plots
])


@callback(
    Output({'type': 'value_plot', 'index': ALL}, 'figure'),
    Input('shared-data-selection', 'data'),
    prevent_initial_call=True
)
def update_page2(data):
    datasets = data.get('datasets', [])
    filtered = value_data[value_data['dataset'].isin(datasets)]

    figs = []
    for key in set(value_data['key']):
        figs.append(px.line(filtered[filtered['key'] == key],
                            x='step',
                            y='value',
                            color='dataset'))

    return figs
