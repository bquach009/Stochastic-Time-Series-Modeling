#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 20:32:53 2023

@author: brandonquach
"""

import dash_core_components as dcc
from dash import html, Dash
from dash.dependencies import Input, Output, State
from estimate_merton_parameters import pull_data
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

###################
#APPLICATION DASH
###################

app = Dash(__name__)

# Colors
colors = {
    'background': '#111111',
    'text': '#FFFFFF'
}

input_styling = {'height': '100%', 'margin' : '0px'}


app.layout = html.Div(style={'backgroundColor': colors['background'], 
                             'color': colors['text']}, 
                      children=[
    html.H1(children='Stock Data Visualizer', style={'textAlign': 'center'
        }),
    html.Div([
        dcc.Input(id='ticker-input', type='text', placeholder='Ticker', style=input_styling),
        dcc.DatePickerRange(
            style=input_styling,
            id='my-date',
            start_date=str((datetime.now() - timedelta(days=365*5)).date()),
            end_date=str(datetime.now().date())),
        html.Button('Submit', id='submit-val', n_clicks=0, style=input_styling)], className='row'),
    html.Div([
        html.Div(id='Text', style={'font-size': '26px'}),    
        dcc.Graph(id='totalsales'),
])])

# Inputs trigger callback, States are read only
@app.callback(
    [Output('totalsales', 'figure'), Output('Text', 'children')],
    [Input('submit-val', 'n_clicks'),
     State('my-date', 'start_date'),
     State('my-date', 'end_date'),
     State('ticker-input', 'value')])
def update_output(n_clicks, start_date, end_date, ticker):
    if not n_clicks:
        raise PreventUpdate
        
    ticker = ticker.upper()
    data = pull_data(ticker)
    data = data[data.index <= end_date]
    data = data[data.index >= start_date]
    
    if data.empty:
        message = "Couldn't pull data, check ticker again or wifi connection"
        return go.Figure(), message
    
    title = f'Stock Data: {ticker}'

    fig_totalsales=go.Figure()
    fig_totalsales.add_trace(go.Scatter(x=data.index, y=data['Close'], visible=True, name='Closing'))
    fig_totalsales.update_layout(title = 'Daily Closing', width=1350, height=450)
    
    return fig_totalsales, title

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)