#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 20:32:53 2023

@author: brandonquach
"""

import dash_core_components as dcc
from dash import html, Dash
from dash.dependencies import Input, Output, State
from estimate_merton_parameters import pull_data, optimize_parameters
from datetime import datetime, timedelta
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import numpy as np
import pandas as pd

###################
#APPLICATION DASH
###################

app = Dash(__name__)

# Colors
colors = {
    'background': '#111111',
    'text': '#FFFFFF'
}

input_styling = {}


app.layout = html.Div(children=[
    html.H1(children='Stock Data Visualizer'),
    html.Div([
        dcc.Input(id='ticker-input', 
                  type='text', 
                  placeholder='Ticker', 
                  className='item'),
        dcc.DatePickerRange(
            id='my-date',
            start_date=str((datetime.now() - timedelta(days=365*5)).date()),
            end_date=str(datetime.now().date()),
            className='item'),
        html.Button('Submit', 
                    id='submit-val', 
                    n_clicks=0, 
                    className='item')], 
        className='row'),
        dcc.Loading(
    html.Div([
        html.Div(id='Text'),    
        dcc.Graph(id='totalsales'),
]))])

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
    
    returns = data['log_return'].dropna().values[-365*2:]

    mu, sigma, jump_rate, jump_mean, jump_std = optimize_parameters(returns, trials=2500, mode='likelihood')   
    
    # Simulate 365 draws
    n = 365
    starting_point = data['Close'].iloc[-1]
    
    if data.empty:
        message = "Couldn't pull data, check ticker again or wifi connection"
        return go.Figure(), message
    
    title = f'Stock Data: {ticker}'

    fig_totalsales=go.Figure()
    fig_totalsales.add_trace(go.Scatter(x=data.index, y=data['Close'], visible=True, name='Closing'))
    n_sim = 500
    simulation_df = []
    for i in range(n_sim):
        # Generate OOS
        predicted_log_returns = np.random.normal(mu, sigma, n) + np.random.normal(jump_mean, jump_std, n) * np.random.poisson(jump_rate, n)
        predicted_returns = np.exp(predicted_log_returns) - 1
    
        # generate prices using the predicted returns
        prices = starting_point * np.cumprod(1 + predicted_returns)
        simulation_df.append(list(prices))
        
        # Add traces
        fig_totalsales.add_trace(go.Scatter(x=pd.date_range(data.index[-1] + pd.Timedelta(days=1), data.index[-1] + pd.Timedelta(days=n + 1)), 
                                 y=prices,
                                 mode='lines',
                                 name='simulated', 
                                 line_color='rgba(128,128,128, 0.1)', 
                                 showlegend=False if i != 0 else True, 
                                 hoverinfo='skip'))
    simulation_df = pd.DataFrame(simulation_df)
    fig_totalsales.add_trace(go.Scatter(x=pd.date_range(data.index[-1] + pd.Timedelta(days=1), data.index[-1] + pd.Timedelta(days=n + 1)), 
                             y=simulation_df.sort_values(by=[n-1]).iloc[n_sim // 2],
                            mode='lines',
                            name='median_simulation'))
    fig_totalsales.update_layout(title = 'Daily Closing', width=1350, height=450)
    
    return fig_totalsales, title

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)