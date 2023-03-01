#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 20:32:53 2023

@author: brandonquach
"""

import dash_core_components as dcc
from dash import html, Dash
import dash_table
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
        html.H2(id='Text'),    
        dcc.Graph(id='simulated-prices', responsive=True),
        html.H2(id='Text2', children='Simulated Return Statistics Grid'),
        dash_table.DataTable(id='statistics-grid', 
                             style_table={'overflowX': 'auto', 'minWidth': '100%'}, 
                             fixed_columns={'headers': True, 'data': 1},
                             style_cell_conditional=[
                                 {'if': {'column_id': 'Percentile'},
                                  'minWidth': '150px'}])
]))])

# Inputs trigger callback, States are read only
@app.callback(
    [Output('simulated-prices', 'figure'), Output('Text', 'children'), 
     Output('statistics-grid', 'data'), Output('statistics-grid', 'columns')],
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
    fig_totalsales.update_layout(title = 'Daily Closing')
    
    # Calculate simulated returns
    simulation_df = (simulation_df - starting_point) / starting_point   
    
    # Calculate percentile statistics for each day out
    simulation_df = simulation_df.quantile(q=[0.1, 0.25, 0.5, 0.75, 0.9], axis=0)
    simulation_df = simulation_df.round(2)
    simulation_df.columns = ['Day ' + str(c) for c in simulation_df.columns]
    simulation_df.index.rename('Percentile', inplace=True)
    simulation_df = simulation_df.reset_index()
    
    columns=[{'id': c, 'name': c} for c in simulation_df.columns]
    
    return fig_totalsales, title, simulation_df.to_dict('records'), columns

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)