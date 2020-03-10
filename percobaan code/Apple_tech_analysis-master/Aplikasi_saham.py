import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
# Initialize the app
app = dash.Dash(__name__)
df_aapl_raw = pd.read_csv("Apple_tech_analysis-master\data\AAPL.csv")
df_spc_raw = pd.read_csv("Apple_tech_analysis-master\data\GSPC.csv")

df_aapl_slice = df_aapl_raw[2:].reset_index()
df_spc_slice = df_spc_raw[:-3].reset_index() 
df_aapl_slice['Year'] = pd.DatetimeIndex(df_aapl_slice['Date']).year
df_spc_slice['Year'] = pd.DatetimeIndex(df_spc_slice['Date']).year

external_stylesheets =['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([html.H1("Moving Average Crossover Strategy For Apple Stocks ")], style={'textAlign': "center"}),
    html.Div([
        html.Div([
            html.Div([dcc.Graph(id="my-graph")], className="row", style={"margin": "auto"}),
            html.Div([html.Div(dcc.RangeSlider(id="year selection", updatemode='drag',
                                               marks={i: '{}'.format(i) for i in df_aapl_slice.Year.unique().tolist()},
                                               min=df_aapl_slice.Year.min(), max=df_aapl_slice.Year.max(), value=[2014, 2019]),
                               className="row", style={"padding-bottom": 30,"width":"60%","margin":"auto"}),
                      html.Span("Moving Average : Select Window Interval", className="row",
                                style={"padding-top": 30,"padding-left": 40,"display":"block",
                                       "align-self":"center","width":"80%","margin":"auto"}),
                      html.Div(dcc.Slider(id="select-range1", updatemode='drag',
                                          marks={i * 10: str(i * 10) for i in range(0, 21)},
                                          min=0, max=200, value=50), className="row", style={"padding": 10}),
                      html.Div(dcc.Slider(id="select-range2", updatemode='drag',
                                          marks={i * 10: str(i * 10) for i in range(0, 21)},
                                          min=0, max=200, value=170), className="row", style={"padding": 10})

                      ], className="row")
        ], className="six columns",style={"margin-right":0,"padding":0}),
        html.Div([
            dcc.Graph(id="plot-graph")
        ], 
        className="six columns",style={"margin-left":0,"padding":0}),
    ], className="row")
   ], className="container")

@app.callback(
    Output("my-graph", 'figure'),
    [Input("year selection", 'value'),
     Input("select-range1", 'value'),
     Input("select-range2", 'value')])
def update_ma(year, range1, range2):
    df_apl = df_aapl_slice[(df_aapl_slice["Year"] >= year[0]) & (df_aapl_slice["Year"] <= year[1])]

    rolling_mean1 = df_apl['Adj Close'].rolling(window=range1).mean()
    rolling_mean2 = df_apl['Adj Close'].rolling(window=range2).mean()

    trace1 = go.Scatter(x=df_apl['Date'], y=df_apl['Adj Close'],
                        mode='lines', name='AAPL')
    trace_a = go.Scatter(x=df_apl['Date'], y=rolling_mean1, mode='lines', yaxis='y', name=f'SMA {range1}')
    trace_b = go.Scatter(x=df_apl['Date'], y=rolling_mean2, mode='lines', yaxis='y', name=f'SMA {range2}')

    layout1 = go.Layout({'title': 'Stock Price With Moving Average',
                         "legend": {"orientation": "h","xanchor":"left"},
                         "xaxis": {
                             "rangeselector": {
                                 "buttons": [
                                     {"count": 6, "label": "6M", "step": "month",
                                      "stepmode": "backward"},
                                     {"count": 1, "label": "1Y", "step": "year",
                                      "stepmode": "backward"},
                                     {"count": 1, "label": "YTD", "step": "year",
                                      "stepmode": "todate"},
                                     {"label": "5Y", "step": "all",
                                      "stepmode": "backward"}
                                 ]
                             }}})

    figure = {'data': [trace1],
              'layout': layout1
              }
    figure['data'].append(trace_a)
    figure['data'].append(trace_b)
    return figure


@app.callback(
    Output("plot-graph", 'figure'),
    [Input("year selection", 'value')])
def update_return(year):

    df_apl = df_aapl_slice[(df_aapl_slice["Year"] >= year[0]) & (df_aapl_slice["Year"] <= year[1])]
    df_sp = df_spc_slice[(df_spc_slice["Year"] >= year[0]) & (df_spc_slice["Year"] <= year[1])]

    stocks = pd.DataFrame({"Date": df_sp["Date"], "AAPL": df_apl["Adj Close"],
                           "S&P500": df_sp["Adj Close"]})
    stocks = stocks.set_index('Date')
    stock_return = stocks.apply(lambda x: ((x - x[0]) / x[0])*100)

    trace2 = go.Scatter(x=df_sp['Date'], y=stock_return['AAPL'], mode='lines', name='Apple')
    trace3 = go.Scatter(x=df_sp['Date'], y=stock_return['S&P500'], mode='lines', name='S&P 500')

    layout2 = go.Layout({'title': 'Returns (%) : AAPL vs S&P 500 ',
                         "legend": {"orientation": "h","xanchor":"left"}, })

    fig = {'data': [trace2],
           'layout': layout2
           }
    fig['data'].append(trace3)
    return fig
if __name__ == '__main__':
    app.run_server(debug=True)