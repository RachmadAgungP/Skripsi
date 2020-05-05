import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
from dash.dependencies import Input, Output, State
import dash_table
import dash_table as dt
import main_lstm
import plotly.graph_objs as go
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

app.layout = html.Div(
    children=[
        dbc.Navbar(
            [    
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=PLOTLY_LOGO, height="40px")),
                        dbc.Col(dbc.NavbarBrand("Navbar", className="ml-2")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
            ],
            color="dark",
            dark=True,
        ),
        dbc.Container([ 
        dbc.Card([  
            html.Div(
                className="text-center",
                children=[
                    dbc.Card(
                        className="text-body",
                        children=[
                            html.H2(["PREDIKSI HARGA SAHAM"]),
                            html.H5(["Menggunakan Metode LSTM (Long Short Term Memory)"]),
                            dbc.Row(
                                className="text-body",
                                children=[
                                    dbc.Col(
                                        className="col-sm-12",
                                        style={"margin-bottom": "30px"},
                                        children=[
                                            dbc.Card([
                                                dcc.Graph(id="f_ori")],
                                            color="light", inverse=True),
                                            ],          
                                        ),
                                    dbc.Col(
                                        className="col-sm-4",
                                        children=[
                                            dbc.Card([
                                            html.Div(style={"margin": "20px"},
                                                children=[
                                                    html.H3(["Parameter"]),
                                                    dbc.Row([
                                                        dbc.Col([html.Label(["Banyak data"]),]),
                                                        dbc.Col([ dcc.Input(
                                                                style={'width': 80},
                                                                id="times-input",
                                                                placeholder="Enter a value...",
                                                                type="number",
                                                                value=100,
                                                                # debounce=True,
                                                                min=3,
                                                                disabled=True
                                                                ),])
                                                        ]),
                                                    dbc.Row([
                                                        dbc.Col([html.Label(["Skenario"]),]),
                                                        dbc.Col([ dcc.Input(
                                                                style={'width': 80},
                                                                id="Skenario",
                                                                placeholder="Enter a value...",
                                                                type="number",
                                                                value=4,
                                                                # debounce=True,
                                                                min=1,
                                                                max=5,
                                                                ),
                                                                ])
                                                        ]),
                                                    dbc.Row([
                                                        dbc.Col([html.Label(["Learning Rate"]),]),
                                                        dbc.Col([ dcc.Input(
                                                                style={'width': 80},
                                                                id="rate",
                                                                placeholder="Enter a value...",
                                                                type="number",
                                                                value=0.1,
                                                                # debounce=True,
                                                                min=0.01, max=1, step=0.01,
                                                            ),
                                                            ])
                                                        ]),
                                                    dbc.Row([dbc.Col([html.Label(["Epoch"]),]),
                                                        dbc.Col([ dcc.Input(
                                                            style={'width': 80},
                                                            id="Epoch",
                                                            placeholder="Enter a value...",
                                                            type="number",
                                                            # debounce=True,
                                                            value=1,
                                                            min=1,
                                                            ),])
                                                        ]),
                                                        html.Button(id='submit-button', n_clicks=0, children='Submit',style={"margin":"10px"}),
                                                    ]) 
                                                ])      
                                            ],
                                        ),
                                    dbc.Col(
                                        className="col-sm-8",
                                        children=[
                                            dbc.Card([
                                                dcc.Graph(id="results-graph_hasil_prediksi")]),
                                            dbc.Row([
                                                dbc.Col([html.Div(children=["MAPE"]),dbc.Card(children=["MAPE"],id="MAPE1"),
                                                ]),
                                                dbc.Col([html.Div(children=["Akurasi"]),dbc.Card(id="ac"),
                                                ]),
                                                dbc.Col([html.Div(children=["MSE"]),dbc.Card(id="MSE1"), 
                                                ])
                                                ])
                                            ],       
                                        ),
                                    dbc.Col(
                                        className="col-sm-12",
                                        children=[
                                        html.Div(id='tbl_hasil',style={"margin": "20px"}) 
                                        ]
                                        
                                        ),
                                    
                                    dbc.Col(className="col-sm-4 text-body",
                                        children=[
                                            dbc.Card(className="text-body",children=[html.H3("Data Trainig"),html.Div(id='tra',style={"margin-right":"30px","margin-left":"30px"})]), 
                                        ]
                                        
                                        ),
                                    dbc.Col(className="col-sm-8 text-body",
                                        children=[
                                            dbc.Card(className="text-body",children=[dcc.Graph(id="f_tra",style={"margin":"8px"})]), 
                                        ]
                                        
                                        ),
                               
                                    dbc.Col(className="col-sm-4 text-body",
                                        children=[
                                            dbc.Card(className="text-body",children=[html.H3("Data Testing"),html.Div(id='fore',style={"margin-right":"30px","margin-left":"30px"})]), 
                                        ]
                                        
                                        ),
                                    dbc.Col(className="col-sm-8 text-body",
                                        children=[
                                            dbc.Card(className="text-body",children=[dcc.Graph(id="f_fore",style={"margin":"8px"})]), 
                                        ]
                                        
                                        ),
                                
                                
                                ],
                                style={"margin": "10px"},
                                ),
                            
                            ],
                            style={"margin-top": "20px"},
                        ),

                    ],
                ),
            ],
            
            )
        ],
        style={"margin-top":"-50px"},
        ),
    ],
)

def figur(fig_data):
    figure = go.Figure(
        data=fig_data,
        layout=go.Layout(
            xaxis=dict(zeroline=False),
            yaxis=dict(
                title=dict(
                    text="Close",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=12,
                    ),
                ),
                type="log",
                rangemode="tozero",
                zeroline=False,
                showticklabels=True,
            ),
            margin=dict(l=40, r=30, b=50, t=50),
            showlegend=True,
            height=294,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)",
            xaxis_rangeslider_visible=True
        ),
    )
    return figure

@app.callback(
    [Output("results-graph_hasil_prediksi", "figure"),
    Output("MAPE1", 'children'),
    Output("ac", "children"),
    Output("tra", "children"),
    Output("fore", "children"),
    Output("f_tra", "figure"),
    Output("f_fore", "figure"),
    Output("f_ori", "figure"),
    Output("MSE1", "children"),
    Output("tbl_hasil", "children"),],
    [Input('submit-button', 'n_clicks')],
    [State("Skenario","value"),State("Epoch","value"),State("rate","value")],
)
def update_output1(n_clicks, input1, input2,  input3):
    from_lstm = main_lstm.maini(input1, input2,  input3)
    traa = from_lstm[9]
    foree = from_lstm[10]
    ori = from_lstm[11]
    bayak_data = len(ori)
    fig_ori = []
    fig_ori.append(go.Scatter(x= ori["date"].tolist(), y= ori["x(close)"].tolist(), mode='lines', name='close'))
    figure_ori = figur(fig_ori)

    MAPE = from_lstm[6]
    accuracy = from_lstm[7] 
    MSE = from_lstm[8]
    tbl_lstm = from_lstm[12]

    fig_data = []

    fig_data.append(go.Scatter(x= tbl_lstm["times"].tolist(), y= tbl_lstm["real"].tolist(), mode='lines', name='real'))
    fig_data.append(go.Scatter(x= tbl_lstm["times"].tolist(), y= tbl_lstm["prediksi"].tolist(), mode='lines', name='prediksi'))
    
    figure = figur(fig_data)
    
    df = traa
    df1 = foree

    fig_data_training = []
    fig_data_testing = []

    fig_data_training.append(go.Scatter(x= df["date"].tolist(), y= df["x(close)"].tolist(), mode='lines', name='Train'))
    fig_data_testing.append(go.Scatter(x= df1["date"].tolist(), y= df1["x(close)"].tolist(), mode='lines', name='Test'))
    
    figure_training = figur(fig_data_training)
    figure_testing = figur(fig_data_testing)

    data = df.to_dict('rows')
    columns =  [{"name": i, "id": i,} for i in (df.columns)]

    data1 = df1.to_dict('rows')
    columns1 =  [{"name": i, "id": i,} for i in (df1.columns)]

    data2 = tbl_lstm.to_dict('rows')
    columns2 =  [{"name": i, "id": i,} for i in (tbl_lstm.columns)]

    tra = dt.DataTable(data=data, columns=columns,page_size=5 )
    fore = dt.DataTable(data=data1, columns=columns1,page_size=5)
    tbl_hasil = dt.DataTable(data=data2, columns=columns2,page_size=5)
    return figure, MAPE, accuracy , tra, fore, figure_training, figure_testing, figure_ori, MSE, tbl_hasil



if __name__ == "__main__":
    app.run_server(debug=True)
