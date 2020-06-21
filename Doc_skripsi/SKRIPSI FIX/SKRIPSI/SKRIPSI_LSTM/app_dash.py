import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
import dash_table
import dash_table as dt

import SKRIPSI_LSTM
import plotly.graph_objs as go
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

Header_Data_Saham = dbc.CardHeader(
                        className="bodyyy shadow rounded",
                        children=[
                            html.Div(
                                style={"float": "left"},
                                className="",
                                children=[
                                    html.H5(["Data Saham Semen Indonesia"]),
                                ]
                            ), 
                            html.Div(
                                style={"float": "right","margin-left":"10px"},
                                children=[
                                    html.Button(className="btn content",id="submit-button1", n_clicks=0, children='Skenario'),
                                ]
                            ),
                            html.Div(
                                style={"float": "right"},
                                children=[
                                    dcc.Input(
                                        id="Skenar",
                                        placeholder="Enter a value...",
                                        type="number",
                                        debounce=True,
                                        min=1,
                                        max=5,
                                    ),
                                ]
                            ),
                        ]
                    )

Header_training = dbc.CardHeader(
                className="bodyyy shadow rounded",
                children=[
                    html.Div(
                        style={"float": "left"},
                        className="",
                        children=[
                            html.H5(["Training Data"]),
                        ]
                    ),
                    html.Div(
                        style={"float": "right","margin-left":"10px"},
                        children=[
                            html.Button(className="btn content",id='submit-button2', n_clicks=0, children='Training'),
                        ]
                    ), 
                    dbc.Row(
                        style={"float":"right"},
                        children=[
                                html.Label(["Rate :"]),
                                dbc.Col(
                                    children=[
                                        dcc.Input(
                                            id="rate",
                                            placeholder="Enter a value...",
                                            type="number",
                                            debounce=True,
                                            min=0.01, 
                                            step=0.01,
                                        ),
                                    ]
                                ),
                                html.Label(["Epoch :"]),
                                dbc.Col(
                                    style={"margin-left":"5px"},
                                    children=[
                                        dcc.Input(
                                            id="Epoch",
                                            placeholder="Enter a value...",
                                            type="number",
                                            debounce=True,
                                            min=1,
                                        ),
                                    ]
                                ),      
                            ]
                        ),
                    ]
                )

Header_testing = dbc.CardHeader(
                    className="bodyyy shadow rounded",
                    children=[
                        html.Div(
                            style={"float": "left"},
                            className="",
                            children=[
                                html.H5(["Testing Data"]),
                            ]
                        ), 
                        html.Div(
                            style={"float": "right","margin-left":"10px"},
                            children=[
                                html.Button(className="btn content",id="submit-button3", n_clicks=0, children='Testing'),
                            ]
                        ),
                ])

Header_prediksi = dbc.CardHeader(
                    className="bodyyy shadow rounded",
                    children=[
                        html.Div(
                            style={"float": "left"},
                            className="",
                            children=[
                                html.H5(["Prediksi Data"]),
                            ]
                        ), 
                        html.Div(
                            style={"float": "right","margin-left":"10px"},
                            children=[ 
                                html.Button(className="btn content",id="submit-button4", n_clicks=0, children='Prediksi'),
                            ]
                        ),
                        html.Div(
                            style={"float": "right"},
                            children=[
                            dcc.Input(
                                id="prediksi_ke",
                                placeholder="Enter a value...",
                                type="number",
                                debounce=True,
                                min=1,
                                max=5,
                            ),
                        ]
                    ),
                ]
            )

app.layout = html.Div(
    children=[
        dbc.Navbar(
            className="content",
            children=[    
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=PLOTLY_LOGO, height="40px", className="ml-5")),
                        dbc.Col(dbc.NavbarBrand("Prediksi Harga Saham", className="ml-5")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
            ],
            color="dark",
            dark=True,
        ),
        html.Div(
            style={"margin":"60px","margin-left":"130px","margin-right":"130px"},
            className="text-center",
            children=[ 
                dbc.Card(
                    className="shadow rounded ",
                    children=[
                        dbc.CardHeader(
                            className="content",
                            children=[
                                html.Img(src="https://www.bootstrapdash.com/demo/purple-admin-free/assets/images/dashboard/circle.svg", className="card-img-absolute", alt="circle-image"),
                                html.H2(["PREDIKSI HARGA SAHAM"]),
                            ]
                        ),
                        dbc.CardBody(
                            className="bodyy",
                            children=[
                            # html.H5(["Metode LSTM (Long Short Term Memory)"]),
                            # Data Saham Semen Indonesia
                                dbc.Row(
                                    className="",
                                    style={"margin": "10px"},
                                    children=[
                                        dbc.Col(
                                            style={"margin-bottom":"15px"},
                                            className="col-sm-12",
                                            children=[
                                                dbc.Card(
                                                    className="bodyyy shadow rounded",
                                                    children=[
                                                        Header_Data_Saham,
                                                        html.Div(id="data")#Body_Data_Saham
                                                    ]
                                                ),
                                            ]
                                        ),
                                        # ------------------------------------
                                        # Training
                                        dbc.Col(
                                            style={"margin-bottom":"15px"},
                                            className="col-sm-12",
                                            children=[
                                                dbc.Card(
                                                    className="bodyyy shadow rounded",
                                                    children=[
                                                        Header_training,
                                                        html.Div(id="data2")# Body_training
                                                    ]
                                                ),
                                            ]
                                        ),
                                        # Testing 
                                        dbc.Col(
                                            style={"margin-bottom":"15px"},
                                            className="col-sm-12",
                                            children=[
                                                dbc.Card(
                                                    className="bodyyy shadow rounded",
                                                    children=[
                                                        Header_testing,
                                                        html.Div(id="data3")# Body_testing
                                                    ]
                                                ),
                                            ]
                                        ), 
                                        # Prediksi 
                                        dbc.Col(
                                            style={"margin-bottom":"15px"},
                                            className="col-sm-12",
                                            children=[
                                                dbc.Card(
                                                    className="bodyyy shadow rounded",
                                                    children=[
                                                        Header_prediksi,
                                                        html.Div(id="data4")# Body_prediksi
                                                    ]
                                                ),
                                            ]
                                        ), 
                                    ]
                                ),
                            ]
                        ),
                        dbc.CardFooter(
                            className="footy text-right text-dark",
                            children=[
                                html.Div(["Rachmad Agung Pambudi // 160411100032"])
                            ]
                        )
                    ],
                ),
            ],
        ),
    ],
)

sequenceLength = 5

from data.preprosesing_data import readData
DataSahamStr = 'data\SMGR.JKq.csv'
# data = pd.read_csv(DataSahamStr)
# I_DataSaham = SKRIPSI_LSTM.readData(data,skenarioI)
I_DataSaham = readData(DataSahamStr)
print ("Data yang dipakai adalah %s"%DataSahamStr)
data = I_DataSaham
originalData = data[3]
corpusData = data[0]

print ("corpusData ",corpusData)
max_ex = data[1]
min_ex = data[2]
corpusData = np.concatenate((np.ones((corpusData.shape[0], 1)), corpusData), axis=1)
W = [[-0.245714286	,0.850360602	,0.029262045	,0.184398087]
                ,[0.868020398	,0.860429754	,-0.379580925	,0.079506914]
                ,[-0.206444161	,-0.24856166	,-0.085253247	,0.25112624	]
                ,[0.842874383	,-0.324206065	,0.907722829	,-0.593738792]]
lstm = SKRIPSI_LSTM.LSTMCell(corpusData.shape[1], corpusData.shape[1]-2,W)

from obj.view_konten import data,train,test,prediksi 

@app.callback(Output("data", "children"),
                [Input('submit-button1', 'n_clicks')],
                [State("Skenar","value")])

def main(btn1,input1):
    ctx = dash.callback_context
    if input1 is None :
        # PreventUpdate prevents ALL outputs updating
        raise dash.exceptions.PreventUpdate

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'submit-button1':
            print (button_id)
            r1 = data(btn1, input1, originalData)
            return (r1)

@app.callback(Output("data2", "children"),
            [Input('submit-button2', 'n_clicks')],
            [State("Epoch","value"),State("rate","value"),State("Skenar","value")])

def main2(btn2,input2,input3,input1):
    ctx = dash.callback_context
    if input2 is None :
        raise dash.exceptions.PreventUpdate

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'submit-button2':
            print (button_id)
            r2 = train(btn2,input2,input3,input1, corpusData, sequenceLength, max_ex, min_ex, originalData)
            return (r2)

@app.callback(Output("data3", "children"),
                [Input('submit-button3', 'n_clicks')],
                [State("Skenar","value")])

def main3(btn3,input1):
    ctx = dash.callback_context
    if input1 is None :
        raise dash.exceptions.PreventUpdate

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'submit-button3':
            print (button_id)
            r3 = test(btn3, input1, originalData, sequenceLength, corpusData, lstm, max_ex, min_ex)
            return (r3)

@app.callback(Output("data4", "children"),
                [Input('submit-button4', 'n_clicks')],
                [State("prediksi_ke","value"),State("Skenar","value")])

def main4(btn4,input4,input1):
    ctx = dash.callback_context
    if input4 is None  :
        raise dash.exceptions.PreventUpdate

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'submit-button4':
            print (button_id)
            r4 = prediksi(btn4,input4,input1,corpusData, sequenceLength, lstm, max_ex, min_ex)
            return (r4)

if __name__ == "__main__":
    app.run_server(debug=True)

