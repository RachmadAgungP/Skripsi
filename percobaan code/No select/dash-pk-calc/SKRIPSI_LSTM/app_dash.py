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
# import main_lstm
import SKRIPSI_LSTM
import plotly.graph_objs as go
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

Header_Data_Saham = dbc.CardHeader(className="bodyyy shadow rounded",children=[
                        html.Div(
                            style={"float": "left"},
                            className="",
                            children=[
                            html.H5(["Data Saham Semen Indonesia"]),
                        ]), 
                        html.Div(
                            style={"float": "right","margin-left":"10px"},
                            children=[
                            html.Button(className="btn content",id="submit-button1", n_clicks=0, children='Skenario'),
                        ]),
                        html.Div(
                            style={"float": "right"},
                            children=[
                            dcc.Input(
                                id="Skenar",
                                placeholder="Enter a value...",
                                type="number",
                                # value=4,
                                debounce=True,
                                min=1,
                                max=5,
                            ),
                        ]),
                ])

Header_training = dbc.CardHeader(className="bodyyy shadow rounded",children=[
                html.Div(
                            style={"float": "left"},
                            className="",
                            children=[
                            html.H5(["Training Data"]),
                        ]),
                html.Div(
                            style={"float": "right","margin-left":"10px"},
                            children=[
                            html.Button(className="btn content",id='submit-button2', n_clicks=0, children='Training'),
                        ]), 
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
                                                # value=0.1,
                                                debounce=True,
                                                min=0.01, 
                                                    
                                                step=0.01,
                                            ),
                                    ]),
                                    html.Label(["Epoch :"]),
                                    dbc.Col(
                                    style={"margin-left":"5px"},
                                    children=[
                                        dcc.Input(
                                            id="Epoch",
                                            placeholder="Enter a value...",
                                            type="number",
                                            debounce=True,
                                            # value=1,
                                            min=1,
                                            ),
                                        ]),      
                        ]),
                ])

Header_testing = dbc.CardHeader(className="bodyyy shadow rounded",children=[
                        html.Div(
                            style={"float": "left"},
                            className="",
                            children=[
                            html.H5(["Testing Data"]),
                        ]), 
                        html.Div(
                            style={"float": "right","margin-left":"10px"},
                            children=[
                            html.Button(className="btn content",id="submit-button3", n_clicks=0, children='Testing'),
                        ]),
                ])

Header_prediksi = dbc.CardHeader(className="bodyyy shadow rounded",children=[
                        html.Div(
                            style={"float": "left"},
                            className="",
                            children=[
                            html.H5(["Prediksi Data"]),
                        ]), 
                        html.Div(
                            style={"float": "right","margin-left":"10px"},
                            children=[ 
                            html.Button(className="btn content",id="submit-button4", n_clicks=0, children='Prediksi'),
                        ]),
                        html.Div(
                            style={"float": "right"},
                            children=[
                            dcc.Input(
                                id="prediksi_ke",
                                placeholder="Enter a value...",
                                type="number",
                                # value=5,
                                debounce=True,
                                # disabled=True,
                                min=1,
                            ),
                        ]),
                ])

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
                            ]),
                            dbc.CardBody(
                            className="bodyy",
                            children=[
                            # html.H5(["Metode LSTM (Long Short Term Memory)"]),
                            # Data Saham Semen Indonesia
                            dbc.Row(
                                className="",
                                children=[
                                    dbc.Col(className="col-sm-12",
                                    children=[
                                    dbc.Card(
                                        className="bodyyy shadow rounded",children=[
                                        Header_Data_Saham,
                                        html.Div(id="data")#Body_Data_Saham
                                        ]),
                                    ],style={"margin-bottom":"15px"}),
                                    # ------------------------------------
                                    # Training
                                    dbc.Col(className="col-sm-12",
                                    children=[
                                    dbc.Card(className="bodyyy shadow rounded",children=[
                                        Header_training,
                                        html.Div(id="data2")# Body_training
                                        ]),
                                    ],style={"margin-bottom":"15px"}),
                                
                                    # Testing 
                                    dbc.Col(className="col-sm-12",
                                    children=[
                                    dbc.Card(className="bodyyy shadow rounded",children=[
                                        Header_testing,
                                        html.Div(id="data3")# Body_testing
                                        ]),
                                    ],style={"margin-bottom":"15px"}), 

                                    # Prediksi 
                                    dbc.Col(className="col-sm-12",
                                    children=[
                                    dbc.Card(className="bodyyy shadow rounded",children=[
                                        Header_prediksi,
                                        html.Div(id="data4")# Body_prediksi
                                        ]),
                                    ],style={"margin-bottom":"15px"}), 
                                ],
                                style={"margin": "10px"},
                                ),
                                ]),
                                dbc.CardFooter(className="footy text-right text-dark",children=[
                                    html.Div(["Rachmad Agung Pambudi // 160411100032"])
                                ])
                            
                            ],
                            
                        ),
        ],
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

sequenceLength = 5

DataSahamStr = 'data\SMGR.JKq.csv'
# data = pd.read_csv(DataSahamStr)
# I_DataSaham = SKRIPSI_LSTM.readData(data,skenarioI)
I_DataSaham = SKRIPSI_LSTM.readData(DataSahamStr)
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
            r1 = data(btn1, input1)
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
            r2 = train(btn2,input2,input3,input1)
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
            r3 = test(btn3, input1)
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
            r4 = prediksi(btn4,input4,input1)
            return (r4)

def data(n_clicks, input1):
    
    data_no_normalisasi = SKRIPSI_LSTM.sk(input1,originalData)
    # data saham semuanya 
    originalData_no_normalisasi = pd.DataFrame(data=originalData,columns=["date","x(close)"])
    
    fig_ori = []
    fig_ori.append(go.Scatter(x= originalData_no_normalisasi["date"].tolist(), y= originalData_no_normalisasi["x(close)"].tolist(), mode='lines', name='close'))
    figure_ori = figur(fig_ori)

    # data saham 
    data_ori = originalData_no_normalisasi.to_dict('rows')
    columns_ori =  [{"name": i, "id": i,} for i in (originalData_no_normalisasi.columns)]
    tbl_ori = dt.DataTable(data=data_ori, columns=columns_ori,page_size=5,style_cell={'textAlign': 'center',"color":"#09203f" },style_header={
        "background-image": "linear-gradient(to top, #09203f 0%, #537895 100%)",
        'fontWeight': 'bold',
        "color":"white"
    })
    Body_Data_Saham = dbc.CardBody(
                    children=[
                        dbc.Row([
                        dbc.Col(className="col-sm-4",
                            children=[
                                dbc.Card(className="footy text-dark shadow rounded",children=[dbc.CardHeader(className="footy text-dark rounded",children=[html.H6("Data SAHAM")]),html.Div(children=[tbl_ori],style={"margin-top":"15px","margin-right":"30px","margin-left":"30px"})]), 
                            ]),
                        dbc.Col(
                            className="col-sm-8",
                            children=[
                                    dbc.Card(className="footy text-dark shadow rounded",children=[dcc.Graph(figure=figure_ori,style={"margin":"8px"})]),   
                                ],          
                            ),
                        ]),

                    ])
    return (Body_Data_Saham)

def train(n_clicks, input1, input2, value):
    print ("value t",value)
    
    W = [[-0.245714286	,0.850360602	,0.029262045	,0.184398087]
                ,[0.868020398	,0.860429754	,-0.379580925	,0.079506914]
                ,[-0.206444161	,-0.24856166	,-0.085253247	,0.25112624	]
                ,[0.842874383	,-0.324206065	,0.907722829	,-0.593738792]]
    lstm = SKRIPSI_LSTM.LSTMCell(corpusData.shape[1], corpusData.shape[1]-2,W)
    skenarioP = SKRIPSI_LSTM.sk(value,corpusData)
    trainingData = skenarioP[0]
    train_error = lstm.train(trainingData, input1, input2, sequenceLength,max_ex,min_ex)

    error = []
    error.append(go.Scatter(x= train_error["urutan"].tolist(), y= train_error["error"].tolist(), mode='lines', name='error'))
    figure_error = figur(error)

    data_no_normalisasi = SKRIPSI_LSTM.sk(value,originalData)

    trainingData_no_normalisasi = pd.DataFrame(data=data_no_normalisasi[0],columns=["date","x(close)"])

    df =  trainingData_no_normalisasi
    # data training
    data = df.to_dict('rows')
    columns =  [{"name": i, "id": i,} for i in (df.columns)]
    tra = dt.DataTable(data=data, columns=columns,page_size=5,style_cell={'textAlign': 'center',"color":"#09203f"},style_header={
        "background-image": "linear-gradient(to top, #09203f 0%, #537895 100%)",
        'fontWeight': 'bold',
        "color":"white"
    })

    fig_data_training = []
    fig_data_training.append(go.Scatter(x= df["date"].tolist(), y= df["x(close)"].tolist(), mode='lines', name='Train'))
    figure_training = figur(fig_data_training)
    Body_training = dbc.CardBody(
                    className="",
                    children=[
                        dbc.Row([
                        dbc.Col(className="col-sm-4 ",
                            children=[
                                dbc.Card(className="footy text-dark shadow rounded",children=[dbc.CardHeader(className="footy text-dark rounded",children=[html.H6("Data Training")]),html.Div([tra],style={"margin-top":"15px","margin-right":"30px","margin-left":"30px"})]), 
                                    ]),
                                dbc.Col(className="col-sm-8 ",
                                    children=[
                                        dbc.Card(className="footy text-dark shadow rounded",children=[dcc.Graph(figure=figure_training,style={"margin":"8px"})]), 
                                    ]),
                                dbc.Col(className="col-sm-12 ",
                                    children=[
                                        dbc.Card(className="footy text-dark shadow rounded",children=[dcc.Graph(figure=figure_error,style={"margin":"8px"})]), 
                                    ],style={"margin-top":"15px"}),
                        ])
                    
                    ])
    return (Body_training)

def test(n_clicks,value):
    data_no_normalisasi = SKRIPSI_LSTM.sk(value,originalData)
    forecast_ori_Sequences = SKRIPSI_LSTM.forecastSequenceProducer(data_no_normalisasi[1], sequenceLength)
    skenarioP = SKRIPSI_LSTM.sk(value,corpusData)
    forecastData = skenarioP[1]
    forecastSequences = SKRIPSI_LSTM.forecastSequenceProducer(forecastData, sequenceLength)

    hasil_predict = SKRIPSI_LSTM.prediksi(forecast_ori_Sequences,forecastSequences,lstm,max_ex,min_ex,sequenceLength)
    waktu = hasil_predict[0]
    real = hasil_predict[1]
    prediksi = hasil_predict[2]
    
    MAPE = hasil_predict[3]
    print (MAPE)
    accuracy = hasil_predict[4]
    print(accuracy)
    MSE = hasil_predict[5]
    tbl_lstm = hasil_predict[6]
    # x,y=SKRIPSI_LSTM.intersection(waktu,real,waktu,prediksi)

    forecastData_no_normalisasi = pd.DataFrame(data=data_no_normalisasi[1],columns=["date","x(close)"])
    df1 = forecastData_no_normalisasi

    # data testing
    fig_data_testing = []
    fig_data_testing.append(go.Scatter(x= df1["date"].tolist(), y= df1["x(close)"].tolist(), mode='lines', name='Test'))
    figure_data_testing = figur(fig_data_testing)

    data1 = df1.to_dict('rows')
    columns1 =  [{"name": i, "id": i,} for i in (df1.columns)]
    fore = dt.DataTable(data=data1, columns=columns1,page_size=5,style_cell={'textAlign': 'center',"color":"#09203f"},style_header={
        "background-image": "linear-gradient(to top, #09203f 0%, #537895 100%)",
        'fontWeight': 'bold',
        "color":"white"
    })

    # tabel hasil prediksi 
    fig_data_hasil = []
    fig_data_hasil.append(go.Scatter(x= tbl_lstm["times"].tolist(), y= tbl_lstm["real"].tolist(), mode='lines', name='real'))
    fig_data_hasil.append(go.Scatter(x= tbl_lstm["times"].tolist(), y= tbl_lstm["prediksi"].tolist(), mode='lines', name='prediksi'))   
    figure_hasil = figur(fig_data_hasil)

    data2 = tbl_lstm.to_dict('rows')
    columns2 =  [{"name": i, "id": i,} for i in (tbl_lstm.columns)]
    tbl_hasil = dt.DataTable(data=data2, columns=columns2,page_size=5,style_cell={'textAlign': 'center',"color":"#09203f"},style_header={
        "background-image": "linear-gradient(to top, #09203f 0%, #537895 100%)",
        'fontWeight': 'bold',
        "color":"white"
    })

    Body_testing = dbc.CardBody(
                    className="",
                    children=[
                        dbc.Row([
                            dbc.Col(className="col-sm-4 ",
                                    children=[
                                    dbc.Card(className="footy text-dark shadow rounded",
                                        children=[dbc.CardHeader(className="footy text-dark rounded",children=[html.H6("Data Testing")]),html.Div([fore],style={"margin-top":"15px","margin-right":"30px","margin-left":"30px"})]), 
                                        ]),
                                    dbc.Col(className="col-sm-8 ",
                                        children=[
                                            dbc.Card(className="footy text-dark shadow rounded",children=[dcc.Graph(figure=figure_data_testing,style={"margin":"8px"})]), 
                                        ]),
                                    dbc.Col(
                                        style={"margin-top":"15px"},
                                        className="col-sm-12",
                                        children=[
                                            dbc.Card(className="footy text-dark shadow rounded",children=[
                                                dcc.Graph(figure=figure_hasil),
                                            dbc.Row([
                                                dbc.Col([dbc.Card(className="content shadow rounded",style={"margin":"15px"},children=[html.Div(children=["MAPE %"]),html.Div([MAPE])]),
                                                ]),
                                                dbc.Col([dbc.Card(className="content shadow rounded",style={"margin":"15px"},children=[html.Div(children=["Akurasi %"]),html.Div([accuracy])]),
                                                ]),
                                                dbc.Col([dbc.Card(className="content shadow rounded",style={"margin":"15px"},children=[html.Div(children=["MSE"]),html.Div([MSE])]), 
                                                ])
                                                ])
                                            ],       
                                        ),
                                    ]),
                                    dbc.Col(
                                        className="col-sm-12",
                                        children=[
                                        dbc.Card(className="footy text-dark shadow rounded",children=[
                                            dbc.CardHeader(className="footy text-dark shadow rounded",children=[html.H5("Hasil Prediksi")]),
                                            dbc.CardBody([html.Div(children=[tbl_hasil],style={"margin-top":"15px","margin-right":"50px","margin-left":"50px"})])
                                        ]) 
                                        ],style={"margin-top":"15px"}),
                    ])
                ])
    return Body_testing

def prediksi(n_clicks,input4,value):
    skenarioP = SKRIPSI_LSTM.sk(value,corpusData)
    forecastData = skenarioP[1]
    forecastSequences = SKRIPSI_LSTM.forecastSequenceProducer(forecastData, sequenceLength)

    hasil_Mypredict = SKRIPSI_LSTM.myprediksi(forecastSequences,lstm,max_ex,min_ex,input4)

    fig_data_hasil_Mypredict = []
    fig_data_hasil_Mypredict.append(go.Scatter(x= hasil_Mypredict["date"].tolist(), y= hasil_Mypredict["close"].tolist(), mode='lines', name='prediksi'))  
    figure_hasil = figur(fig_data_hasil_Mypredict)

    data3 = hasil_Mypredict.to_dict('rows')
    columns3 = [{"name":i,"id":i,} for i in (hasil_Mypredict.columns)]
    tbl_hasil = dt.DataTable(data=data3, columns=columns3,page_size=5,style_cell={'textAlign': 'center',"color":"#09203f"},style_header={
        "background-image": "linear-gradient(to top, #09203f 0%, #537895 100%)",
        'fontWeight': 'bold',
        "color":"white"
    })

    Body_prediksi = dbc.CardBody(
                    className="",
                    children=[
                        dbc.Row([
                            dbc.Col(className="col-sm-4 ",
                                    children=[
                                    dbc.Card(className="footy text-dark shadow rounded",
                                        children=[dbc.CardHeader(className="footy text-dark rounded",children=[html.H6("Data Prediksi")]),html.Div([tbl_hasil],style={"margin-top":"15px","margin-right":"30px","margin-left":"30px"})]), 
                                        ]),
                                    dbc.Col(className="col-sm-8 ",
                                        children=[
                                            dbc.Card(className="footy text-dark shadow rounded",children=[dcc.Graph(figure=figure_hasil,style={"margin":"8px"})]), 
                                        ]),
                    ])
                ])
    return Body_prediksi


if __name__ == "__main__":
    app.run_server(debug=True)

