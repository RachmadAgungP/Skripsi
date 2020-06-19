import SKRIPSI_LSTM
from obj.grafik import figur
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import dash_table as dt
import plotly.graph_objs as go

def data(n_clicks, input1, originalData):
    
    data_no_normalisasi = SKRIPSI_LSTM.sk(input1,originalData)
    # data tabel saham semuanya 
    originalData_no_normalisasi = pd.DataFrame(data=originalData,columns=["date","x(close)"])
    
    fig_ori = []
    fig_ori.append(go.Scatter(x= originalData_no_normalisasi["date"].tolist(), y= originalData_no_normalisasi["x(close)"].tolist(), mode='lines', name='close'))
    figure_ori = figur(fig_ori)

    # data tabel saham 
    data_ori = originalData_no_normalisasi.to_dict('rows')
    columns_ori =  [{"name": i, "id": i,} for i in (originalData_no_normalisasi.columns)]
    tbl_ori = dt.DataTable(data=data_ori, columns=columns_ori,page_size=5,style_cell={'textAlign': 'center',"color":"#09203f" },style_header={
        "background-image": "linear-gradient(to top, #09203f 0%, #537895 100%)",
        'fontWeight': 'bold',
        "color":"white"
    })
    Body_Data_Saham = dbc.CardBody(
                        children=[
                            dbc.Row(
                                [
                                    dbc.Col(
                                        className="col-sm-4",
                                        children=[
                                            dbc.Card(
                                                className="footy text-dark shadow rounded",
                                                children=[
                                                    dbc.CardHeader(
                                                        className="footy text-dark rounded",
                                                        children=[html.H6("Data SAHAM")]),
                                                    html.Div(
                                                        children=[tbl_ori],
                                                        style={"margin-top":"15px","margin-right":"30px","margin-left":"30px"}
                                                    )
                                                ]
                                            ), 
                                        ]
                                    ),
                                    dbc.Col(
                                        className="col-sm-8",
                                        children=[
                                            dbc.Card(
                                                className="footy text-dark shadow rounded",
                                                children=[
                                                    dcc.Graph(figure=figure_ori,style={"margin":"8px"})
                                                ]
                                            ),   
                                        ],          
                                    ),
                                ]
                            ),
                        ]
                    )
    
    return (Body_Data_Saham)

def train(n_clicks,input1,input2,value, corpusData,sequenceLength,max_ex,min_ex,originalData):
    print ("value t",value)
    
    W = [   
            [-0.245714286	,0.850360602	,0.029262045	,0.184398087]
            ,[0.868020398	,0.860429754	,-0.379580925	,0.079506914]
            ,[-0.206444161	,-0.24856166	,-0.085253247	,0.25112624	]
            ,[0.842874383	,-0.324206065	,0.907722829	,-0.593738792]
        ]
    
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
    
    # data tabel training
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
                            dbc.Row(
                                [
                                    dbc.Col(
                                        className="col-sm-4 ",
                                        children=[
                                            dbc.Card(
                                                className="footy text-dark shadow rounded",
                                                children=[
                                                    dbc.CardHeader(
                                                        className="footy text-dark rounded",
                                                        children=[html.H6("Data Training")]
                                                    ),
                                                    html.Div([tra],style={"margin-top":"15px","margin-right":"30px","margin-left":"30px"})
                                                ]
                                            ), 
                                        ]
                                    ),
                                    dbc.Col(
                                        className="col-sm-8 ",
                                        children=[
                                            dbc.Card(
                                                className="footy text-dark shadow rounded",
                                                children=[dcc.Graph(figure=figure_training,style={"margin":"8px"})]
                                            ), 
                                        ]
                                    ),
                                    dbc.Col(
                                        className="col-sm-12 ",
                                        children=[
                                            dbc.Card(
                                                className="footy text-dark shadow rounded",
                                                children=[dcc.Graph(figure=figure_error,style={"margin":"8px"})]
                                            ), 
                                        ],
                                        style={"margin-top":"15px"}
                                    ),
                        ])
                    
                    ])
    
    return (Body_training)

def test(n_clicks,value, originalData,sequenceLength,corpusData,lstm,max_ex,min_ex):
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

    # data tabel testing
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

    # tabelhasil prediksi 
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
                                dbc.Col(
                                    className="col-sm-4 ",
                                    children=[
                                        dbc.Card(className="footy text-dark shadow rounded",
                                            children=[
                                                dbc.CardHeader(
                                                    className="footy text-dark rounded",
                                                    children=[html.H6("Data Testing")]
                                                ),
                                                html.Div([fore],style={"margin-top":"15px","margin-right":"30px","margin-left":"30px"})
                                            ]
                                        ), 
                                    ]
                                ),
                                dbc.Col(className="col-sm-8 ",
                                    children=[
                                        dbc.Card(className="footy text-dark shadow rounded",children=[dcc.Graph(figure=figure_data_testing,style={"margin":"8px"})]), 
                                    ]
                                ),
                                dbc.Col(
                                    style={"margin-top":"15px"},
                                    className="col-sm-12",
                                    children=[
                                        dbc.Card(
                                            className="footy text-dark shadow rounded",
                                            children=[
                                                dcc.Graph(figure=figure_hasil),
                                                dbc.Row([
                                                    dbc.Col([
                                                        dbc.Card(
                                                            className="content shadow rounded",
                                                            style={"margin":"15px"},
                                                            children=[html.Div(children=["MAPE %"]),html.Div([MAPE])]),
                                                    ]),
                                                    dbc.Col([
                                                        dbc.Card(
                                                            className="content shadow rounded",
                                                            style={"margin":"15px"},
                                                            children=[html.Div(children=["Akurasi %"]),html.Div([accuracy])]),
                                                    ]),
                                                    dbc.Col([
                                                        dbc.Card(
                                                            className="content shadow rounded",
                                                            style={"margin":"15px"},
                                                            children=[html.Div(children=["MSE"]),html.Div([MSE])]), 
                                                    ])
                                                ])
                                            ],       
                                        ),
                                    ]
                                ),
                                dbc.Col(
                                    className="col-sm-12",
                                    children=[
                                        dbc.Card(
                                            className="footy text-dark shadow rounded",
                                            children=[
                                                dbc.CardHeader(
                                                    className="footy text-dark shadow rounded",
                                                    children=[html.H5("Hasil Prediksi")]),
                                                dbc.CardBody([html.Div(children=[tbl_hasil],style={"margin-top":"15px","margin-right":"50px","margin-left":"50px"})])
                                            ]
                                        ) 
                                    ],
                                    style={"margin-top":"15px"}
                                ),
                            ])
                        ]
                    )
    return Body_testing

def prediksi(n_clicks,input4,value, corpusData,sequenceLength,lstm,max_ex,min_ex):
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
                                        dbc.Card(
                                            className="footy text-dark shadow rounded",
                                            children=[
                                                dbc.CardHeader(
                                                    className="footy text-dark rounded",
                                                    children=[html.H6("Data Prediksi")]),
                                                html.Div([tbl_hasil],
                                            style={"margin-top":"15px","margin-right":"30px","margin-left":"30px"})
                                            ]
                                        ), 
                                    ]
                                ),
                                dbc.Col(
                                    className="col-sm-8 ",
                                    children=[
                                        dbc.Card(className="footy text-dark shadow rounded",children=[dcc.Graph(figure=figure_hasil,style={"margin":"8px"})]), 
                                    ]
                                ),
                            ])
                        ]
                    )
    return Body_prediksi
