import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
def Header_Data():
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
    return (Header_Data_Saham)

def Header_train():
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
    return Header_training

def Header_test():
    Header_testing = dbc.CardHeader(
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
                                html.Button(className="btn content",id="submit-button3", n_clicks=0, children='Testing'),
                            ]
                        ),
                ])
    return Header_testing

def Header_pred():
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
                            ),
                        ]
                    ),
                ]
            )
    return Header_prediksi