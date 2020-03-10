import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash
from dash.dependencies import Input, Output

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div(
    [
        dcc.Input(id="dfalse", type="number", 
        placeholder="Panjang Memory"
        ),
        dcc.Input(
            id="dtrue", type="number",
            debounce=True, placeholder="Banyak Epoch",
        ),
        dcc.Input(
            id="input_range", type="number", 
            placeholder="Sekenario",
            min=1, max=5, step=1,
        ),
        dcc.Input(
            id="input_rate", type="number", 
            placeholder="Learning rate",
            min=0.01, max=1, step=0.01,
        ),
        html.Hr(),
        html.Div(id="number-out"),
    ]
)


@app.callback(
    Output("number-out", "children"),
    [Input("dfalse", "value"), Input("dtrue", "value"), Input("input_range", "value"), Input("input_rate", "value")],
)

def number_render(fval, tval, rangeval, Leaval):
    return "Panjang Memory : {}, Banyak Epoch : {}, Skenario ke : {}, Learning_Rate : {}".format(fval, tval, rangeval,Leaval)

if __name__ == "__main__":
    app.run_server(debug=True)