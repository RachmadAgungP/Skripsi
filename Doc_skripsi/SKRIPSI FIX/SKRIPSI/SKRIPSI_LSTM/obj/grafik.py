import plotly.graph_objs as go

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
                            size=12,),
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