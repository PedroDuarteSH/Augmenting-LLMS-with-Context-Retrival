from tkinter.font import ROMAN
import plotly as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def plotTokensPercentil(num_tokens, limit = 2048):
    num_tokens = sorted(num_tokens)

    fig = go.Figure()
    percentiles = list(range(1,101,1))


    y = [num_tokens[int(x)] if len(num_tokens[:int(x)]) else 0 for x in np.percentile(range(len(num_tokens)), percentiles)]


    x_limit_pos = 0
    y_limit_pos = 0
    for i in range(len(y)):
        if y[i] > limit:
            x_limit_pos = i + 1
            y_limit_pos = y[i]
            break


    fig.add_trace(go.Scatter(
        x=[x_limit_pos, x_limit_pos],
        y=[0, y_limit_pos],
        mode="lines",
        line=go.scatter.Line(color="red", dash="dash"),
        showlegend=False))

    fig.add_trace(go.Scatter( 
        x=[0, x_limit_pos],
        y=[y_limit_pos, y_limit_pos],
        mode="lines",
        line=go.scatter.Line(color="red", dash="dash"),
        showlegend=False))
    fig.add_trace(go.Scatter(
        x=percentiles,
        y=y,
        mode='lines',
        line=dict(color='blue'),  # Change 'red' to your desired color
        name='Average Number of Tokens'
    ))


    fig.update_layout(
        title="",
        xaxis_title="Percentage of Considered Instances",
        yaxis_title="Number of Tokens",
        showlegend=False,
        # Add xaxis values
    )

    annotations=[
        dict(
            x=x_limit_pos,
            y=y_limit_pos,
            xref="x",
            yref="y",
            text=str(x_limit_pos) + "%",
        
            ax=0,
            ay=-40
        ),

    ]
    fig.update_layout(annotations=annotations)
    fig.show()

    del num_tokens



def compareTokensPercentiles(num_tokens_lists : list[pd.Series], legend, limit = 2048):
    
    num_figs = len(num_tokens_lists)
    
    legend_pos = 100 / (num_figs + 1)

    fig = go.Figure()
    percentiles = list(range(1,101,1))


    
    fig.add_trace( go.Scatter(
            x=[0, 101],
            y=[limit, limit],
            mode="lines",
            line=go.scatter.Line(color="red", dash="dash"),
            showlegend=False))

    fig.update_layout(
            title="",
            xaxis_title="Percentage of Considered Instances",
            yaxis_title="Number of Tokens",
 
        )
    
    annotations = []
    for i, num_tokens in enumerate(num_tokens_lists):
        average_num_tokens = np.average(num_tokens[num_tokens.notna()])
        num_tokens = sorted(num_tokens)
        y = [num_tokens[int(x)] if len(num_tokens[:int(x)]) else 0 for x in np.percentile(range(len(num_tokens)), percentiles)]


        x_limit_pos = 0
        y_limit_pos = 0
        for j in range(len(y)):
            if y[j] > limit:
                x_limit_pos = j + 1
                y_limit_pos = y[j]
                break

        
        fig.add_trace(go.Scatter(
            x=percentiles,
            y=y,
            mode='lines',
            name=f"{legend[i]}"
        ))
        
        
        print(average_num_tokens)
        fig.add_trace(go.Scatter(
            x=[0, 101],
            y=[average_num_tokens, average_num_tokens],
            mode="lines",
            name=f"{legend[i]} - Average")
        )

        annotations.append(dict(
                    x=legend_pos * (i+1),
                    y=average_num_tokens,
                    xref="x",
                    yref="y",
                    text=f"{average_num_tokens:.2f}",
                    ax=0,
                    ay=-20
                )
            )       


        if x_limit_pos != 0 and y_limit_pos != 0:
            annotations.append(dict(
                    x=x_limit_pos,
                    y=y_limit_pos,
                    xref="x",
                    yref="y",
                    text=str(x_limit_pos) + "%",
                    ax=0,
                    ay=-40
                )
            )       

        
        

        del num_tokens
    fig.update_layout(annotations=annotations)
    fig.show()
