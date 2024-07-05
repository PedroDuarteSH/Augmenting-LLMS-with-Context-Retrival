from ast import mod
import plotly as plt
import numpy as np
import plotly.express as px
import scipy.optimize as sp
import pandas as pd

import plotly.graph_objects as go
def linear_fit(num_tokens, a, b):
    return a* num_tokens + b 


def plotTime(num_tokens, generation_time):
    index = pd.notna(generation_time)
    
    x_values = np.array(num_tokens[index])
    y_values = np.array(generation_time[index])

    popt, _ = sp.curve_fit(linear_fit, x_values, y_values)

    x_new = np.arange(100, 2048)

    y_new = linear_fit(x_new, *popt)

    fig = px.line(x=x_new, y=y_new, labels={"x": "Number of Tokens", "y": "Generation Time (s)"}, title="Generation Time per Number of Tokens")
    
    sample_size = int(0.02 * len(num_tokens))
    random_indexes =  np.random.choice(len(x_values), size=sample_size, replace=False)

    x_points_random = x_values[random_indexes]
    y_points_random = y_values[random_indexes]

    # Compute y-values using the fitted models
    y_pred_linear = linear_fit(x_values, *popt)
    # Calculate R^2 for linear fit
    ss_res_linear = np.sum((y_values - y_pred_linear) ** 2)
    ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
    r2_linear = 1 - (ss_res_linear / ss_tot)
    print(r2_linear)
    
    fig.add_scatter(x =  x_points_random, y = y_points_random, mode="markers", opacity=0.2) 


    fig.show()


def compareTime(num_tokens_list, generation_time_list, legend, limit = 2048):
    fig = go.Figure()
    annotations = []
    max_y = 0
    for i, num_tokens in enumerate(num_tokens_list):
        generation_time = generation_time_list[i]
        index = pd.notna(generation_time)

        x_values = np.array(num_tokens[index])
        y_values = np.array(generation_time[index])

        average_generation_time = np.average(y_values)


        popt, _ = sp.curve_fit(linear_fit, x_values, y_values)

        x_new = np.arange(100, limit)

        y_new = linear_fit(x_new, *popt)

        max_y = max(np.max(y_new), max_y)
        fig.add_scatter(x=x_new, y=y_new, mode="lines", name = legend[i] + " Trendline")
        
       

        # Compute y-values using the fitted models
        y_pred_linear = linear_fit(x_values, *popt)
        # Calculate R^2 for linear fit
        ss_res_linear = np.sum((y_values - y_pred_linear) ** 2)
        ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
        r2_linear = 1 - (ss_res_linear / ss_tot)
        print(r2_linear)
        formula = f"{popt[0]:.3e} * num_tokens + {popt[1]:.3e}"
        annotations.append(dict(
                x=2048,
                y=y_new[-1],
                xref="x",
                yref="y",
                text=formula,

                ax=100,
                ay=-10
            )
        )

        if len(num_tokens > 100000):
            sample_size = int(0.02 * len(num_tokens))
            random_indexes =  np.random.choice(len(x_values), size=sample_size, replace=False)

            x_values = x_values[random_indexes]
            y_values = y_values[random_indexes]
        fig.add_scatter(x =  x_values, y = y_values, mode="markers", opacity=0.2, name = legend[i] + " Points") 
        
        fig.add_scatter(
            x=[0, limit],
            y=[average_generation_time, average_generation_time],
            mode="lines",
            name=f"{legend[i]} - Average")
        
        annotations.append(dict(
                x=2048,
                y=average_generation_time,
                xref="x",
                yref="y",
                text=f"{average_generation_time:.2f}",

                ax=100,
                ay=-10
            )
        )
        
    full_fig = fig.full_figure_for_development()
    current_yaxis_range = full_fig.layout.yaxis.range
 
    min_y = current_yaxis_range[0] if current_yaxis_range else 0
    
    fig.update_layout(annotations=annotations,
                      yaxis=dict(range=[min_y, max_y + 0.05 * max_y]))
    fig.show()