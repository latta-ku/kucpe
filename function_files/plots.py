import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='notebook'

def return_start_end_date(df):
    # Expecting year month day columns
    start_year, start_month, start_day = str(df["year"].iloc[0]), str(df["month"].iloc[0]), str(df["day"].iloc[0])
    end_year, end_month, end_day = str(df["year"].iloc[-1]), str(df["month"].iloc[-1]), str(df["day"].iloc[-1])

    start_date = str(pd.to_datetime(start_year + "-" + start_month + "-" + start_day, format='%Y-%m-%d').date())
    end_date = str(pd.to_datetime(end_year + "-" + end_month + "-" + end_day, format='%Y-%m-%d').date())

    return start_date, end_date

def plot_type_0(df, name = None, webgl = True):
    # Just plot values of water_lv column

    fig = go.Figure()

    # df = df[0].copy()
    x = df["year"].astype(str) + "-" + df["month"].astype(str) + "-" + df["day"].astype(str) + " " + df["time"]
    y = df["water_lv"]

    start_date, end_date = return_start_end_date(df)

    if (webgl):
        fig.add_trace(go.Scattergl(x=x,
                                y=y, 
                                line=dict(width=2, color = "black"),
                                name = "Water Level",
                                marker=dict(size=2),
                                mode = "lines+markers"
                                )
                    )
    else:
        fig.add_trace(go.Scatter(x=x,
                        y=y, 
                        line=dict(width=2, color = "black"),
                        name = "Water Level",
                        marker=dict(size=2),
                        mode = "lines+markers"
                        )
            )

    fig.update_layout(title=f'{name} ({start_date} to {end_date})',
                      xaxis_title='Date (Year-Month-Day)',
                      xaxis = {'tickformat': '%Y-%m-%d %I:%M %p',
                               #'tickformat': '%Y-%m-%d',
                               'range' : (list(x)[0], list(x)[-1])
                      },
                      yaxis_title='Water Level (m msl)',
                      width=1400,
                      height=600,
                      template = 'plotly_white',
                      showlegend = True,
                      legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="left",
                            x=0
                        )
                     )

    fig.update_xaxes(gridwidth=1.4, gridcolor='LightGrey')
    fig.update_yaxes(gridwidth=1.4, gridcolor='LightGrey')
    fig.update_yaxes(zeroline=True, zerolinewidth=1.4, zerolinecolor='Grey')

    return fig

def plot_type_1(df_ori, df_new = None, name = None, new_water_lv = True, interval = True, webgl = True):
    # Plot anomaly dots with or without interval
    fig = go.Figure()

    datetime = df_new["year"].astype(str) + "-" + df_new["month"].astype(str) + "-" + df_new["day"].astype(str) + " " + df_new["time"]
    x = list(datetime)
    y = df_ori["water_lv"]
    
    anomaly_x = datetime[df_new["anomaly_type"] == 1]
    anomaly_y = df_ori[df_new["anomaly_type"] == 1]["water_lv"]

    start_date, end_date = return_start_end_date(df_new)

    if (webgl):
        scatter_plot = go.Scattergl
    else:
        scatter_plot = go.Scatter

    if (interval):
        x_rev = x[::-1]
        upper = df_new["upper"].to_list()
        lower = df_new["lower"].to_list()
        
        fig.add_trace(scatter_plot(x=x+x_rev,
                                   y=upper+lower[::-1], 
                                   line=dict(width=2,
                                             color='grey'),
                                   name = "PI",
                                   # mode = "markers",
                                   fill='tozerox',
                                   opacity = 0.5
                                  )
                     ) 
    
    if (new_water_lv):
        y2= df_new["water_lv"]
        fig.add_trace(scatter_plot(x=x,
                                y=y2, 
                                line=dict(width=2,
                                            color='blue'),
                                name = "New Water Level",
                                marker=dict(size=2),
                                mode = "lines+markers"
                                )
                    )    
    
    fig.add_trace(scatter_plot(x=x,
                               y=y, 
                               line=dict(width=2,
                                         color='black'),
                               name = "Original Water Level",
                               marker=dict(size=2),
                               mode = "lines+markers"
                              )
                 )
    
    fig.add_trace(scatter_plot(x=anomaly_x,
                               y=anomaly_y, 
                               line=dict(width=3),
                               name = "Detected Anomalies",
                               mode = "markers",
                               marker=dict(size=7.5,
                                           color = "red",
                                           line=dict(color='black',
                                                     width=0.5
                                                    )
                                          )
                              )
                 )

    fig.update_layout(title=f'{name} ({start_date} to {end_date})',
                      xaxis_title='Date (Year-Month-Day)',
                      xaxis = {'tickformat': '%Y-%m-%d %I:%M %p',
                               #'tickformat': '%Y-%m-%d',
                               'range' : (list(x)[0], list(x)[-1])
                      },
                      yaxis_title='Water Level (m msl)',
                      width=1400,
                      height=600,
                      template = 'plotly_white',
                      legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="left",
                            x=0
                        )
                     )
    fig.update_xaxes(gridwidth=1.4, gridcolor='LightGrey')
    fig.update_yaxes(gridwidth=1.4, gridcolor='LightGrey')
    fig.update_yaxes(zeroline=True, zerolinewidth=1.4, zerolinecolor='LightGrey')
    
    return fig

def plot_type_2(df, name = None, webgl = True):
    # REDACTED
    pass

def plot_type_3(data, inv_yhat, step = None, rev = False):   
    # REDACTED
    pass

def plot_type_4(data_ori, data_new, name = None):
    # REDACTED
    pass

def plot_type_5(data, data_predict, cut_range_list, name = None):
    # For lstm prediction

    x = data["year"].astype(str) + "-" + data["month"].astype(str) + "-" + data["day"].astype(str) + " " + data["time"]
    x = x.to_list()
    y1 = data_predict["water_lv"]

    y21 = [np.nan for i in range(len(data_predict.index))]
    y22 = [np.nan for i in range(len(data_predict.index))]
    y23 = [np.nan for i in range(len(data_predict.index))]
    y3 = [np.nan for i in range(len(data_predict.index))]

    for cut_range in cut_range_list:
        y21[cut_range[0] : cut_range[1] + 1] = data_predict["water_lv_forward"].iloc[cut_range[0] : cut_range[1] + 1].to_list()
        y22[cut_range[0] : cut_range[1] + 1] = data_predict["water_lv_backward"].iloc[cut_range[0] : cut_range[1] + 1].to_list()
        y23[cut_range[0] : cut_range[1] + 1] = data_predict["water_lv_combined"].iloc[cut_range[0] : cut_range[1] + 1].to_list()
        y3[cut_range[0] : cut_range[1] + 1] = data["water_lv"].iloc[cut_range[0] : cut_range[1] + 1].to_list()

    fig = go.Figure()

    fig.add_trace(go.Scattergl(x=x,
                                y=y21, 
                                line=dict(width=4, color = "purple"),
                                name = "forward",
                                opacity = 0.5,
                                marker=dict(size=4),
                                mode = "lines+markers"
                            )
                 )

    fig.add_trace(go.Scattergl(x=x,
                                y=y22, 
                                line=dict(width=4, color = "orange"),
                                name = "backward",
                                opacity = 0.5,
                                marker=dict(size=4),
                                mode = "lines+markers"
                            )
                 )

    fig.add_trace(go.Scattergl(x=x,
                                y=y23, 
                                line=dict(width=4, color = "black"),
                                name = "combined",
                                opacity = 1,
                                marker=dict(size=4),
                                mode = "lines+markers"
                            )
                 )

    fig.add_trace(go.Scattergl(x=x,
                                y=y1, 
                                line=dict(width=2, color = "blue"),
                                name = "original",
                                marker=dict(size=2),
                                mode = "lines+markers"
                                #opacity = 0.2
                            )
                 )


    fig.add_trace(go.Scattergl(x=x,
                                y=y3, 
                                line=dict(width=5, color = "black"),
                                name = "original",
                                opacity = 0.2,
                                marker=dict(size=5),
                                mode = "lines+markers"
                            )
                 )


    fig.update_layout(title=f'{name}',
                      xaxis_title='Date (Year-Month-Day)',
                      xaxis = {
                       'tickformat': '%Y-%m-%d %I:%M %p',
                       'range' : (list(x)[0], list(x)[-1])
                      },
                      yaxis_title='Water Level (m msl)',
                      width=1400,
                      height=600,
                      template = 'plotly_white',
                      legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="left",
                            x=0
                        )
                     )

    fig.update_xaxes(gridwidth=1.4, gridcolor='LightGrey')
    fig.update_yaxes(gridwidth=1.4, gridcolor='LightGrey')
    fig.update_yaxes(zeroline=True, zerolinewidth=1.4, zerolinecolor='Grey')
    
    return fig

def plot_type_6(df_ori, df_new, name = None, webgl = True):
    # REDACTED
    pass

def plot_type_7(df1, df2, name1 = None, name2 = None, webgl = True):
    # REDACTED
    pass