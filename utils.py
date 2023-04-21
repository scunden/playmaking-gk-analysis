import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from matplotlib.ticker import FixedLocator, FixedFormatter
import matplotlib as mpl


def generate_stats(df, player, metrics, stats='Statistics'):
    df_temp = df.loc[df.Player==player][metrics].transpose()
    if df_temp.shape[1]>1:
        df_temp = pd.DataFrame(df_temp.mean(axis=1))
    df_temp.columns=[stats]
    return df_temp


def plot_radar(player_dict, categories, title, value='Statistics'):

  fig = go.Figure()

  for player in player_dict.keys():
    df_temp = player_dict[player]

    fig.add_trace(go.Scatterpolar(
          r=df_temp.loc[categories][value].values,
          theta=categories,
          fill='toself',
          name=player,
          
    ))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True,
  )
  fig.update_layout(title_text=title, title_x=0.5)
  fig.update_layout(
    width=800,
    height=600,)
  fig.update_layout(polar = dict(radialaxis = dict(showticklabels = False)))
  fig.show()
  return fig

def silhouette_coefficients(X, k_models, k_range, ss, figsize=(11, 9), savepath='images', label='raisin', model='k'):
    plt.figure(figsize=figsize)

    for idx, k in enumerate(k_range):
        plt.subplot(2, 2, idx+1)
        
        if model=='k':
            y_pred = k_models[k - 1].labels_
        else:
            y_pred = k_models[k - 1].predict(X)
            
        silhouette_coefficients = silhouette_samples(X, y_pred)

        padding = len(X) // 30
        pos = padding
        ticks = []
        for i in range(k):
            coeffs = silhouette_coefficients[y_pred == i]
            coeffs.sort()

            color = mpl.cm.Spectral(i / k)
            plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                            facecolor=color, edgecolor=color, alpha=0.7)
            ticks.append(pos + len(coeffs) // 2)
            pos += len(coeffs) + padding

        plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
        plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
        if idx in (0, 2):
            plt.ylabel("Cluster")
        
        if idx in (2, 3):
            plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.xlabel("Silhouette Coefficient")
        else:
            plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.tick_params(labelbottom=False)

        plt.axvline(x=ss[k - 2], color="red", linestyle="--")
        plt.title("$k={}$".format(k), fontsize=16)

    plt.savefig("{}\ss_coef_{}.png".format(savepath, label))