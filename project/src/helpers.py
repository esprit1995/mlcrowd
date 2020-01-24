from sklearn.datasets import make_swiss_roll
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from pathlib import Path
import plotly.graph_objects as go

filepath = "../../../bigassdata/GoogleNews-vectors-negative300.bin"


def create_word2vec_sample(n_samples, filename):
    """
    Create a sample dataset of word2vec embeddings
    :param n_samples: size of dataset to create (int)
    :param filename: name for the file in format "name.csv" - string
    :return: void
    """
    wv_from_bin = KeyedVectors.load_word2vec_format(filepath, binary=True)

    COUNT = 0
    coeff_arr = np.empty((n_samples, 300))
    word_arr = list()
    for word, vector in zip(wv_from_bin.vocab, wv_from_bin.vectors):
        coefs = np.asarray(vector, dtype='float32')
        word_arr.append(word)
        coeff_arr[COUNT, :] = coefs
        COUNT += 1
        if COUNT == n_samples:
            break
    df = pd.DataFrame(columns=[str(i) for i in range(1, 301)], data=coeff_arr)
    df["word"] = word_arr
    df.to_csv(Path("../samples_word2vec") / filename)


def make_NoisySwissRoll(n_samples, n_noise):
    """
    Create a classic Swiss Roll dataset with additional noise columns
    :param n_samples: nrows for the dataset
    :param n_noise: number of additional noise columns
    :return: (created noisy dataset (numpy), corresponding dataset with no noise (numpy), colors)
    """
    data, colors = make_swiss_roll(n_samples=n_samples)
    data_df = pd.DataFrame(columns=[str(i) for i in range(3)], data=data)
    for j in range(n_noise):
        data_df[str(j + 3)] = pd.Series(np.random.rand(n_samples))
    return data_df.to_numpy(), data_df[["0", "1", "2"]].to_numpy(), colors


def draw_3d_plotly(data, colors=None):
    """
    Draw a 3d plot for given data.
    :param data: data in numpy format. Must have 3 dimensions.
    :param colors: optional, if coloring of points is wanted.
    :return:
    """
    xdata = data[:,0]
    ydata = data[:,1]
    zdata = data[:,2]

    fig = go.Figure(data=[go.Scatter3d(x=xdata,
                                       y=ydata,
                                       z=zdata,
                                       mode='markers',
                                       marker = dict(
                                                size=12,
                                                color=colors,
                                                colorscale='Viridis',
                                                opacity=0.8))])

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()
