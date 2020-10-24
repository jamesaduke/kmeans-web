import numpy as np
import pickle
import pandas as pd
# visualization
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from sklearn.cluster import KMeans
from sklearn.cluster import KMeans

from kmeansmodel import model
model
import kmeans

import streamlit as st


# #
# pickle_in = open("segmenter.pkl", 'rb')
# updated_kmeans_model = pickle.load(pickle_in)
print(customers)

# def cluster_groups():
#     cluster = KMeans(n_clusters=3)
#     fit = cluster.fit_predict(customers.iloc[:, 3:])
#     print(fit)


# print(cluster_groups())

# def main():
#     st.title("Customer Segmenter")
#
#     clusters = st.text_input("Clusters")
#     cluster_groups()
#
#
# if __name__ == '__main__':
#     main()
