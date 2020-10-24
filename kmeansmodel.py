import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
# from IPython import get_ipython
from plotly.subplots import make_subplots
#
# # for data preprocessing and clustering
from sklearn.cluster import KMeans

# Streamlit
import streamlit as st

#
# get_ipython().run_line_magic('matplotlib', 'inline')
# # to include graphs inline within the frontends next to code
#
# get_ipython().run_line_magic('config', "InlineBackend.figure_foasrmat='retina'")
# # to enable retina (high resolution) plots

pd.options.mode.chained_assignment = None


# to bypass warnings in various dataframe assignments


# **Investigate data**
def model(n_groups):
    # load data into a dataframe
    customers_orders = pd.read_csv("Orders - Analysis Task.csv")

    # first rows of the dataset
    # print(customers_orders.head())

    # first glance of customers_orders data
    customers_orders.info()

    # descriptive statistics of the non-object columns
    customers_orders.describe()

    # There were significant number of rows whose `ordered_item_quantity` is 0 and `net_quantity` is less than 0,
    # which means they are not ordered/sold at all; but the fact that they have returns requires investigation.

    print("Number of rows that net quantity is negative:",
          customers_orders[customers_orders.net_quantity < 0].shape[0])

    # **These rows will be excluded from the orders dataset for the project.**

    # exclude not sold/ordered SKUs from the dataset
    customers_orders = customers_orders[
        customers_orders["ordered_item_quantity"] > 0]

    # ## 1. Products ordered
    # It is the count of the products ordered in product_type column by a customer. <br>

    # **Create functions to identify customers who order multiple products**

    def encode_column(column):
        if column > 0:
            return 1
        if column <= 0:
            return 0

    def aggregate_by_ordered_quantity(dataframe, column_list):
        '''this function:
        1. aggregates a given dataframe by column list,
        as a result creates a aggregated dataframe by counting the ordered item quantities

        2. adds number_of_X ordered where X is the second element in the column_list
        to the aggregated dataframe by encoding ordered items into 1

        3. creates final dataframe containing information about
        how many of X are ordered, based on the first element passed in the column list'''

        aggregated_dataframe = (dataframe
                                .groupby(column_list)
                                .ordered_item_quantity.count()
                                .reset_index())

        aggregated_dataframe["products_ordered"] = (aggregated_dataframe
                                                    .ordered_item_quantity
                                                    .apply(encode_column))

        final_dataframe = (aggregated_dataframe
                           .groupby(column_list[0])
                           .products_ordered.sum()  # aligned with the added column name
                           .reset_index())

        return final_dataframe

    # apply functions to customers_orders
    customers = aggregate_by_ordered_quantity(customers_orders, ["customer_id", "product_type"])

    # print(customers.head())

    # ## 2. Average Return Rate It is the ratio of returned item quantity and ordered item quantity. This ratio is first
    # calculated per order and then averaged for all orders of a customer.

    # aggregate data per customer_id and order_id,
    # to see ordered item sum and returned item sum
    ordered_sum_by_customer_order = (customers_orders
                                     .groupby(["customer_id", "order_id"])
                                     .ordered_item_quantity.sum()
                                     .reset_index())

    returned_sum_by_customer_order = (customers_orders
                                      .groupby(["customer_id", "order_id"])
                                      .returned_item_quantity.sum()
                                      .reset_index())

    # merge two dataframes to be able to calculate unit return rate
    ordered_returned_sums = pd.merge(ordered_sum_by_customer_order, returned_sum_by_customer_order)

    # calculate unit return rate per order and customer
    ordered_returned_sums["average_return_rate"] = (-1 *
                                                    ordered_returned_sums["returned_item_quantity"] /
                                                    ordered_returned_sums["ordered_item_quantity"])

    # ordered_returned_sums.head()

    # take average of the unit return rate for all orders of a customer
    customer_return_rate = (ordered_returned_sums
                            .groupby("customer_id")
                            .average_return_rate
                            .mean()
                            .reset_index())

    return_rates = pd.DataFrame(customer_return_rate["average_return_rate"]
                                .value_counts()
                                .reset_index())

    return_rates.rename(columns=
                        {"index": "average return rate",
                         "average_return_rate": "count of unit return rate"},
                        inplace=True)

    return_rates.sort_values(by="average return rate")

    # add average_return_rate to customers dataframe
    customers = pd.merge(customers,
                         customer_return_rate,
                         on="customer_id")

    # ## 3. Total spending
    # Total spending is the aggregated sum of total sales value which is the amount after the taxes and returns.

    # aggreagate total sales per customer id
    customer_total_spending = (customers_orders
                               .groupby("customer_id")
                               .total_sales
                               .sum()
                               .reset_index())

    customer_total_spending.rename(columns={"total_sales": "total_spending"},
                                   inplace=True)

    # ## Create features data frame

    # add total sales to customers dataframe
    customers = customers.merge(customer_total_spending,
                                on="customer_id")

    print("The number of customers from the existing customer base:", customers.shape[0])

    # drop id column since it is not a feature
    customers.drop(columns="customer_id",
                   inplace=True)

    # customers.head()

    # ### Visualize features

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=("Products Ordered",
                                        "Average Return Rate",
                                        "Total Spending"))

    fig.append_trace(go.Histogram(x=customers.products_ordered),
                     row=1, col=1)

    fig.append_trace(go.Histogram(x=customers.average_return_rate),
                     row=2, col=1)

    fig.append_trace(go.Histogram(x=customers.total_spending),
                     row=3, col=1)

    fig.update_layout(height=800, width=800,
                      title_text="Distribution of the Features")

    # fig.show()

    # ## Scale Features: Log Transformation

    def apply_log1p_transformation(dataframe, column):
        """This function takes a dataframe and a column in the string format
        then applies numpy log1p transformation to the column
        as a result returns log1p applied pandas series"""

        dataframe["log_" + column] = np.log1p(dataframe[column])
        return dataframe["log_" + column]

    # ### 1. Products ordered

    apply_log1p_transformation(customers, "products_ordered")

    # ### 2. Average return rate

    apply_log1p_transformation(customers, "average_return_rate")

    # ### 3. Total spending

    apply_log1p_transformation(customers, "total_spending")

    # ### Visualize log transformation applied features

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=("Products Ordered",
                                        "Average Return Rate",
                                        "Total Spending"))

    fig.append_trace(go.Histogram(x=customers.log_products_ordered),
                     row=1, col=1)

    fig.append_trace(go.Histogram(x=customers.log_average_return_rate),
                     row=2, col=1)

    fig.append_trace(go.Histogram(x=customers.log_total_spending),
                     row=3, col=1)

    fig.update_layout(height=800, width=800,
                      title_text="Distribution of the Features after Logarithm Transformation")

    # fig.show()

    # customers.head()

    # features we are going to use as an input to the model
    customers.iloc[:, 3:]

    # ## Create K-means model

    # create initial K-means model
    kmeans_model = KMeans(init='k-means++',
                          max_iter=500,
                          random_state=42)

    kmeans_model.fit(customers.iloc[:, 3:])

    # print the sum of distances from all examples to the center of the cluster
    print("within-cluster sum-of-squares (inertia) of the model is:", kmeans_model.inertia_)

    # ## Hyperparameter tuning: Find optimal number of clusters

    def make_list_of_K(K, dataframe):
        '''inputs: K as integer and dataframe
        apply k-means clustering to dataframe
        and make a list of inertia values against 1 to K (inclusive)
        return the inertia values list
        '''

        cluster_values = list(range(1, K + 1))
        inertia_values = []

        for c in cluster_values:
            model = KMeans(
                n_clusters=c,
                init='k-means++',
                max_iter=500,
                random_state=42)
            model.fit(dataframe)
            inertia_values.append(model.inertia_)

        return inertia_values

    # ### Visualize different K and models

    # save inertia values in a dataframe for k values between 1 to 15
    results = make_list_of_K(15, customers.iloc[:, 3:])

    k_values_distances = pd.DataFrame({"clusters": list(range(1, 16)),
                                       "within cluster sum of squared distances": results})

    # visualization for the selection of number of segments
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=k_values_distances["clusters"],
                             y=k_values_distances["within cluster sum of squared distances"],
                             mode='lines+markers'))

    fig.update_layout(xaxis=dict(
        tickmode='linear',
        tick0=1,
        dtick=1),
        title_text="Within Cluster Sum of Squared Distances VS K Values",
        xaxis_title="K values",
        yaxis_title="Cluster sum of squared distances")

    # fig.show()

    # ## Update K-Means Clustering

    # create clustering model with optimal k=4
    updated_kmeans_model = KMeans(n_clusters=n_groups)
    updated_kmeans_model.fit_predict(customers.iloc[:, 3:])

    # ### Add cluster centers to the visualization

    # create cluster centers and actual data arrays
    cluster_centers = updated_kmeans_model.cluster_centers_
    actual_data = np.expm1(cluster_centers)
    add_points = np.append(actual_data, cluster_centers, axis=1)
    # add_points

    # add labels to customers dataframe and add_points array
    add_points = np.append(add_points, [[0], [1], [2], [3]], axis=1)
    customers["clusters"] = updated_kmeans_model.labels_

    # create centers dataframe from add_points
    centers_df = pd.DataFrame(data=add_points, columns=["products_ordered",
                                                        "average_return_rate",
                                                        "total_spending",
                                                        "log_products_ordered",
                                                        "log_average_return_rate",
                                                        "log_total_spending",
                                                        "clusters"])
    # centers_df.head()

    # align cluster centers of centers_df and customers
    centers_df["clusters"] = centers_df["clusters"].astype("int")

    # centers_df.head()

    # customers.head()

    # differentiate between data points and cluster centers
    customers["is_center"] = 0
    centers_df["is_center"] = 1

    # add dataframes together
    customers = customers.append(centers_df, ignore_index=True)

    customers.tail()

    # ### Visualize Customer Segmentation

    # add clusters to the dataframe
    customers["cluster_name"] = customers["clusters"].astype(str)

    # visualize log_transformation customer segments with a 3D plot

    global fig_three
    fig_three = px.scatter_3d(customers,
                              x="log_products_ordered",
                              y="log_average_return_rate",
                              z="log_total_spending",
                              color='cluster_name',
                              hover_data=["products_ordered",
                                          "average_return_rate",
                                          "total_spending"],
                              category_orders={"cluster_name":
                                                   ["0", "1", "2", "3"]},
                              symbol="is_center"
                              )

    fig_three.update_layout(title='IRR', autosize=False, width=800, height=800, margin=dict(l=40, r=40, b=40, t=40))
    st.plotly_chart(fig_three)

    # ## Check for Cluster Magnitude

    # values for log_transformation
    cardinality_df = pd.DataFrame(
        customers.cluster_name.value_counts().reset_index())

    cardinality_df.rename(columns={"index": "Customer Groups",
                                   "cluster_name": "Customer Group Magnitude"},
                          inplace=True)

    # cardinality_df

    fig = px.bar(cardinality_df, x="Customer Groups",
                 y="Customer Group Magnitude",
                 color="Customer Groups",
                 category_orders={"Customer Groups": ["0", "1", "2", "3"]})

    fig.update_layout(xaxis=dict(
        tickmode='linear',
        tick0=1,
        dtick=1),
        yaxis=dict(
            tickmode='linear',
            tick0=1000,
            dtick=1000))

    # fig.show()


def main():
    st.title("Customer Segmenter")

    # clusters = st.number_input('Clusters', step=1.0)
    model(4)


if __name__ == '__main__':
    main()
