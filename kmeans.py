#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation with K-Means Clustering
# 
# Customer segmentation will be applied to an e-commerce customer database using K-means clustering from scikit-learn. It is an [extension](https://github.com/cereniyim/Data-Science-Projects/blob/master/ECommerce-Sales-Data-EDA/ECommerce_Sales_Data_Analysis.ipynb) of a case study solved couple of months ago. 
# 
# The provided customers database is visualized as part of a case study. This project is taking the case study one step further with the following motive:
# 
# **Can this customer database be grouped to develop customized relationships?** 
# 
# **To answer this question 3 features will be created and used:** <br>
# - products ordered
# - average return rate
# - total spending
# 
# **Dataset represents real customers & orders data between November 2018 - April 2019 and it is pseudonymized for confidentiality.**

# **Imports**

# In[2]:


# data wrangling
import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# for data preprocessing and clustering
from sklearn.cluster import KMeans

get_ipython().run_line_magic('matplotlib', 'inline')
# to include graphs inline within the frontends next to code

get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
# to enable retina (high resolution) plots

pd.options.mode.chained_assignment = None
# to bypass warnings in various dataframe assignments


# **Investigate data**

# In[3]:


# load data into a dataframe
customers_orders = pd.read_csv("Orders - Analysis Task.csv")

# In[4]:


# first rows of the dataset
customers_orders.head()

# In[5]:


# first glance of customers_orders data
customers_orders.info()

# In[6]:


# descriptive statistics of the non-object columns
customers_orders.describe()

# There were significant number of rows whose `ordered_item_quantity` is 0 and `net_quantity` is less than 0,
# which means they are not ordered/sold at all; but the fact that they have returns requires investigation.

# In[7]:


print("Number of rows that net quantity is negative:",
      customers_orders[customers_orders.net_quantity < 0].shape[0])

# **These rows will be excluded from the orders dataset for the project.**

# In[8]:


# exclude not sold/ordered SKUs from the dataset
customers_orders = customers_orders[
    customers_orders["ordered_item_quantity"] > 0]


# ## 1. Products ordered
# It is the count of the products ordered in product_type column by a customer. <br>

# **Create functions to identify customers who order multiple products**

# In[9]:


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


# In[10]:


# apply functions to customers_orders
customers = aggregate_by_ordered_quantity(customers_orders, ["customer_id", "product_type"])

# In[11]:


print(customers.head())

# ## 2. Average Return Rate
# It is the ratio of returned item quantity and ordered item quantity. This ratio is first calculated per order and then averaged for all orders of a customer.

# In[12]:


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

# In[13]:


# calculate unit return rate per order and customer
ordered_returned_sums["average_return_rate"] = (-1 *
                                                ordered_returned_sums["returned_item_quantity"] /
                                                ordered_returned_sums["ordered_item_quantity"])

# In[14]:


ordered_returned_sums.head()

# In[15]:


# take average of the unit return rate for all orders of a customer
customer_return_rate = (ordered_returned_sums
                        .groupby("customer_id")
                        .average_return_rate
                        .mean()
                        .reset_index())

# In[16]:


return_rates = pd.DataFrame(customer_return_rate["average_return_rate"]
                            .value_counts()
                            .reset_index())

return_rates.rename(columns=
                    {"index": "average return rate",
                     "average_return_rate": "count of unit return rate"},
                    inplace=True)

return_rates.sort_values(by="average return rate")

# In[17]:


# add average_return_rate to customers dataframe
customers = pd.merge(customers,
                     customer_return_rate,
                     on="customer_id")

# ## 3. Total spending
# Total spending is the aggregated sum of total sales value which is the amount after the taxes and returns.

# In[18]:


# aggreagate total sales per customer id
customer_total_spending = (customers_orders
                           .groupby("customer_id")
                           .total_sales
                           .sum()
                           .reset_index())

customer_total_spending.rename(columns={"total_sales": "total_spending"},
                               inplace=True)

# ## Create features data frame

# In[19]:


# add total sales to customers dataframe
customers = customers.merge(customer_total_spending,
                            on="customer_id")

# In[20]:


print("The number of customers from the existing customer base:", customers.shape[0])

# In[21]:


# drop id column since it is not a feature
customers.drop(columns="customer_id",
               inplace=True)

# In[22]:


customers.head()

# ### Visualize features

# In[23]:


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

fig.show()


# ## Scale Features: Log Transformation

# In[24]:


def apply_log1p_transformation(dataframe, column):
    '''This function takes a dataframe and a column in the string format
    then applies numpy log1p transformation to the column
    as a result returns log1p applied pandas series'''

    dataframe["log_" + column] = np.log1p(dataframe[column])
    return dataframe["log_" + column]


# ### 1. Products ordered

# In[25]:


apply_log1p_transformation(customers, "products_ordered")

# ### 2. Average return rate

# In[26]:


apply_log1p_transformation(customers, "average_return_rate")

# ### 3. Total spending

# In[27]:


apply_log1p_transformation(customers, "total_spending")

# ### Visualize log transformation applied features

# In[28]:


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

fig.show()

# In[29]:


customers.head()

# In[30]:


# features we are going to use as an input to the model
customers.iloc[:, 3:]

# ## Create K-means model

# In[31]:


# create initial K-means model
kmeans_model = KMeans(init='k-means++',
                      max_iter=500,
                      random_state=42)

# In[32]:


kmeans_model.fit(customers.iloc[:, 3:])

# print the sum of distances from all examples to the center of the cluster
print("within-cluster sum-of-squares (inertia) of the model is:", kmeans_model.inertia_)


# ## Hyperparameter tuning: Find optimal number of clusters

# In[33]:


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

# In[34]:


# save inertia values in a dataframe for k values between 1 to 15 
results = make_list_of_K(15, customers.iloc[:, 3:])

k_values_distances = pd.DataFrame({"clusters": list(range(1, 16)),
                                   "within cluster sum of squared distances": results})

# In[35]:


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

fig.show()

# ## Update K-Means Clustering

# In[51]:


# create clustering model with optimal k=4
updated_kmeans_model = KMeans(n_clusters=4,
                              )

updated_kmeans_model.fit_predict(customers.iloc[:, 3:])

# ### Add cluster centers to the visualization

# In[37]:


# create cluster centers and actual data arrays
cluster_centers = updated_kmeans_model.cluster_centers_
actual_data = np.expm1(cluster_centers)
add_points = np.append(actual_data, cluster_centers, axis=1)
add_points

# In[38]:


# add labels to customers dataframe and add_points array
add_points = np.append(add_points, [[0], [1], [2], [3]], axis=1)
customers["clusters"] = updated_kmeans_model.labels_

# In[39]:


# create centers dataframe from add_points
centers_df = pd.DataFrame(data=add_points, columns=["products_ordered",
                                                    "average_return_rate",
                                                    "total_spending",
                                                    "log_products_ordered",
                                                    "log_average_return_rate",
                                                    "log_total_spending",
                                                    "clusters"])
centers_df.head()

# In[40]:


# align cluster centers of centers_df and customers
centers_df["clusters"] = centers_df["clusters"].astype("int")

# In[41]:


centers_df.head()

# In[42]:


customers.head()

# In[43]:


# differentiate between data points and cluster centers
customers["is_center"] = 0
centers_df["is_center"] = 1

# add dataframes together
customers = customers.append(centers_df, ignore_index=True)

# In[44]:


customers.tail()

# ### Visualize Customer Segmentation

# In[45]:


# add clusters to the dataframe
customers["cluster_name"] = customers["clusters"].astype(str)

# In[46]:


# visualize log_transformation customer segments with a 3D plot
fig = px.scatter_3d(customers,
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

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()

# ## Check for Cluster Magnitude

# In[47]:


# values for log_transformation
cardinality_df = pd.DataFrame(
    customers.cluster_name.value_counts().reset_index())

cardinality_df.rename(columns={"index": "Customer Groups",
                               "cluster_name": "Customer Group Magnitude"},
                      inplace=True)

# In[48]:


cardinality_df

# In[49]:


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

fig.show()

# In[50]:


import pickle

pickle_out = open("segmenter.pkl", "wb")
pickle.dump(updated_kmeans_model, pickle_out)
pickle_out.close()

# In[ ]:
