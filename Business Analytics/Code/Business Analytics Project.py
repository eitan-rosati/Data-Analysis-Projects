"""Project - Business Analytics
    We analyze data of the analytical department at Yandex.Afisha. Our task is to help optimize marketing expenses.
    For do that we have the following data:
    * Server logs with data on Yandex.Afisha visits from June 2017 through May 2018
    * Dump file with all orders for the period
    * Marketing expenses statistics

    We are going to study:
    * How people use the product
    * When they start to buy
    * How much money each customer brings
    * When they pay off."""

# **Imports packages:**
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
from operator import attrgetter
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# ## 1. Read datasets and prepare the data for the analysis
# Functions area:
def check_dataset(df_name, df):
    # Check dataset info
    print(df_name, 'dataset info:')
    df.info()

    # Display dataset first lines
    print('\n' + 'Display the head of the', df_name, 'dataset:')
    print(df.head(10))

    # Check missing values in the dataset
    check_missing_values(df)

    # Check duplicates in the dataset    
    check_duplicates(df_name, df)


def check_missing_values(df):
    print('\n' + 'Check missing values:\n')
    report = df.isna().sum().to_frame()
    report = report.rename(columns={0: 'missing_values'})
    report['% of total'] = (report['missing_values'] / df.shape[0]).round(4) * 100
    report.sort_values(by='missing_values', ascending=False, inplace=True)
    print(report)


def check_duplicates(df_name, df):
    if df.duplicated().sum() != 0:
        print("\n" + "The", df_name, "dataframe have duplicated rows.\n")
    else:
        print("\n" + "The", df_name, "dataframe don't have duplicated rows.\n")


# **visits dataset examination**


# Read dataset
try:
    visits = pd.read_csv('/datasets/visits_log_us.csv')
except:
    visits = pd.read_csv('https://code.s3.yandex.net/datasets/visits_log_us.csv')

# check visits dataset general information     
check_dataset("visits", visits)

# **orders dataset examination**


# Read dataset
try:
    orders = pd.read_csv('/datasets/orders_log_us.csv')
except:
    orders = pd.read_csv('https://code.s3.yandex.net/datasets/orders_log_us.csv')

# check orders dataset general information     
check_dataset("orders", orders)

# **costs dataset examination**


try:
    costs = pd.read_csv('/datasets/costs_us.csv')
except:
    costs = pd.read_csv('https://code.s3.yandex.net/datasets/costs_us.csv')

# check costs dataset general information     
check_dataset("costs", costs)

# **Covert and downcast datasets columns data types**

# 1. cast columns name to lowercase
visits = visits.rename(columns=str.lower)
orders = orders.rename(columns=str.lower)
costs = costs.rename(columns=str.lower)

# 2. Convert columns that contain date objects to datetime:
visits['end ts'] = pd.to_datetime(visits['end ts'], format='%Y-%m-%d %H:%M:%S')
visits['start ts'] = pd.to_datetime(visits['start ts'], format='%Y-%m-%d %H:%M:%S')
orders['buy ts'] = pd.to_datetime(orders['buy ts'], format='%Y-%m-%d %H:%M:%S')
costs['dt'] = pd.to_datetime(costs['dt'], format='%Y-%m-%d')

# 3. Add additional information in the datasets to group by the dataset by specific attributes:
# visits dataset:
visits['start_date'] = visits['start ts'].dt.date
visits['time_spent'] = (visits['end ts'] - visits['start ts']).dt.seconds
visits['month'] = visits['start ts'].dt.to_period('M')
visits['year'] = visits['start ts'].dt.year
visits['week'] = visits['start ts'].astype('datetime64[W]')
# orders dataset:
orders['month'] = orders['buy ts'].dt.to_period('M')
orders['year'] = orders['buy ts'].dt.year
# costs dataset:
costs['month'] = costs['dt'].dt.to_period('M')
costs['year'] = costs['dt'].dt.month

# 4. Downcast numeric columns(to save bits)
# visits dataset:
visits['source id'] = pd.to_numeric((visits['source id']), downcast='integer')
visits['time_spent'] = pd.to_numeric((visits['time_spent']), downcast='integer')
visits['year'] = pd.to_numeric((visits['year']), downcast='integer')
# orders dataset:
orders['revenue'] = pd.to_numeric((orders['revenue']), downcast='float')
orders['year'] = pd.to_numeric((orders['year']), downcast='integer')
# costs dataset:
costs['source_id'] = pd.to_numeric((visits['source id']), downcast='integer')
costs['costs'] = pd.to_numeric((costs['costs']), downcast='float')
costs['year'] = pd.to_numeric((costs['year']), downcast='integer')

# check dataframes info's after casting:
visits.info()
print()
orders.info()
print()
costs.info()

# 1. check the daily average of users entry
daily_traffic = visits.groupby(by='start_date')['uid'].count().reset_index()
print('The daily traffic average is:', int(daily_traffic.mean()))
print(daily_traffic)
fig = px.line(daily_traffic, x="start_date", y="uid", title='user number per day',
              labels={"start_date": "Date", "uid": "user number"}, color_discrete_sequence=['red'])
fig.add_hline(y=daily_traffic['uid'].median(),
              annotation_text="Median",
              annotation_position="top left")
fig.show()

# 2. check the monthly traffic of users entry
monthly_traffic = visits.groupby(by='month').count()[['uid']]
print('The monthly traffic per month is:', int(monthly_traffic.mean()))
# plot bar plot of the monthly traffic
monthly_traffic.plot.bar(rot=0, figsize=(15, 8), title='Monthly Traffic', color='blue')
plt.ylabel("amount of visits")
plt.xlabel("Month")
plt.show()

# 2. check the traffic per year of users entry
year_traffic = visits.groupby(by='year').count()[['uid']]
# plot bar plot of the monthly traffic
year_traffic.plot.bar(rot=0, figsize=(15, 8), title='Traffic per Year', color=['purple'])
plt.ylabel("amount of visits")
plt.xlabel("year")
plt.show()

# **Check How many people use every day, week, and month**

dau = visits.groupby(by='start_date')['uid'].nunique().reset_index()
print(dau)
fig = px.line(dau, x="start_date", y="uid", title='DAU', labels={"start_date": "Date", "uid": "Unique user number"})
fig.add_hline(y=dau['uid'].median(),
              annotation_text="Median",
              annotation_position="top left")
fig.show()

wau = orders.groupby(visits['week'])['uid'].nunique().reset_index()

# plot
fig = go.Figure(layout=go.Layout(
    title=go.layout.Title(text="WAU")))
fig.add_trace(go.Scatter(x=wau['week'],
                         y=wau['uid'],
                         mode='lines+markers',
                         name='lines+markers', line=dict(color='orange', width=3)))
# adding reference line with average DAU over time
fig.show()

monthly_unique_users = visits.groupby(by='month')[['uid']].nunique()
print(monthly_unique_users)

monthly_unique_users.plot.bar(rot=0, figsize=(15, 8), title='Unique users per month', color='blue')
plt.ylabel("amount of unique visits")
plt.xlabel("Month")
plt.show()

year_unique_users = visits.groupby(by='year')[['uid']].nunique()
print(year_unique_users)

year_unique_users.plot.bar(rot=0, figsize=(15, 8), title='Unique users per year', color='orange')
plt.ylabel("amount of unique visits")
plt.xlabel("Year")
plt.show()

# **Check length of each session**


session_duration = visits[['time_spent']]
session_duration['time_spent'].describe()

ax1 = session_duration['time_spent'].hist(figsize=(15, 8), color='cyan', range=(0, 3600), alpha=0.9)
ax1.set_title('Session duration histogram')
ax1.set_xlabel('session duration(in seconds)')
ax1.set_ylabel('Amount of sessions')
ax1.plot()

# **How often do users come back**


# 1. find the first session of every user
min_visit = visits.groupby('uid')['start_date'].min().reset_index()
min_visit.columns = ['uid', 'first_session']
print(min_visit)

# 2. merge the visits dataset with the the dataset we created before to find the first session
visits = visits.merge(min_visit, how='inner', on='uid')
print(visits.head(10))

# 3. find cohort and age and then pivot
visits['cohort'] = pd.to_datetime(visits['first_session']).dt.to_period('M')
visits['age'] = ((pd.to_datetime(visits['start_date']) - pd.to_datetime(visits['first_session'])) /
                 np.timedelta64(1, 'M')).round().astype('int')
cohorts = visits.pivot_table(index='cohort',
                             columns='age',
                             values='uid',
                             aggfunc='nunique').fillna(0)

retention = cohorts.iloc[:, 0:].div(cohorts[0], axis=0)

# plot heatmap
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(retention, annot=True, fmt='.1%', linewidths=1, linecolor='grey', vmax=0.1,
            cbar_kws={'orientation': 'horizontal'}
            ).set(title='Retention Rate')
plt.show()

# #### Check the influence of the different devices used by the user


# display pie chart for the amount of session
devices_sessions = visits.groupby(by='device')[['uid']].count()
devices_sessions.columns = ['session amount']
ax = devices_sessions.plot.pie(y='session amount', figsize=(10, 10), autopct='%1.1f%%')
ax.set_title('Percentage of the total number of sessions according to each platform')
plt.show()

# check unique entry for mobile devices:
touch_devices = visits[visits['device'] == 'touch']
dau_touch = touch_devices.groupby(by='start_date')['uid'].nunique().reset_index()
fig = px.line(dau_touch, x="start_date", y="uid", title='DAU - Touch devices',
              labels={"start_date": "Date", "uid": "Unique user number"})
fig.add_hline(y=dau_touch['uid'].median(),
              annotation_text="Median",
              annotation_position="top left")
fig.show()

# check unique entry for desktop devices:
desktop_devices = visits[visits['device'] == 'desktop']
dau_desktop = desktop_devices.groupby(by='start_date')['uid'].nunique().reset_index()
fig = px.line(dau_desktop, x="start_date", y="uid", title='DAU - Desktop devices',
              labels={"start_date": "Date", "uid": "Unique user number"})
fig.add_hline(y=dau_touch['uid'].median(),
              annotation_text="Median",
              annotation_position="bottom right")
fig.show()

# Check session duration according to the kind of device

# check the distribution of session length on touch devices
touch_session_duration = touch_devices[['time_spent']]
touch_session_duration['time_spent'].describe()

ax3 = touch_session_duration['time_spent'].hist(figsize=(15, 8), color='gray', range=(0, 3600), alpha=1)
ax3.set_title('Session duration histogram for touch devices')
ax3.set_xlabel('session duration(in seconds)')
ax3.set_ylabel('Amount of sessions')
plt.show()

# check the distribution of session length on touch devices
desktop_session_duration = desktop_devices[['time_spent']]
desktop_session_duration['time_spent'].describe()

ax4 = desktop_session_duration['time_spent'].hist(figsize=(15, 8), color='green', range=(0, 3600), alpha=0.9)
ax4.set_title('Session duration histogram for Desktop devices')
ax4.set_xlabel('session duration(in seconds)')
ax4.set_ylabel('Amount of sessions')
plt.show()

# ### B. Sales

# Revenue per month
# group by month
month_revenue = orders.groupby(['month'])['revenue'].sum().reset_index()

# plot
month_revenue.set_index('month', drop=True, inplace=True)
month_revenue.plot.bar(rot=0, figsize=(15, 8), title='Total revenue per month', color='brown')

plt.show()

# 1. find out the time of first order for each user.
first_order = orders.groupby('uid')['buy ts'].min().reset_index()
first_order.columns = ['uid', 'first_order_date']
print(first_order.head(5))

# 2. merge first_order and min_visit to the orders table
orders = orders.merge(first_order, how='left', on='uid')
orders = orders.merge(min_visit, how='left', on='uid')
orders['conversion'] = (
        (pd.to_datetime(orders['first_order_date']) - pd.to_datetime(orders['first_session'])) /
        np.timedelta64(1, 'D')).astype('int')

print(orders)

# 3. print() the histogram of the conversation
fig = px.histogram(orders, x="conversion", nbins=15, color_discrete_sequence=['green'],
                   labels={'conversion': 'Conversation rate'}, title="Conversation Histogram")
fig.show()

# **Check how many orders the users will make during a given period of time**


# 1. Add the month of the first order as a column
orders['first_order_month'] = orders['first_order_date'].dt.to_period('M')
# 2. check when the first purchase of any user and save it in different df.
cohort_by_month = orders.groupby('first_order_month').agg({'uid': 'nunique'}).reset_index()
cohort_by_month.columns = ['first_order_month', 'cohort_size']
cohort_by_month.head()

# calculating number of purchases for cohort and month
cohort = orders.groupby(['first_order_month', 'month'])['revenue'].count().reset_index()
cohort.columns = ['first_order_month', 'month', 'orders']
# merge cohort with cohort size
cohort = cohort.merge(cohort_by_month, on=['first_order_month'])
cohort['age_month'] = (cohort['month'] - cohort['first_order_month']).apply(attrgetter('n'))
cohort['orders_per_buyer'] = cohort['orders'] / cohort['cohort_size']
print(cohort)

cohort_piv = cohort.pivot_table(
    index='first_order_month',
    columns='age_month',
    values='orders_per_buyer',
    aggfunc='sum'
).cumsum(axis=1)

cohort_piv = cohort_piv.fillna(0)

# print() a heat map:
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cohort_piv, annot=True, linewidths=1, linecolor='gray', cbar_kws={'orientation': 'horizontal'}
            ).set(title='mean orders number for user during a given period of time')
plt.show()

# **Check what is the average purchase size**


revenue_per_user = orders.groupby(['uid'])['revenue'].sum().reset_index()
# check the distribution before plotting the histogram
revenue_per_user['revenue'].describe()

# lets check the percentage of the users that the company had between 0 to 100 dollars of revenue.
percentage = (len(revenue_per_user[revenue_per_user['revenue'] <= 100]) / len(revenue_per_user))
print('percentage of the users their total orders cost was between 0 to 100 dollars is: {:.2%}'.format(percentage))

fig = px.histogram(revenue_per_user, x="revenue", color_discrete_sequence=['purple'], range_x=[0, 100],
                   title="revenue per user")
fig.show()

revenue_per_cohort = orders.groupby(['first_order_month', 'month'])['revenue'].mean().reset_index()
revenue_per_cohort['age_month'] = (revenue_per_cohort['month'] - revenue_per_cohort['first_order_month']).apply(
    attrgetter('n'))
print(revenue_per_cohort.head(12))

# pivot to find the mean and plot
avg_cohort_piv = revenue_per_cohort.pivot_table(index='first_order_month',
                                                columns='age_month',
                                                values='revenue',
                                                aggfunc='mean'
                                                )
# plot
fig, ax = plt.subplots(figsize=(10, 10))
avg_cohort_piv = avg_cohort_piv.round(1).fillna(0)
sns.heatmap(avg_cohort_piv, vmax=20, annot=True, linewidths=1, linecolor='gray', cbar_kws={'orientation': 'horizontal'}
            ).set(title='average purchase size per cohort')
plt.show()

# **Calculate LTV - how much money the company will earn from the revenues**


# Calculate the revenue per cohort in each month
ltv_cohort = orders.groupby(['first_order_month', 'month'])['revenue'].sum().reset_index()
print(ltv_cohort)
ltv_cohort.columns = ['first_order_month', 'end_month', 'cohort_revenue']
# Merge and calculate ltv 
ltv_cohort = ltv_cohort.merge(cohort_by_month, on=['first_order_month'])
ltv_cohort['age'] = (ltv_cohort['end_month'] - ltv_cohort['first_order_month']).apply(attrgetter('n'))
ltv_cohort['ltv'] = ltv_cohort['cohort_revenue'] / ltv_cohort['cohort_size']
print(ltv_cohort.head(15))

# pivot and print() hit map
ltv_cohort_piv = ltv_cohort.pivot_table(
    index='first_order_month',
    columns='age',
    values='ltv',
    aggfunc='sum'
).cumsum(axis=1)
ltv_cohort_piv = ltv_cohort_piv.fillna(0)

# plot
fig, ax = plt.subplots(figsize=(10, 10))
ltv_cohort_piv.index = ltv_cohort_piv.index.astype(str)
sns.heatmap(ltv_cohort_piv, annot=True, fmt='.2f', linewidths=1, linecolor='grey',
            cbar_kws={'orientation': 'horizontal'}
            ).set(title='LTV')
plt.show()

# ### C. Marketing 

# **Check how much money was spent (Overall/per source/over time)**

# Total marketing costs

# Calculate the total marketing costs
costs = costs.sort_values(by=['dt', 'source_id'])
print('The Total marketing cost is ', costs['costs'].sum())

# cost per month compare revenue

# group by month
month_cost = costs.groupby(['month'])['costs'].sum().reset_index()

# plot
month_cost.set_index('month', drop=True, inplace=True)
month_cost['revenue'] = month_revenue['revenue']
month_cost.plot.bar(rot=0, figsize=(15, 8), title='Total cost per month')
plt.show()

# Total cost per source id

# group by source_id and calculate per each source id all his costs
costs_per_source = costs.groupby(['source_id'])['costs'].sum().reset_index()
print(costs_per_source)

# plot
costs_per_source.set_index('source_id', drop=True, inplace=True)
costs_per_source.plot.bar(rot=0, figsize=(15, 8), title='Total cost per source id', color='blue')
plt.show()

# costs over time

fig = px.line(costs, x="dt", y="costs", color='source_id')
fig.show()

# **How much did customer acquisition from each of the sources cost**

# group by month for find the payment for each month
costs['pay_month'] = costs['dt'].astype('datetime64[M]')
cost_per_month = costs.groupby(['pay_month'])['costs'].sum().reset_index()

# check the first orders of those months
orders['first_order_month'] = orders['first_order_month'].astype(str)
orders['first_order_month'] = pd.to_datetime(orders['first_order_month']).astype('datetime64[M]')
buyers_per_first_time = orders.groupby(['first_order_month'])['uid'].nunique().reset_index()
buyers_per_first_time.columns = ['pay_month', 'orders']
print(buyers_per_first_time.head())

# Merge and show CAC per month
CAC_per_month = cost_per_month.merge(buyers_per_first_time, how='left', on=['pay_month'])
CAC_per_month['CAC'] = CAC_per_month['costs'] / CAC_per_month['orders']
fig = px.line(CAC_per_month, x="pay_month", y="CAC", title='CAC per month', color_discrete_sequence=['purple'])
fig.show()

# **CAC per source id**

# create a table for the first user entry source
first_sources = visits.sort_values('start ts').groupby('uid').first()['source id'].reset_index()
first_sources.columns = ['uid', 'first_source']

# merge it with the orders table
orders = orders.merge(first_sources, on=['uid'], how='left')
print(orders.head())

# group costs sum by month and source id
costs_by_month_source = costs.groupby(['pay_month', 'source_id'])['costs'].sum().reset_index()

# merge
buyers_per_moth_source = orders.groupby(['first_order_month', 'first_source'])['uid'].nunique().reset_index()
buyers_per_moth_source.columns = ['pay_month', 'source_id', 'buyers']

# cac calculations
CAC_per_month_source = costs_by_month_source.merge(buyers_per_moth_source, how='left', on=['pay_month', 'source_id'])
CAC_per_month_source['CAC'] = CAC_per_month_source['costs'] / CAC_per_month_source['buyers']
print(CAC_per_month_source.head())

# plotting
fig = px.line(CAC_per_month_source, x="pay_month", y="CAC", color='source_id', title='CAC per source id')
fig.show()

# **Check how worthwhile where the investments (ROI)**


# roi = ltv/cac lets take this 2 information and calculate the ROI
CAC_per_month_ROI = CAC_per_month[['pay_month', 'CAC']]
CAC_per_month_ROI.columns = ['first_order_month', 'CAC']

# convert to datetime 
ltv_cohort['first_order_month'] = ltv_cohort['first_order_month'].astype(str)
ltv_cohort['first_order_month'] = pd.to_datetime(ltv_cohort['first_order_month']).astype('datetime64[M]')

# merge
ROI = ltv_cohort.merge(CAC_per_month_ROI, on=['first_order_month'], how='left')
ROI.head()

# Calculate the ROI and pivot the table
ROI['ROI'] = ROI['ltv'] / ROI['CAC']
ROI['first_order_month'] = ROI['first_order_month'].dt.to_period('M')
roi_piv = ROI.pivot_table(
    index='first_order_month', columns='age', values='ROI', aggfunc='mean'
).cumsum(axis=1).round(2)
roi_piv = roi_piv.fillna(0)

# plot the heat-map

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(roi_piv, annot=True, fmt='.1%', linewidths=1, linecolor='grey', cbar_kws={'orientation': 'horizontal'}
            ).set(title='ROI per cohort')
plt.show()

ltv_per_source = orders.groupby(['first_source'])['uid', 'revenue'].agg(
    {'uid': 'nunique', 'revenue': 'sum'}).reset_index()
ltv_per_source.columns = ['source_id', 'buyers', 'revenue']
ltv_per_source['ltv'] = ltv_per_source['revenue'] / ltv_per_source['buyers']
print(ltv_per_source)

roi_per_source = costs_per_source.merge(ltv_per_source, on=['source_id'])
roi_per_source['cac'] = roi_per_source['costs'] / roi_per_source['buyers']
roi_per_source['romi'] = roi_per_source['ltv'] / roi_per_source['cac']
fig = px.bar(roi_per_source, x='source_id', y='romi', title="romi per source id", color='source_id')
fig.update_xaxes(type='category')
fig.show()
