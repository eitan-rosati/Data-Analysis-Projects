import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from scipy import stats as st
import math


# 1. Check the general Information of the dataset
# functions area:
def check_dataset(df):
    # print dataset info:
    print('1. Info:')
    df.info()

    # print the table
    print('\n' + '2. print table:')
    print(df.head())

    # Check the missing data propotion:
    print('\n' + '3. Check missing values:\n')
    report = df.isna().sum().to_frame()
    report = report.rename(columns={0: 'missing_values'})
    report['% of total'] = (report['missing_values'] / df.shape[0]).round(2)
    report.sort_values(by='missing_values', ascending=False)
    print(report, '\n')

    # Check duplicates
    print('\n' + '4. Check duplicates:')
    if df.duplicated().sum() != 0:
        print("The table have duplicated \n")
    else:
        print('No duplicated detected \n')


# Open the datasets and exam each one of them:
try:
    megaline_calls = pd.read_csv('megaline_calls.csv')
except:
    megaline_calls = pd.read_csv('https://code.s3.yandex.net/datasets/megaline_calls.csv')

try:
    megaline_internet = pd.read_csv('megaline_internet.csv')
except:
    megaline_internet = pd.read_csv('https://code.s3.yandex.net/datasets/megaline_internet.csv')

try:
    megaline_messages = pd.read_csv('messages.csv')
except:
    megaline_messages = pd.read_csv('https://code.s3.yandex.net/datasets/megaline_messages.csv')

try:
    megaline_plans = pd.read_csv('megaline_plans.csv')
except:
    megaline_plans = pd.read_csv('https://code.s3.yandex.net/datasets/megaline_plans.csv')

try:
    megaline_users = pd.read_csv('megaline_users.csv')
except:
    megaline_users = pd.read_csv('https://code.s3.yandex.net/datasets/megaline_users.csv')

# tables examination:
print("_________megaline_calls table examination:_________\n")
check_dataset(megaline_calls)

print("_________megaline_internet table examination:_________\n")
check_dataset(megaline_internet)

print("_________megaline_messages table examination:_________\n")
check_dataset(megaline_messages)

print("_________megaline_plans table examination:_________\n")
check_dataset(megaline_plans)

print("_________megaline_users table examination:_________\n")
check_dataset(megaline_users)


# Part 2 - Data Processing
# Functions area: 
def convert_id_to_numeric(user_id):
    str_id = str(user_id)
    '''remove all non digit characters from id and return it as a integer'''
    numeric_id = int(re.sub("[^0-9]", "", str_id))
    return numeric_id


def calculate_calls_number(user_id):
    total_calls_num = len(megaline_calls[megaline_calls['user_id'] == user_id])
    return total_calls_num


def calculate_calls_total_duration(user_id):
    total_calls_duration = megaline_calls[megaline_calls['user_id'] == user_id]['duration'].sum()
    return math.ceil(total_calls_duration)


def calculate_message_number(user_id):
    total_message_number = len(megaline_messages[megaline_messages['user_id'] == user_id])
    return total_message_number


def calculate_mb_used_per_month(user_id):
    total_mb_used = megaline_internet[megaline_internet['user_id'] == user_id]['mb_used'].sum()
    # return the result in MB
    return math.ceil(total_mb_used / 1000)


def calculate_revenue(data):
    # data = 0: user_id, 1: first_name, 2:last_name, 3:age, 4:city, 5:reg_date, 6:plan, 7:churn_date, 
    #        8:monthly_calls_number, 9:monthly_calls_duration, 10:monthly_messages_number, 11:monthly_volume

    # Calculate the monthly revenue from each user (subtract the free package limit from the total number of calls,
    # text messages, and data; multiply the result by the calling plan value. the monthly charge depends on the calling
    # plan)

    # Take the package calls,nm and messages limit: 
    payment = megaline_plans.loc[megaline_plans['plan_name'] == data[6]]['usd_monthly_pay'].unique()[0]
    package_calls_duration = megaline_plans.loc[megaline_plans['plan_name'] == data[6]]['minutes_included'].unique()[0]
    package_messages_number = megaline_plans.loc[megaline_plans['plan_name'] == data[6]]['messages_included'].unique()[0]
    package_internet_limit = megaline_plans.loc[megaline_plans['plan_name'] == data[6]]['mb_per_month_included'].unique()[0]

    # Check the difference between the package limit and what user used
    extra_minutes = data[8] - package_calls_duration
    extra_messages = data[10] - package_messages_number
    extra_mb = data[11] - package_internet_limit

    if extra_minutes > 0:
        usd_per_minute = megaline_plans.loc[megaline_plans['plan_name'] == data[6]]['usd_per_minute'].unique()[0]
        payment += extra_minutes * usd_per_minute

    if extra_messages > 0:
        usd_per_message = megaline_plans.loc[megaline_plans['plan_name'] == data[6]]['usd_per_message'].unique()[0]
        payment += extra_messages * usd_per_message

    if extra_mb > 0:
        extra_gb = extra_mb / 1000
        usd_per_gb = megaline_plans.loc[megaline_plans['plan_name'] == data[6]]['usd_per_gb'].unique()[0]
        payment += extra_gb * usd_per_gb

    # round up if there's number after the floating point 
    return payment


# **1. Downcast the data and convert the dates to datetime**
# ___________________megaline_calls table:___________________

# a.Fix the id column values to a valid integers:
megaline_calls['id'] = megaline_calls['id'].apply(convert_id_to_numeric)

# b. convert the call_date column to datetime: 
megaline_calls['call_date'] = pd.to_datetime(megaline_calls['call_date'], format='%Y-%m-%d')

# c. Downcast user_id and duration columns:
megaline_calls['user_id'] = pd.to_numeric((megaline_calls['user_id']), downcast='integer')
megaline_calls['duration'] = pd.to_numeric((megaline_calls['duration']), downcast='float')

# d. Check changes:
print('Check changes on the table:\n')
megaline_calls.info()
print(megaline_calls.head())

# ___________________megaline_internet table:___________________

# a.Fix the id column values to a valid integers:
megaline_internet['id'] = megaline_internet['id'].apply(convert_id_to_numeric)

# b. convert the session_date column to datetime: 
megaline_internet['session_date'] = pd.to_datetime(megaline_internet['session_date'], format='%Y-%m-%d')

# c. Downcast user_id and duration columns:
megaline_internet['user_id'] = pd.to_numeric((megaline_internet['user_id']), downcast='integer')
megaline_internet['mb_used'] = pd.to_numeric((megaline_internet['mb_used']), downcast='float')

# d. Check changes:
print('Check changes on the table:\n')
megaline_internet.info()
print(megaline_internet.head())

# ___________________megaline_messages table:___________________

# a.Fix the id column values to a valid integers:
megaline_messages['id'] = megaline_messages['id'].apply(convert_id_to_numeric)

# b. convert the message_date column to datetime: 
megaline_messages['message_date'] = pd.to_datetime(megaline_messages['message_date'], format='%Y-%m-%d')

# c. Downcast 'user_id' and duration columns:
megaline_messages['user_id'] = pd.to_numeric((megaline_messages['user_id']), downcast='integer')

# d. Check changes:
print('Check changes on the table:\n')
megaline_messages.info()
print(megaline_messages.head())

# ___________________megaline_plans table:___________________

# a. Downcast user_id and duration columns:
megaline_plans['messages_included'] = pd.to_numeric((megaline_plans['messages_included']), downcast='integer')
megaline_plans['mb_per_month_included'] = pd.to_numeric((megaline_plans['mb_per_month_included']), downcast='integer')
megaline_plans['minutes_included'] = pd.to_numeric((megaline_plans['minutes_included']), downcast='integer')
megaline_plans['usd_monthly_pay'] = pd.to_numeric((megaline_plans['usd_monthly_pay']), downcast='integer')
megaline_plans['usd_per_gb'] = pd.to_numeric((megaline_plans['usd_per_gb']), downcast='integer')
megaline_plans['usd_per_message'] = pd.to_numeric((megaline_plans['usd_per_message']), downcast='float')
megaline_plans['usd_per_minute'] = pd.to_numeric((megaline_plans['usd_per_minute']), downcast='float')

# b. check changes:
print('Check changes on the table:\n')
megaline_plans.info()
print(megaline_plans.head())

# ___________________megaline_users table:___________________
# a. cast churn_date and reg_date to datetime type:
megaline_users['reg_date'] = pd.to_datetime(megaline_users['reg_date'], format='%Y-%m-%d')
megaline_users['churn_date'] = pd.to_datetime(megaline_users['churn_date'], format='%Y-%m-%d', errors='ignore')

# b. Downcast numeric columns:
megaline_users['user_id'] = pd.to_numeric((megaline_users['user_id']), downcast='integer')
megaline_users['age'] = pd.to_numeric((megaline_users['age']), downcast='integer')

# c. check changes:
print('Check changes on the table:\n')
megaline_users.info()
print(megaline_users.head())

# **2. Add new columns to the megaline_user dataframe**

# In[7]:


# a. monthly_calls_number - The number of calls made per month
megaline_users['monthly_calls_number'] = megaline_users['user_id'].apply(calculate_calls_number)

# b. monthly_calls_duration = The calls minutes used per month
megaline_users['monthly_calls_duration'] = megaline_users['user_id'].apply(calculate_calls_total_duration)

# c. monthly_messages_number = The number of text messages sent per month
megaline_users['monthly_messages_number'] = megaline_users['user_id'].apply(calculate_message_number)

# d. monthly_volume = The volume of data per month(In GB)
megaline_users['monthly_volume'] = megaline_users['user_id'].apply(calculate_mb_used_per_month)

# e. monthly_revenue = The monthly revenue from each user (subtract the free package limit from the total
#    number of calls, text messages,and data; multiply the result by the calling plan value; 
#    add the monthly charge depending on the calling plan)
megaline_users['monthly_revenue'] = megaline_users.apply(calculate_revenue, axis=1)

# Show the table withe new columns
print('\n' + 'megaline_users dataset after added the new columns:\n')
megaline_users.info()
print(megaline_users.head())


# Part 3 - Analyze the data

# _______________________Analyze the data_______________________

# a. Create two datasets(one for the in surf plan and the second for ultimate plan) for the next steps
surf_plan_users = megaline_users.loc[megaline_users['plan'] == 'surf']
ultimate_plan_users = megaline_users.loc[megaline_users['plan'] == 'ultimate']

# reset index for both datasets:
surf_plan_users.reset_index()
ultimate_plan_users.reset_index()

# b. Check mean, variance, and standard deviation for the minutes, texts, and volume of data the users of each plan

# _________calls:_________
# mean:
surf_calls_mean = surf_plan_users['monthly_calls_duration'].mean()
ultimate_calls_mean = ultimate_plan_users['monthly_calls_duration'].mean()

# variance:
surf_calls_varience = surf_plan_users['monthly_calls_duration'].var()
ultimate_calls_varience = ultimate_plan_users['monthly_calls_duration'].var()

# standard deviation:
surf_calls_sigma = np.sqrt(surf_calls_varience)
ultimate_calls_sigma = np.sqrt(ultimate_calls_varience)

print('______Calls mean,variance, and standard deviation______')
print('\n' + 'Users with surf plan:')
print('Mean:', surf_calls_mean, '\n' + 'Varience:', surf_calls_varience, '\n' + 'standard deviation:', surf_calls_sigma)
print('\n' + 'Users with ultimate plan:')
print('Mean:', ultimate_calls_mean, '\n' + 'Varience:', ultimate_calls_varience, '\n' + 'standard deviation:',
      ultimate_calls_sigma)

# print histogram:
ax1 = surf_plan_users['monthly_calls_duration'].hist(figsize=(12, 6), bins=8, grid=True, color='blue',
                                                     label='surf plan')
ax2 = ultimate_plan_users['monthly_calls_duration'].hist(figsize=(12, 6), bins=8, grid=True, color='orange',
                                                         label='ultimate plan')

ax1.set_title('users calls histogram')
ax1.set_xlabel('volume(GB)')
ax1.set_ylabel('Amount of users')
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
ax1.plot()
ax2.plot()


# _________messages:_________
# mean:
surf_messages_mean = surf_plan_users['monthly_messages_number'].mean()
ultimate_messages_mean = ultimate_plan_users['monthly_messages_number'].mean()

# variance:
surf_messages_varience = surf_plan_users['monthly_messages_number'].var()
ultimate_messages_varience = ultimate_plan_users['monthly_messages_number'].var()

# standard deviation:
surf_messages_sigma = np.sqrt(surf_messages_varience)
ultimate_messages_sigma = np.sqrt(ultimate_messages_varience)

print('______Messages mean,variance, and standard deviation______')
print('\n' + 'Users with surf plan:')
print('Mean:', surf_messages_mean, '\n' + 'Varience:', surf_messages_varience, '\n' + 'standard deviation:',
      surf_messages_sigma)
print('\n' + 'Users with ultimate plan:')
print('Mean:', ultimate_messages_mean, '\n' + 'Varience:', ultimate_messages_varience, '\n' + 'standard deviation:',
      ultimate_messages_sigma)

# print histogram:
ax3 = surf_plan_users['monthly_messages_number'].hist(figsize=(12, 6), bins=5, grid=True, color='green',
                                                      label='surf plan')
ax4 = ultimate_plan_users['monthly_messages_number'].hist(figsize=(12, 6), bins=5, grid=True, color='yellow',
                                                          label='ultimate plan')
ax3.set_xlabel('Message users')
ax3.set_ylabel('Amount of users')
ax3.set_title('users messages histogram')
ax3.legend(loc='upper right')
ax4.legend(loc='upper right')


# _________Data volume:_________
# mean:
surf_volume_mean = surf_plan_users['monthly_volume'].mean()
ultimate_volume_mean = ultimate_plan_users['monthly_volume'].mean()

# variance:
surf_volume_varience = surf_plan_users['monthly_volume'].var()
ultimate_volume_varience = ultimate_plan_users['monthly_volume'].var()

# standard deviation:
surf_volume_sigma = np.sqrt(surf_volume_varience)
ultimate_volume_sigma = np.sqrt(ultimate_volume_varience)

print('______Data volume mean,variance, and standard deviation______')
print('\n' + 'Users with surf plan:')
print('Mean:', surf_volume_mean, '\n' + 'Varience:', surf_volume_varience, '\n' + 'standard deviation:',
      surf_volume_sigma)
print('\n' + 'Users with ultimate plan:')
print('Mean:', ultimate_volume_mean, '\n' + 'Varience:', ultimate_volume_varience, '\n' + 'standard deviation:',
      ultimate_volume_sigma)

# print histogram:
ax3 = surf_plan_users['monthly_volume'].hist(figsize=(12, 6), bins=6, grid=True, color='purple', label='surf plan')
ax4 = ultimate_plan_users['monthly_volume'].hist(figsize=(12, 6), bins=6, grid=True, color='red', label='ultimate plan')

ax3.set_title('users data volume histogram')
ax3.legend(loc='upper right')
ax3.set_xlabel('volume(GB)')
ax3.set_ylabel('Amount of users')
ax4.legend(loc='upper right')


# ## Part 4 - Test the hypotheses 

# In this part we test 2 hypotses:
# 1. The average revenue from users of Ultimate and Surf calling plans differs.
# 2. The average revenue from users in NY-NJ area is different from that of the users from other regions.

# your code: set a critical statistical significance level for both tests:
alpha = 0.05

# ______________________First test:______________________

print('____________TEST1:____________\n')
print('Null hypothesis: The average revenue from users of Ultimate and Surf calling plans are equal.')
print('Alternative hypothesis: The average revenue from users of Ultimate and Surf calling plans differs.')

surf_revenue_mean = surf_plan_users['monthly_revenue'].mean()
ultimate_revenue_mean = ultimate_plan_users['monthly_revenue'].mean()
print('surf users revenue mean:', surf_revenue_mean)
print('ultimate users revenue mean:', ultimate_revenue_mean)

results1 = st.ttest_ind(surf_plan_users['monthly_revenue'], ultimate_plan_users['monthly_revenue'], equal_var=False)
print('p-value of the alternative hypothesis:', results1.pvalue)

if (results1.pvalue < alpha):
    print("We reject the null hypothesis\n")
else:
    print("We can't reject the null hypothesis\n")

# ______________________Second test:______________________

print('____________TEST2:____________\n')
print(
    'Null hypothesis: The average revenue from users in NY-NJ area is different from that of the users from other regions.')
print(
    'Alternative hypothesis: The average revenue from users in NY-NJ area is equal from that of the users from other regions.')

ny_nj_revenues = megaline_users[megaline_users['city'].str.contains('NY-NJ')]['monthly_revenue']
other_revenues = megaline_users[~megaline_users['city'].str.contains('NY-NJ')]['monthly_revenue']

ny_nj_revenues_mean = ny_nj_revenues.mean()
other_revenues_mean = other_revenues.mean()
print('Users revenue mean in NY-NJ', ny_nj_revenues_mean)
print('Users revenue revenue mean in other cities:', other_revenues_mean)

results2 = st.ttest_ind(ny_nj_revenues, other_revenues, equal_var=False)
print('p-value of the alternative hypothesis:', results2.pvalue)

if (results2.pvalue < alpha):
    print("We reject the null hypothesis\n")
else:
    print("We can't reject the null hypothesis\n")

