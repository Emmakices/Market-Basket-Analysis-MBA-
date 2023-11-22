#!/usr/bin/env python
# coding: utf-8

# # IMPORT DATASET

# In[1]:


import pandas as pd
# Import the Excel data file
csv_file_path = r"C:\Users\Administrator\Desktop\dataset\E commerce business transactions.csv"
data = pd.read_csv(csv_file_path)
# Display the data
data.head()


# In[2]:


data.shape


# In[3]:


data.info()


# # DESCRIPTIVE STATISTICS

# In[4]:


# Generating descriptive statistics
descriptive_stats = data.drop(columns='Customer ID').describe(include='all', datetime_is_numeric=True)
descriptive_stats


# # DATA CLEANSING AND PREPROCESSING

# In[5]:


# Checking for missing values
missing_values = data.isnull().sum()
# Data types of each column
data_types = data.dtypes
missing_values, data_types


# In[6]:


# Removing rows with missing Customer ID values
ecommerce_cleaned = data.dropna(subset=['Customer ID'])
# Converting the 'Date' column to datetime format
ecommerce_cleaned['Date'] = pd.to_datetime(ecommerce_cleaned['Date'], format='%m/%d/%Y')
# Checking for duplicate entries
duplicates = ecommerce_cleaned.duplicated().sum()
# Displaying the number of duplicates and the first few rows of the cleaned data
duplicates, ecommerce_cleaned.head()


# # EDA

# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns

# Product Analysis
#Calculating the total quantity sold for each product
product_quantity = ecommerce_cleaned.groupby('ProductName')['Quantity'].sum().sort_values(ascending=False).head(10)

#Calculating total sales (Price * Quantity) for each product
ecommerce_cleaned['TotalSales'] = ecommerce_cleaned['Price'] * ecommerce_cleaned['Quantity']
product_sales = ecommerce_cleaned.groupby('ProductName')['TotalSales'].sum().sort_values(ascending=False).head(10)

# Plotting the results
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.barplot(y=product_quantity.index, x=product_quantity.values, palette="viridis")
plt.title('Top 10 Most Popular Products by Quantity Sold')
plt.xlabel('Total Quantity Sold')
plt.ylabel('Product Name')

plt.subplot(1, 2, 2)
sns.barplot(y=product_sales.index, x=product_sales.values, palette="magma")
plt.title('Top 10 Products by Sales Revenue')
plt.xlabel('Total Sales Revenue')
plt.ylabel('Product Name')

plt.tight_layout()
plt.show()


# In[8]:


data['TotalSales'] = data['Price'] * data['Quantity']

# Customer Analysis
# Segmenting customers based on transaction frequency
customer_frequency = data.groupby('Customer ID')['TransactionNo'].nunique().sort_values(ascending=False).head(10)

# Segmenting customers based on total spending
customer_spending = data.groupby('Customer ID')['TotalSales'].sum().sort_values(ascending=False).head(10)

# Plotting the results
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.barplot(y=customer_frequency.index.astype(str), x=customer_frequency.values, palette="coolwarm")
plt.title('Top 10 Customers by Transaction Frequency')
plt.xlabel('Number of Transactions')
plt.ylabel('Customer ID')

plt.subplot(1, 2, 2)
sns.barplot(y=customer_spending.index.astype(str), x=customer_spending.values, palette="spring")
plt.title('Top 10 Customers by Total Spending')
plt.xlabel('Total Spending')
plt.ylabel('Customer ID')

plt.tight_layout()
plt.show()


# In[9]:


# Temporal Analysis
# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Grouping data by month
data['Month'] = data['Date'].dt.to_period('M')
monthly_sales = data.groupby('Month')['TotalSales'].sum()

# Grouping data by day
daily_sales = data.groupby(data['Date'].dt.date)['TotalSales'].sum()

# Plotting the results
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
monthly_sales.plot(kind='bar', color='skyblue')
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Total Sales')

plt.subplot(1, 2, 2)
daily_sales.plot(color='salmon')
plt.title('Daily Sales')
plt.xlabel('Date')
plt.ylabel('Total Sales')

plt.tight_layout()
plt.show()


# In[10]:


# Geographical Analysis

# Calculating total sales by country
country_sales = data.groupby('Country')['TotalSales'].sum().sort_values(ascending=False)

# Calculating total quantity sold by country
country_quantity = data.groupby('Country')['Quantity'].sum().sort_values(ascending=False)

# Plotting the results
plt.figure(figsize=(15, 12))

plt.subplot(2, 1, 1)
sns.barplot(y=country_sales.index, x=country_sales.values, palette="ocean")
plt.title('Total Sales by Country')
plt.xlabel('Total Sales')
plt.ylabel('Country')

plt.subplot(2, 1, 2)
sns.barplot(y=country_quantity.index, x=country_quantity.values, palette="ocean_r")
plt.title('Total Quantity Sold by Country')
plt.xlabel('Total Quantity Sold')
plt.ylabel('Country')

plt.tight_layout()
plt.show()


# In[11]:


import seaborn as sns

data['TotalSales'] = data['Price'] * data['Quantity']

# Transaction Analysis
#Calculate Average Basket Size
average_basket_size = data.groupby('TransactionNo')['ProductName'].count().mean()
print(f"Average Basket Size: {average_basket_size:.2f} products per transaction")

# Analyze Transaction Size Distribution
# By Sales Value
transaction_sales_distribution = data.groupby('TransactionNo')['TotalSales'].sum()
# By Quantity
transaction_quantity_distribution = data.groupby('TransactionNo')['Quantity'].sum()

# Visualizing the distribution of transaction sizes
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(transaction_sales_distribution, bins=30, kde=True, color='blue')
plt.title('Distribution of Transaction Sizes by Sales Value')
plt.xlabel('Transaction Size (Sales Value)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(transaction_quantity_distribution, bins=30, kde=True, color='green')
plt.title('Distribution of Transaction Sizes by Quantity')
plt.xlabel('Transaction Size (Quantity)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[12]:


#Calculate the total quantity for each customer
customer_total_quantity = data.groupby('Customer ID')['Quantity'].sum()

#Calculate the average quantity per customer
average_quantity_per_customer = customer_total_quantity.mean()
print(f"Average Quantity Per Customer: {average_quantity_per_customer:.2f}")

#Calculate the total spending for each customer
customer_total_spending = data.groupby('Customer ID')['TotalSales'].sum()

# Calculate the average spending per customer
average_spending_per_customer = customer_total_spending.mean()
print(f"Average Spending Per Customer: {average_spending_per_customer:.2f}")


# In[13]:


#Group by 'Customer ID'
grouped = data.groupby('Customer ID')

#Aggregate the required metrics
customer_behavior = grouped.agg(
    Total_Quantity=('Quantity', 'sum'),
    Total_Spending=('TotalSales', 'sum'),
    Number_of_Transactions=('TransactionNo', 'nunique')
)

# Calculate the Average Price Per Item for each customer
customer_behavior['Average_Price_Per_Item'] = customer_behavior['Total_Spending'] / customer_behavior['Total_Quantity']

#Resetting index to make 'Customer ID' a column
customer_behavior.reset_index(inplace=True)

#Display the first few rows of the DataFrame
customer_behavior.head()


# # MARKET BASKET ANALYSIS (MBA)

# In[14]:


from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Assuming 'ecommerce_data' has a column 'TransactionNo' and 'ProductName'

#Data Preparation
transactions = data.groupby('TransactionNo')['ProductName'].apply(list).values.tolist()
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

#Apply Apriori
frequent_itemsets = apriori(df, min_support=0.02, use_colnames=True)

#Generate Association Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

#Filter for pairs and sort
# Filter for pairs
rules = rules[rules['antecedents'].apply(lambda x: len(x) == 1) & rules['consequents'].apply(lambda x: len(x) == 1)]

#Sort by confidence, lift, or another metric
top_pairs = rules.sort_values(by='confidence', ascending=False).head(10)

#Creating a DataFrame for top 10 pairs
top_pairs_df = top_pairs[['antecedents', 'consequents', 'confidence', 'lift']]
top_pairs_df = top_pairs_df.reset_index(drop=True)

# Display the DataFrame
top_pairs.head()


# In[15]:


rules.head()


# In[ ]:





# In[ ]:





# In[ ]:




