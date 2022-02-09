import numpy as np
import pandas as pd
import S3_info
from datetime import datetime
import os


import warnings
warnings.filterwarnings("ignore")


pd.set_option("display.precision",2)
np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = '{:,.0f}'.format


## load data
sale_data = pd.read_csv(S3_info.sale_data['Body'])
print(sale_data.info())

## delete missing values
dfgap = sale_data[sale_data.isnull().any(axis=1)]

# delete rows in which we cannot identify the customer
df1 = sale_data.copy()
df1 = df1[pd.notnull(df1["CustomerID"])]

# delete any missing values left
booMiss = df1.isnull().values.any()
if booMiss:
    _ = [print(k,":",v) for k,v in df1.isnull().sum().items() if v!=0]   # number missing

# restrict to transactions with positive quantities
df1 = df1[df1["Quantity"] > 0]

# datetime to date format
df1["InvoiceDate"] = pd.to_datetime(df1["InvoiceDate"]).dt.date #normalize()


# treat CustomerID as a categorical variable
df1["CustomerID"] = df1["CustomerID"].astype(np.int64).astype(object)

#get YearMonth from InvoiceDate
df1['YearMonth'] = df1['InvoiceDate'].map(lambda date:100 * date.year+date.month)
# review the categorical variables
df1.describe(include='object').T


# revenues = quantity * unitprice
df1["Revenue"] = df1["Quantity"] * df1["UnitPrice"]

#add column customer_type
df1['customer_type'] = 'New'
df1['member_start_date'] = pd.to_datetime(df1['member_start_date'] )
df1.loc[df1['InvoiceDate'] > df1['member_start_date'],'customer_type']='Existing'

# delete columns that are not useful
try:
    d1 = df1.drop(["InvoiceNo", "StockCode", "Country","member_start_date","first_trans_date"], axis=1, inplace=True)
except:
    pass
df1.describe(include='object').T

# Create a directory to save sale_data with wrangle
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d")
directory = "sales_data_"+dt_string
path = os.path.join(os.path.dirname(os.path.abspath(__file__))+'/', directory)
os.mkdir(path)

df1.to_csv(path+'/sales_data.csv')



