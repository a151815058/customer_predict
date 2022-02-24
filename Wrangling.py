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
sale_data = pd.read_csv(S3_info.sale_data2['Body'])
print(sale_data.info())

## delete missing values
dfgap = sale_data[sale_data.isnull().any(axis=1)]

#delete categorydescription equal to '系統用','贈品','市場開發','店用商品'
dfgap = dfgap.drop(dfgap.loc[dfgap['categorydescription'].isin(['系統用','贈品','市場開發','店用商品'])].index)


# delete rows in which we cannot identify the customer
df1 = dfgap.copy()
df1 = df1[pd.notnull(df1["member_code"])]

# delete any missing values left
booMiss = df1.isnull().values.any()
if booMiss:
    _ = [print(k,":",v) for k,v in df1.isnull().sum().items() if v!=0]   # number missing

# restrict to transactions with positive quantities
df1 = df1[df1["sales_qty"] > 0]

# datetime to date format
df1["sale_date"] = pd.to_datetime(df1["sale_date"]).dt.date #normalize()


# treat CustomerID as a categorical variable
df1["member_code"] = df1["member_code"].astype(np.int64).astype(object)

#get YearMonth from InvoiceDate
df1['YearMonth'] = df1['sale_date'].map(lambda date:100 * date.year+date.month)
# review the categorical variables
df1.describe(include='object').T


# revenues = quantity * unitprice
df1["Revenue"] = df1["sales_qty"] * df1["unit_price"]

#add column customer_type
df1['customer_type'] = 'New'
df1['member_start_date'] = pd.to_datetime(df1['member_start_date'] )
df1.loc[df1['sale_date'] > df1['member_start_date'],'customer_type']='Existing'

#catch date <= 2019-12-31 as train data
df1_train = df1.loc[df1['sale_date'] <= datetime.strptime('20201031', "%Y%m%d").date()]
print((df1_train))

# delete columns that are not useful
try:
    df1_train = df1_train.drop(["sales_no", "stock_id", "name","member_start_date","first_trans_date"], axis=1)
except:
    pass
#df1.describe(include='object').T

# Create a directory to save sale_data with wrangle
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d")
directory = "sales_data_"+dt_string
path = os.path.join(os.path.dirname(os.path.abspath(__file__))+'/', directory)
os.mkdir(path)
df1_train.to_csv(path+'/sales_data.csv')



