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

#def寫python的函式
def Outlier_treatment(datacolumn): #傳入datacolumn這個變數
  Q1,Q3 =np.percentile(datacolumn,[25,75]) #算出Q1=25 Q3=75 這個是固定的
  IQR = Q3-Q1
  lower_range = Q1 - (1.5*IQR)
  high_range = Q3 + (1.5*IQR)
  return lower_range,high_range

## load data
sale_data = pd.read_csv(S3_info.sale_data2['Body'])
print(sale_data.info())

#delete categorydescription equal to '系統用','贈品','市場開發','店用商品'
dfgap = sale_data.drop(sale_data.loc[sale_data['categorydescription'].isin(['系統用','贈品','市場開發','店用商品'])].index)


#rename column
dfgap.rename(columns={'sales_no':'InvoiceNo'},inplace=True)
dfgap.rename(columns={'stock_id':'StockCode'},inplace=True)
dfgap.rename(columns={'stock_description':'Description'},inplace=True)
dfgap.rename(columns={'sales_qty':'Quantity'},inplace=True)
dfgap.rename(columns={'sale_date':'InvoiceDate'},inplace=True)
dfgap.rename(columns={'unit_price':'UnitPrice'},inplace=True)
dfgap.rename(columns={'member_code':'CustomerID'},inplace=True)
dfgap.rename(columns={'name':'Country'},inplace=True)
dfgap.rename(columns={'member_start_date':'member_start_date'},inplace=True)
dfgap.rename(columns={'first_trans_date':'first_trans_date'},inplace=True)

# delete rows in which we cannot identify the customer
df1 = dfgap.copy()
df1 = df1[pd.notnull(df1["CustomerID"])]

# delete any missing values left
booMiss = df1.isnull().values.any()
if booMiss:
    _ = [print(k,":",v) for k,v in df1.isnull().sum().items() if v!=0]   # number missing

# restrict to transactions with positive quantities
df1 = df1[df1["Quantity"] > 0]

# revenues = quantity * unitprice
df1["Revenue"] = df1["Quantity"] * df1["UnitPrice"]

#抓出Revenue異常值
lowerbound,highbound = Outlier_treatment(df1['Revenue'])
df1[(df1['Revenue']< lowerbound) | (df1['Revenue']>highbound)]

#去除異常值
df1.drop(df1[df1['Revenue']> highbound].index,inplace=True)

# datetime to date format
df1["InvoiceDate"] = pd.to_datetime(df1["InvoiceDate"]).dt.date #normalize()


# treat CustomerID as a categorical variable
df1["CustomerID"] = df1["CustomerID"].astype(np.int64).astype(object)

#get YearMonth from InvoiceDate
df1['YearMonth'] = df1['InvoiceDate'].map(lambda date:100 * date.year+date.month)
# review the categorical variables
df1.describe(include='object').T


#add column customer_type
df1['customer_type'] = 'New'
df1['member_start_date'] = pd.to_datetime(df1['member_start_date'] )
df1.loc[df1['InvoiceDate'] > df1['member_start_date'],'customer_type']='Existing'


# delete columns that are not useful
try:
    df1_train = df1.drop(["InvoiceNo", "StockCode","member_start_date","first_trans_date"], axis=1)
except:
    pass


# Create a directory to save sale_data with wrangle
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d")
directory = "sales_data_"+dt_string
path = os.path.join(os.path.dirname(os.path.abspath(__file__))+'/', directory)
os.mkdir(path)
df1_train.to_csv(path+'/sales_data_with_wrangling.csv')



