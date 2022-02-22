import os
from datetime import datetime
from lifetimes.utils import summary_data_from_transaction_data
import pandas as pd
pd.options.display.float_format = '{:,.0f}'.format



#get sales_data with wrangle
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d")
directory = "sales_data_"+dt_string
path = os.path.join(os.path.dirname(os.path.abspath(__file__))+'/', directory)
sales_data = pd.read_csv(path+'/sales_data.csv')

#get end date of observations
max_date = sales_data["sale_date"].max()

# determine recency, frequency, T, monetary value for each customer
df_rft = summary_data_from_transaction_data(
    transactions = sales_data,
    customer_id_col = "member_code",
    datetime_col = "sale_date",
    monetary_value_col = "Revenue",
    observation_period_end = max_date,
    freq = "D")

df_rft.to_csv(path+'/rft.csv')
