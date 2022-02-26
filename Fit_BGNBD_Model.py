import os
from datetime import datetime,timedelta
import pandas as pd
from lifetimes import  ModifiedBetaGeoFitter
from lifetimes.utils import calibration_and_holdout_data

pd.options.display.float_format = '{:,.1f}'.format


#get customer rftm
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d")
directory = "sales_data_"+dt_string
path = os.path.join(os.path.dirname(os.path.abspath(__file__))+'/', directory)
print(path)
c_rft = pd.read_csv(path+'/sales_data_with_wrangling.csv')

#Train/Test Split
# train/test split (calibration/holdout)
t_holdout = 240                                         # days to reserve for holdout period

max_date = c_rft["InvoiceDate"].max()                     # end date of observations
max_date = datetime.strptime(max_date, '%Y-%m-%d')
max_cal_date = max_date - timedelta(days=t_holdout)     # end date of chosen calibration period
# calibration:可以想成是training data
# holdout:可以想成是testing data
print(f'Training period: {c_rft["InvoiceDate"].min()} ~ {max_cal_date}')
print(f'Testing period: {max_cal_date+timedelta(days=1)} ~ {max_date}')

df_ch = calibration_and_holdout_data(
        transactions = c_rft,
        customer_id_col = "CustomerID",
        datetime_col = "InvoiceDate",
        monetary_value_col = "Revenue",
        calibration_period_end = max_cal_date,
        observation_period_end = max_date,
        freq = "D")

#Fit the BG/NBD Model
bgf = ModifiedBetaGeoFitter(penalizer_coef=1e-11)
bgf.fit(
        frequency = df_ch["frequency_cal"],
        recency = df_ch["recency_cal"],
        T = df_ch["T_cal"],
        weights = None,
        verbose = True,
        tol = 0.0001)

# training: summary
pd.options.display.float_format = '{:,.3f}'.format
bgf.summary

# rename column
df_rft = df_ch.copy()
df_rft.rename(columns={'frequency_cal':'frequency', 'recency_cal':'recency',
                       'T_cal':'T','monetary_value_cal':'monetary_value'},inplace=True)

# predict purchases for a selected customer for t days
t = 30
custID = 9100000000114

df_rft_C = df_rft.loc[custID,:]
predC = bgf.predict(    t,
                        df_rft_C["frequency"],
                        df_rft_C["recency"],
                        df_rft_C["T"])
print("customer", custID, ": expected number of purchases within", t, "days =", f'{predC:.1f}')


# helper function: predict each customer's purchases over next t days
def predict_purch(df, t):
        df["predict_purch_" + str(t)] = \
                bgf.predict(
                    t,
                    df["frequency"],
                    df["recency"],
                    df["T"])


# call helper function: predict each customer's purchases over multiple time periods
t_FC = [10, 30, 60, 90]
_ = [predict_purch(df_rft, t) for t in t_FC]
print("predicted number of purchases for each customer over next t days:")

# probability that a customer is alive for each customer in dataframe
prob_alive = bgf.conditional_probability_alive(
        frequency = df_rft["frequency"],
        recency = df_rft["recency"],
        T = df_rft["T"])

df_rft["prob_alive"] = prob_alive
pd.options.display.float_format = '{:,.2f}'.format
c_rft.describe()

df_rft["prob_alive"] = prob_alive

df_rft.to_csv(path+'/predict_rlt.csv')

# save model
path_model = os.path.join(os.path.dirname(os.path.abspath(__file__))+'/', 'model')
bgf.save_model(path_model+'/bgf.pkl')
