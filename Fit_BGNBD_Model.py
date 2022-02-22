import os
from datetime import datetime
import pandas as pd
from lifetimes import BetaGeoFitter
pd.options.display.float_format = '{:,.1f}'.format


#get customer rftm
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d")
directory = "sales_data_"+dt_string
path = os.path.join(os.path.dirname(os.path.abspath(__file__))+'/', directory)
print(path)
c_rft = pd.read_csv(path+'/rft_with_clustring.csv')

# BG/NBD model
bgf = BetaGeoFitter(penalizer_coef=1e-06)
bgf.fit(
        frequency = c_rft["frequency"],
        recency = c_rft["recency"],
        T = c_rft["T"],
        weights = None,
        verbose = True,
        tol = 1e-06)
pd.options.display.float_format = '{:,.3f}'.format
print(bgf.summary)

# predict purchases for a selected customer for t days
t = 30
custID = 9100000072913

df_rft_C = c_rft.loc[c_rft['member_code'] == custID]
predC = bgf.predict(    t,
                        df_rft_C["frequency"],
                        df_rft_C["recency"],
                        df_rft_C["T"])
print(predC.values[0])
print("customer", custID, ": expected number of purchases within", t, "days =", f'{predC.values[0]:.1f}')

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
_ = [predict_purch(c_rft, t) for t in t_FC]
print("predicted number of purchases for each customer over next t days:")

# probability that a customer is alive for each customer in dataframe
prob_alive = bgf.conditional_probability_alive(
        frequency = c_rft["frequency"],
        recency = c_rft["recency"],
        T = c_rft["T"])

c_rft["prob_alive"] = prob_alive

c_rft.to_csv(path+'/predict_rlt.csv')

# save model
path_model = os.path.join(os.path.dirname(os.path.abspath(__file__))+'/', 'model')
bgf.save_model(path_model+'/bgf.pkl')
