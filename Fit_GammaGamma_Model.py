import os
from datetime import datetime
import pandas as pd
from lifetimes import BetaGeoFitter,GammaGammaFitter

pd.options.display.float_format = '{:,.2f}'.format

#get predict rlt
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d")
directory = "sales_data_"+dt_string
path = os.path.join(os.path.dirname(os.path.abspath(__file__))+'/', directory)
print(path)
predict_rlt = pd.read_csv(path+'/predict_rlt.csv')

#Loading bgf model
path_model = os.path.join(os.path.dirname(os.path.abspath(__file__))+'/', 'model')
bgf = BetaGeoFitter()
bgf.load_model(path_model+'/bgf.pkl')

# select customers with monetary value > 0
df_rftv = predict_rlt[predict_rlt["monetary_value"] > 0]
print(df_rftv.describe())

# Gamma-Gamma model requires a Pearson correlation close to 0
# between purchase frequency and monetary value
corr_matrix = df_rftv[["monetary_value", "frequency"]].corr()
corr = corr_matrix.iloc[1,0]
print("Pearson correlation: %.3f" % corr)

# outlier
df_rftv[df_rftv["monetary_value"] == df_rftv["monetary_value"].max()]


# fitting the Gamma-Gamma model
ggf = GammaGammaFitter(penalizer_coef = 0)
ggf.fit(
        frequency = df_rftv["frequency"],
        monetary_value = df_rftv["monetary_value"],
        weights = None,
        verbose = True,
        tol = 1e-06,
        q_constraint = True)
pd.options.display.float_format = '{:,.3f}'.format
print(ggf.summary)

# compute customer lifetime value
DISCOUNT_a = 0.06                # annual discount rate
LIFE = 12                        # lifetime expected for the customers in months

discount_m = (1 + DISCOUNT_a)**(1/12) - 1     # monthly discount rate

clv = ggf.customer_lifetime_value(
        transaction_prediction_model = bgf,
        frequency = df_rftv["frequency"],
        recency = df_rftv["recency"],
        T = df_rftv["T"],
        monetary_value = df_rftv["monetary_value"],
        time = LIFE,
        freq = "D",
        discount_rate = discount_m)

df_rftv.insert(0, "CLV", clv)             # expected customer lifetime values
df_rftv= df_rftv.drop('Unnamed: 0',axis =1)
print(df_rftv.describe().T)

print(df_rftv.sort_values(by="CLV", ascending=False))

df_rftv.to_csv(path+'/ltv.csv')