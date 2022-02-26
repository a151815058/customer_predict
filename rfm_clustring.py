import os
from datetime import datetime
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
pd.options.display.float_format = '{:,.0f}'.format

#get sales_data with wrangle
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d")
directory = "sales_data_"+dt_string
path = os.path.join(os.path.dirname(os.path.abspath(__file__))+'/', directory)
ltv_data = pd.read_csv(path+'/ltv_with_clustring.csv')

def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

#def寫python的函式
def Outlier_treatment(datacolumn): #傳入datacolumn這個變數
  Q1,Q3 =np.percentile(datacolumn,[25,75]) #算出Q1=25 Q3=75 這個是固定的
  IQR = Q3-Q1
  lower_range = Q1 - (1.5*IQR)
  high_range = Q3 + (1.5*IQR)
  return lower_range,high_range


#use kmeans
#R
df_rft1 = ltv_data.copy()
df_rft1['R'] = df_rft1['T']-df_rft1['recency']
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_rft1[['R']])
df_rft1['RecencyCluster'] = kmeans.predict(df_rft1[['R']])
df_rft1 = order_cluster('RecencyCluster', 'R',df_rft1,False)
print(df_rft1.groupby('RecencyCluster')['R'].describe())

#F
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_rft1[['frequency']])
df_rft1['FrequencyCluster'] = kmeans.predict(df_rft1[['frequency']])
df_rft1 = order_cluster('FrequencyCluster', 'frequency',df_rft1,True)
print(df_rft1.groupby('FrequencyCluster')['frequency'].describe())

#M
# drop outlier
lowerbound,highbound = Outlier_treatment(df_rft1['monetary_value'])
df_rft1 = df_rft1[(df_rft1['monetary_value'] < highbound)]
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_rft1[['monetary_value']])
df_rft1['RevenueCluster'] = kmeans.predict(df_rft1[['monetary_value']])
df_rft1 = order_cluster('RevenueCluster', 'monetary_value',df_rft1,True)
print(df_rft1.groupby('RevenueCluster')['monetary_value'].describe())


#OverallScore
df_rft1['OverallScore'] = df_rft1['RecencyCluster'] + df_rft1['FrequencyCluster'] + df_rft1['RevenueCluster']
print(df_rft1.groupby('OverallScore')['R'].mean())
print(df_rft1.groupby('OverallScore')['frequency'].mean())
print(df_rft1.groupby('OverallScore')['monetary_value'].mean())

df_rft1['rfm_Segment'] = 'Low-Value'
df_rft1.loc[df_rft1['OverallScore']>2,'rfm_Segment'] = 'Mid-Value'
df_rft1.loc[df_rft1['OverallScore']>4,'rfm_Segment'] = 'High-Value'
df_rft1

df_rft1.to_csv(path+'/ltv_rfm_with_clustring.csv')
