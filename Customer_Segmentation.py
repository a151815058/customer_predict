import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime

#get customer rftm
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d")
directory = "sales_data_"+dt_string
path = os.path.join(os.path.dirname(os.path.abspath(__file__))+'/', directory)
print(path)
c_rft = pd.read_csv(path+'/rft.csv')

#function for ordering cluster numbers
#把cluster做排序
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

def Outlier_treatment(datacolumn):
  Q1,Q3 = np.percentile(datacolumn,[25,75])
  IQR = Q3-Q1
  lower_range = Q1 - (1.5 * IQR)
  high_range = Q3 + (1.5 * IQR)
  return lower_range,high_range



#1. Recency
#build 4 clusters for recency and add it to dataframe
kmeans = KMeans(n_clusters=4)
kmeans.fit(c_rft[['recency']])
c_rft['RecencyCluster'] = kmeans.predict(c_rft[['recency']])
c_rft = order_cluster('RecencyCluster', 'recency',c_rft,False)

#2. Frequency
#k-means
kmeans = KMeans(n_clusters=4)
kmeans.fit(c_rft[['frequency']])
c_rft['FrequencyCluster'] = kmeans.predict(c_rft[['frequency']])
c_rft = order_cluster('FrequencyCluster', 'frequency',c_rft,True)

#3. Monetary
#remove outlier
lowerbound,highbound = Outlier_treatment(c_rft['monetary_value'])
c_rft = c_rft[(c_rft['monetary_value'] < highbound)]
#k-means
kmeans = KMeans(n_clusters=4)
kmeans.fit(c_rft[['monetary_value']])
c_rft['RevenueCluster'] = kmeans.predict(c_rft[['monetary_value']])
c_rft = order_cluster('RevenueCluster', 'monetary_value',c_rft,True)

#calculate overall score and use mean() to see details
c_rft['OverallScore'] = c_rft['RecencyCluster'] + c_rft['FrequencyCluster'] + c_rft['RevenueCluster']

#labeling
c_rft['Segment'] = 'Low-Value'
c_rft.loc[c_rft['OverallScore']>2,'Segment'] = 'Mid-Value'
c_rft.loc[c_rft['OverallScore']>4,'Segment'] = 'High-Value'


c_rft.to_csv(path+'/rft_with_clustring.csv')