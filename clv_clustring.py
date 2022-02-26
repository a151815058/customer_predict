import os
from datetime import datetime
from sklearn.cluster import KMeans
import pandas as pd
pd.options.display.float_format = '{:,.0f}'.format

#get sales_data with wrangle
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d")
directory = "sales_data_"+dt_string
path = os.path.join(os.path.dirname(os.path.abspath(__file__))+'/', directory)
ltv_data = pd.read_csv(path+'/ltv.csv')

def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final


kmeans = KMeans(n_clusters=3)
kmeans.fit(ltv_data[['CLV']])
ltv_data['CLVCluster'] = kmeans.predict(ltv_data[['CLV']])

ltv_data = order_cluster('CLVCluster', 'CLV',ltv_data,True)

ltv_data.groupby('CLVCluster')['CLV'].describe()

di = {0:'Low', 1:'Mid', 2:'High'}
ltv_data['Segment'] = ltv_data['CLVCluster'].map(di)
print(ltv_data)

ltv_data.to_csv(path+'/ltv_with_clustring.csv')
