import os
from datetime import datetime
from flask import Flask,jsonify,render_template,request,flash
import pandas as pd
import numpy as np
import S3_info
from scipy.spatial import distance
import plot_eda_pic
import json

pd.set_option('display.float_format',lambda x:'%.2f'% x)


# Fetch the list of existing buckets
clientResponse = S3_info.client.list_buckets()

#get path
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d")
directory = "sales_data_"+dt_string
path = os.path.join(os.path.dirname(os.path.abspath(__file__))+'/', directory)

# Read sales_data
sale_data = pd.read_csv(path+'/sales_data.csv')

# Read customer ltv
c_ltv = pd.read_csv(path+'/ltv.csv')

#Read stocklens_stock_embedding
stocklens_stock_embedding = pd.read_csv(path+'/stocklens_stock_embedding.csv')



app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
app.jinja_env.auto_reload = True


@app.route('/',methods=['GET', 'POST'])
def index():
    return render_template('input_customer.html' ,index=False)

@app.route('/recommand_item',methods=['GET', 'POST'])
def recommand_item():
    return render_template('recommand_item.html', index=False)

@app.route('/recommand_result',methods=['GET', 'POST'])
def recommand_result():
    if request.method == 'POST':
        member_code = request.values['Name']
        last_item_description,recommand_item = get_recommand_item_10(member_code)
        return render_template('recommand_result.html',recommand_item =recommand_item.to_numpy(),
                                                       last_item_description = last_item_description.values.item(),
                                                       len = len(recommand_item),
                                                       member_code = member_code
                                )

@app.route('/result',methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        member_code = request.values['Name']
        rslt_df=get_predict_result(member_code)
        print(rslt_df)
        if rslt_df['Segment'].to_string(index=False) == 'Mid-Value':
            clustring = '中價值客戶群'
        elif rslt_df['Segment'].to_string(index=False) == 'Low-Value':
            clustring = '低價值客戶群'
        else:
            clustring = '高價值客戶群'

        return render_template('result.html',member_code=member_code,
                                             frequency=rslt_df['frequency'].to_string(index=False),
                                             recency=rslt_df['recency'].to_string(index=False),
                                             T=rslt_df['T'].to_string(index=False),
                                             clustring=clustring,
                                             monetary_value=rslt_df['monetary_value'].to_string(index=False),
                                             predict_purch_10=rslt_df['predict_purch_10'].to_string(index=False),
                                             predict_purch_30=rslt_df['predict_purch_30'].to_string(index=False),
                                             predict_purch_60=rslt_df['predict_purch_60'].to_string(index=False),
                                             predict_purch_90=rslt_df['predict_purch_90'].to_string(index=False),
                                             prob_alive=rslt_df['prob_alive'].to_string(index=False),
                                             CLV=rslt_df['CLV'].to_string(index=False)
                               )

@app.route('/return_json',methods=['GET', 'POST'])
def return_json():
    if request.method == 'POST':
        member_code = request.values['Name']
        rslt_df = get_predict_result(member_code)
        jsonfiles = rslt_df.to_dict('records')
        jsonStr = json.dumps(jsonfiles, indent = 4)
        print(jsonStr)
        return jsonify(jsonfiles)

@app.route('/Monthly_Revenue',methods=['GET', 'POST'])
def Monthly_Revenue():
    chart = plot_eda_pic.get_monthly_revenue(sale_data)
    return render_template('Monthly_Revenue.html', chart=chart)

@app.route('/Monthly_Growth_Rate',methods=['GET', 'POST'])
def Monthly_Growth_Rate():
    chart = plot_eda_pic.get_monthly_growth_rate(sale_data)
    return render_template('Monthly_Growth_Rate.html', chart=chart)

@app.route('/Monthly_Order_Count',methods=['GET', 'POST'])
def Monthly_Order_Count():
    chart = plot_eda_pic.get_monthly_order_count(sale_data)
    return render_template('Monthly_Order_Count.html', chart=chart)

@app.route('/Average_Revenue_per_Order',methods=['GET', 'POST'])
def Average_Revenue_per_Order():
    chart = plot_eda_pic.get_average_revenue_per_order(sale_data)
    return render_template('Average_Revenue_per_Order.html', chart=chart)

@app.route('/Monthly_Active_Customers',methods=['GET', 'POST'])
def Monthly_Active_Customers():
    chart = plot_eda_pic.get_monthly_active_customers(sale_data)
    return render_template('Monthly_Active_Customers.html', chart=chart)

@app.route('/Revenu_Per_Month_For_New_and_Existing_Customers',methods=['GET', 'POST'])
def Revenu_Per_Month_For_New_and_Existing_Customers():
    chart = plot_eda_pic.get_revenu_per_month_for_new_and_existing_customers(sale_data)
    return render_template('Revenu_Per_Month_For_New_and_Existing_Customers.html', chart=chart)

@app.route('/New_Customer_Ratio',methods=['GET', 'POST'])
def New_Customer_Ratio():
    chart = plot_eda_pic.get_new_customer_ratio(sale_data)
    return render_template('New_Customer_Ratio.html', chart=chart)

@app.route('/Get_RFM',methods=['GET', 'POST'])
def Get_RFM():
    if request.method == 'POST':
        file = open(os.path.dirname(os.path.abspath(__file__))+'/Wrangling.py', 'r').read()
        #exec(file)
        #while not os.path.exists(path+'/sales_data.csv'):
        #    return flash('Calculating')
        return '開發中'
    else:
        return render_template('Get_RFM.html')


if __name__ == '__main__':
    app.debug = True
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    app.jinja_env.auto_reload = True
    app.run()

def get_predict_result(member_code):
    rslt_df = c_ltv[c_ltv['member_code'] == np.float64(member_code)]
    rslt_df = rslt_df.drop('Unnamed: 0',axis=1)
    return rslt_df

def get_recommand_item_10(member_code):
    item_description = get_customer_last_product(member_code)
    stocklens_stock_embedding["vector"] = stocklens_stock_embedding["vector"].map(lambda x: np.array(json.loads(x)))

    item_embedding = stocklens_stock_embedding.loc[stocklens_stock_embedding["word"] == item_description.values.item(), "vector"].iloc[0]

    # 余弦相似度
    stocklens_stock_embedding["sim_value"] = stocklens_stock_embedding["vector"].map(lambda x: 1 - distance.cosine(item_embedding, x))
    # 按相似度降序排列，查询前10条
    recommand_item_result = stocklens_stock_embedding.sort_values(by="sim_value", ascending=False)[["word", "sim_value"]].head(10)
    return item_description,recommand_item_result

def get_customer_last_product(member_code):
    member1 = sale_data.loc[sale_data['member_code'] == np.float64(member_code)]
    Description = member1['stock_description'].tail(1)
    return Description



