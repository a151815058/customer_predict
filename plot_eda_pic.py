import pygal
import pandas as pd
pd.set_option('display.float_format',lambda x:'%.0f'% x)


def get_monthly_revenue(sale_data):
    tx_revenue = sale_data.groupby(['YearMonth'])['Revenue'].sum().reset_index()
    tx_revenue['YearMonth'] = tx_revenue['YearMonth'].astype(str)
    line_revenue_chart = pygal.Line()
    line_revenue_chart.add('Monthly Revenue',tx_revenue['Revenue'])
    return line_revenue_chart.render_data_uri()


def get_monthly_growth_rate(sale_data):
    tx_revenue = sale_data.groupby(['YearMonth'])['Revenue'].sum().reset_index()
    tx_revenue['GrowthRate'] = tx_revenue['Revenue'].pct_change()
    tx_revenue.loc[tx_revenue['GrowthRate'].isna(),'GrowthRate'] = 0
    line_monthly_growth_rate_chart = pygal.Line()
    line_monthly_growth_rate_chart.add('Monthly Growth Rate',tx_revenue['GrowthRate'])
    return line_monthly_growth_rate_chart.render_data_uri()

def get_monthly_order_count(sale_data):
    tx_sales_qty = sale_data.groupby(['YearMonth'])['Quantity'].sum().reset_index()
    bar_order_count_chart = pygal.Bar()
    bar_order_count_chart.add('Monthly Order Count', tx_sales_qty['Quantity'])
    return bar_order_count_chart.render_data_uri()

def get_average_revenue_per_order(sale_data):
    tx_revenue_avg = sale_data.groupby(['YearMonth'])['Revenue'].mean().reset_index()
    bar_average_revenue_chart = pygal.Bar()
    bar_average_revenue_chart.add('Average Revenue Per Order', tx_revenue_avg['Revenue'])
    return bar_average_revenue_chart.render_data_uri()

def get_monthly_active_customers(sale_data):
    tx_monthly_actives = sale_data.groupby(['YearMonth'])['CustomerID'].nunique().reset_index()
    Line_monthly_active_chart = pygal.Line()
    Line_monthly_active_chart.add('Monthly Active Customers', tx_monthly_actives['CustomerID'])
    return Line_monthly_active_chart.render_data_uri()
def get_revenu_per_month_for_new_and_existing_customers(sale_data):
    tx_customer_type_revenue = sale_data.groupby(['YearMonth', 'customer_type'])['Revenue'].sum().reset_index()
    Line_new_and_existing_chart = pygal.Line()
    Line_new_and_existing_chart.add('Revenu Per Month For Existing Customers', tx_customer_type_revenue.loc[tx_customer_type_revenue['customer_type'] == 'Existing','Revenue'])
    Line_new_and_existing_chart.add('Revenu Per Month For New Customers', tx_customer_type_revenue.loc[tx_customer_type_revenue['customer_type'] == 'New','Revenue'])
    return  Line_new_and_existing_chart.render_data_uri()

def get_new_customer_ratio(sale_data):
    tx_user_ratio = sale_data.query("customer_type == 'New'").groupby(['YearMonth'])['CustomerID'].nunique() / \
                    sale_data.query("customer_type == 'Existing'").groupby(['YearMonth'])['CustomerID'].nunique()
    tx_user_ratio = tx_user_ratio.reset_index()
    tx_user_ratio = tx_user_ratio.dropna()
    Line_new_custome_chart = pygal.Line()
    Line_new_custome_chart.add('New Customer Ratio', tx_user_ratio['CustomerID'] )
    return Line_new_custome_chart.render_data_uri()
