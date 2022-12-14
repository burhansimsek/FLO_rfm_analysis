########################################################################################################
# Dataset Info
########################################################################################################
# master_id: unique customer id
# order_channel: Which channel of the shopping platform is used (Android, iOS, Desktop, Mobile)
# last_order_channel: Channel of last purchase
# first_order_date: date of customers first purchase
# last_order_date: date of customers last purchase
# last_order_date_online: date of customers last online purchase
# last_order_date_offline: date of customers last offline purchase
# order_num_total_ever_online: total number of purchase in online
# order_num_total_ever_offline: total number of purchase in offline
# customer_value_total_ever_offline: total price of offline purchase
# customer_value_total_ever_online: total price of online purchase
# interested_in_categories_12: List of categories in which the customer shopped in the last 12 months
# 19.942 observation unit, 12 variable
########################################################################################################
import datetime as dt
import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.options.mode.chained_assignment = None

df_ = pd.read_csv("M3_crm_analytics/my_codes/datasets/flo_data_20k.csv")
df = df_.copy()
df.head()

####################
# data understanding
####################
df.head(10)
df.columns
df.isnull().sum()
df.describe().T
df.dtypes
df["master_id"].nunique() == df.shape[0]  # not multiple

###################
# data preparation
###################
df.head()
df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_price"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

date_cols = [col for col in df.columns if "date" in col]
df[date_cols] = df[date_cols].apply(pd.to_datetime)
df.info()

####################
# variable analysis
####################
df.groupby("order_channel").agg({"master_id":"count",
                                 "total_order_num":"sum",
                                 "total_price":"sum"})
df.sort_values("total_price", ascending=False).head(10)
df.sort_values("total_order_num", ascending=False).head(10)


########################
# data preparation func
########################
def data_preparation(dataframe):
    dataframe["total_order_num"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["total_price"] = dataframe["customer_value_total_ever_offline"] + dataframe[
        "customer_value_total_ever_online"]
    date_colss = [col for col in dataframe.columns if "date" in col]
    dataframe[date_colss] = dataframe[date_colss].apply(pd.to_datetime)
    return dataframe


df = data_preparation(df)

########################
# rfm metrics
########################
# recency: today date - last order date
# frequency: total order number
# monetary: total order price

rfm = df[["master_id", "last_order_date", "total_order_num", "total_price"]]
today_date = df["last_order_date"].max() + dt.timedelta(days=2)
rfm["last_order_date"] = rfm["last_order_date"].apply(lambda x: (today_date - x).days)
rfm.columns = ["customer_id", "r_metrics", "f_metrics", "m_metrics"]
rfm

########################
# rfm scores
########################

rfm["r_scores"] = pd.qcut(rfm["r_metrics"], 5, labels=[5, 4, 3, 2, 1])
rfm["f_scores"] = pd.qcut(rfm["f_metrics"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["m_scores"] = pd.qcut(rfm["m_metrics"], 5, labels=[1, 2, 3, 4, 5])
rfm["RF_scores"] = rfm["r_scores"].astype(str) + rfm["f_scores"].astype(str)
rfm.head(10)

# RFM labeling with REGEX
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["RF_scores"].replace(seg_map, regex=True)
rfm.head(10)

rfm["segment"].value_counts()
rfm[["segment", "r_metrics", "f_metrics", "m_metrics"]].groupby("segment").agg(["mean", "count"])

# FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
# tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel
# olarak iletişime geçmek isteniliyor. Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden
# alışveriş yapan kişiler özel olarak iletişim kurulacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına
# kaydediniz.

target_segments_customer_ids = rfm[rfm["segment"].isin(["champions","loyal_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) &(df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
cust_ids.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)
cust_ids.shape


# Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen
# geçmişte iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler,
# uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniyor. Uygun profildeki müşterilerin
# id'lerini csv dosyasına kaydediniz.

target_segments_customer_ids = rfm[rfm["segment"].isin(["about_to_sleep", "at_Risk", "new_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) &
              (df["interested_in_categories_12"].str.contains("ERKEK") |
               df["interested_in_categories_12"].str.contains("COCUK"))]["master_id"]

cust_ids.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)
cust_ids.shape