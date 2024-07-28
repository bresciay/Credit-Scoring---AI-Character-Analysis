#!/usr/bin/env python
# coding: utf-8

# ## Data Preprocessing

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from scipy import stats
import csv


# In[2]:


# Read data dari csv 
df = pd.read_csv('C:/Users/ACER/Downloads/data/dataindividu11.csv')
df.head()


# Menampilkan info data, berdasarkan syntax dibawah ini didapatkan bahwa data tidak terdapat missing value, dan juga tipe data dari masing-masing variabel

# In[3]:


# Read data dari csv 
dfb = pd.read_csv("C:/Users/ACER/Downloads/data/databisnisss.csv",delimiter=";")
dfb.head()


# In[4]:


# Insert several variables from business to personal
df['type_of_business'] = dfb['type_of_business'].fillna("No")
df['industry_business_name'] = dfb['industry_business_name'].fillna("not included")
df['business_identification_number_industry'] = dfb['business_identification_number_industry'].fillna("not included")
df['credit_purpose'] = dfb['credit_purpose'].fillna("not included")
df['type_of_product'] = dfb['type_of_product'].fillna("not included")
df['average_sales_not_profit'] = dfb['average_sales_not_profit'].fillna("not included")
df['amount_of_principal_installment'] = dfb['amount_of_principal_installment'].fillna("No")
df['amount_of_principal_installment_dan_principal_interest'] = dfb['amount_of_principal_installment_dan_principal_interest'].fillna("not included")
df['company_address'] = dfb['company_address'].fillna("not included")
df['last_24_months_credit_perfomance'] = dfb['last_24_months_credit_perfomance'].fillna("not included")
df['business_prospects'] = dfb['business_prospects'].fillna("not included")
df['business_threat'] = dfb['business_threat'].fillna("not included")
df['key_person_ID_existence'] = dfb['key_person_ID_existence'].fillna("No")
df['existency_badan_usaha'] = dfb['existency_badan_usaha'].fillna("not included")
df['all_document_validity'] = dfb['all_document_validity'].fillna("not included")
df['falsification_of_signature_data_letter numbers'] = dfb['falsification_of_signature_data_letter numbers'].fillna("not included")
df['address_match'] = dfb['address_match'].fillna("not included")
df['suitability_of_the_size_of_the _company_with_transaction_value'] = dfb['suitability_of_the_size_of_the _company_with_transaction_value'].fillna("not included")
df['suitability_of_the_company_credit_type_with_ company_prospects'] = dfb['suitability_of_the_company_credit_type_with_ company_prospects'].fillna("No")
df['company_reputation_crawl_results'] = dfb['company_reputation_crawl_results'].fillna("not included")
df['caller_ID_name_on_the_core_of_company'] = dfb['caller_ID_name_on_the_core_of_company'].fillna("not included")
df['number_of_company_related_articles_on_google'] = dfb['number_of_company_related_articles_on_google'].fillna("not included")
df['character_of_the_owner_key_figures_in_media'] = dfb['character_of_the_owner_key_figures_in_media'].fillna("not included")
df['suitability_of_the_size_of_the _company_with_transaction_value'] = dfb['suitability_of_the_size_of_the _company_with_transaction_value'].fillna("not included")

# Move the variables
column = df.pop('type_of_business')
df.insert(31, 'type_of_business', column)
column = df.pop('industry_business_name')
df.insert(33, 'industry_business_name', column)
column = df.pop('business_identification_number_industry')
df.insert(34, 'business_identification_number_industry', column)
column = df.pop('credit_purpose')
df.insert(35, 'credit_purpose', column)
column = df.pop('type_of_product')
df.insert(37, 'type_of_product', column)
column = df.pop('average_sales_not_profit')
df.insert(38, 'average_sales_not_profit', column)
column = df.pop('amount_of_principal_installment')
df.insert(31, 'amount_of_principal_installment', column)
column = df.pop('amount_of_principal_installment_dan_principal_interest')
df.insert(33, 'amount_of_principal_installment_dan_principal_interest', column)
column = df.pop('company_address')
df.insert(34, 'company_address', column)
column = df.pop('last_24_months_credit_perfomance')
df.insert(35, 'last_24_months_credit_perfomance', column)
column = df.pop('business_prospects')
df.insert(37, 'business_prospects', column)
column = df.pop('business_threat')
df.insert(38, 'business_threat', column)
column = df.pop('key_person_ID_existence')
df.insert(31, 'key_person_ID_existence', column)
column = df.pop('existency_badan_usaha')
df.insert(33, 'existency_badan_usaha', column)
column = df.pop('all_document_validity')
df.insert(34, 'all_document_validity', column)
column = df.pop('falsification_of_signature_data_letter numbers')
df.insert(35, 'falsification_of_signature_data_letter numbers', column)
column = df.pop('address_match')
df.insert(37, 'address_match', column)
column = df.pop('suitability_of_the_size_of_the _company_with_transaction_value')
df.insert(38, 'suitability_of_the_size_of_the _company_with_transaction_value', column)
column = df.pop('suitability_of_the_company_credit_type_with_ company_prospects')
df.insert(31, 'suitability_of_the_company_credit_type_with_ company_prospects', column)
column = df.pop('company_reputation_crawl_results')
df.insert(33, 'company_reputation_crawl_results', column)
column = df.pop('caller_ID_name_on_the_core_of_company')
df.insert(34, 'caller_ID_name_on_the_core_of_company', column)
column = df.pop('number_of_company_related_articles_on_google')
df.insert(35, 'number_of_company_related_articles_on_google', column)
column = df.pop('character_of_the_owner_key_figures_in_media')
df.insert(37, 'character_of_the_owner_key_figures_in_media', column)
column = df.pop('suitability_of_the_size_of_the _company_with_transaction_value')
df.insert(38, 'suitability_of_the_size_of_the _company_with_transaction_value', column)

# Check the simple info
df.info()


# In[5]:


df.head()


# In[6]:


print(dfb.dtypes)


# In[7]:


#Convert all the non-numeric columns to numerical data types
for column in df.columns:
    if df[column].dtype == np.number: continue
    # Perform encoding for each non-numeric column
    df[column] = LabelEncoder().fit_transform(df[column])
df


# In[8]:


#df.to_csv('datalagi.csv', index=False)


# In[9]:


df.info()


# Menampilkan deskripsi data, yang terdapat count, mean, standar deviasi, nilai minimum, kuartil bawah, kuartil atas, dan nilai maksimum dari masing-masing variabel dataset

# In[10]:


df.describe()


# Untuk menampilkan banyaknya data yang hilang pada dataset, dan didapatkan hasil bahwa tidak ada data hilang pada dataset

# In[11]:


df.nunique()


# In[12]:


# Drop kolom 
#df.drop(['No', 'Nama', 'RT', 'RW', 'zip_code','IP_address','address','hardware_used','last_user_behavior_recorded_date'], axis=1, inplace =True)


# In[13]:


df.nunique()


# In[14]:


df.isnull().sum()


# Untuk menampilkan banyaknya data yang sama atau terduplicate, didapatkan bahwa tidak ada data yang sama atau terduplicate.
# 
# Untuk Menanggulangi data yang terduplicate dapat menggunakan syntax berikut ini
# 
# df.drop_duplicates(inplace=True)

# In[15]:


df.duplicated().sum()


# In[16]:


# df.drop_duplicates(inplace=True)


# Untuk menampilkan banyaknya data dan kolom, didapatkan bahwa data memiliki 50 variabel dengan masing-masing unit observasinya 100

# In[17]:


df.shape


# Transformasi data dengan memberikan keterangan pada data nominal dan ordinal

# In[18]:


df_vis=df.copy()


# In[19]:


df_vis["income_user"] = df_vis["income_user"].replace({0: "<5jt", 1: "5-10jt", 2: "10-15jt", 3: "15-20jt", 4:">20jt" }) #?nnnnnnnnnnnnnnnnnnnnnn
df_vis["road_type"] = df_vis["road_type"].replace({0: "jalan kecil", 1: "jalan sedang", 2:"jalan raya"})
df_vis["location_criminality"] = df_vis["location_criminality"].replace({0: "rendah", 1: "sedang", 2: "tinggi"})
df_vis["address_suitability"] = df_vis["address_suitability"].replace({0: 'tidak sesuai', 1: 'sesuai'})
df_vis["transaction_purpose"] = df_vis["transaction_purpose"].replace({0: "Konsumsi Pribadi", 1: "Investasi", 2: "Peralatan Kerja", 3: "Persediaan", 4:"Peralatan Rumah Tangga", 5:"Hadiah/Pemberian", 6: "Kebutuhan Bisnis"})
df_vis["payment_method"] = df_vis["payment_method"].replace({0: "Tunai", 1: "Transfer bank", 2: "Kartu Kredit", 3: "Debit Kartu", 4: "Pembayaran Elektronik", 5: "Leasing"})
df_vis["user_behavior"] = df_vis["user_behavior"].replace({0: "bad", 1: "normal", 2: "good"})
df_vis["last_credit_user"] = df_vis["last_credit_user"].replace({0: "no", 1: "yes"})
df_vis["last_user_behavior_status"] = df_vis["last_user_behavior_status"] .replace({0: "none", 1: "Bad", 2: "Good", 3: "Great"})
df_vis["worst_credit_performance"] = df_vis["worst_credit_performance"].replace({0: "none", 1: "macet", 2: "diragukan", 3: "dalam perhatian khusus", 4: "kurang lancar", 5: "lancar"})
df_vis["last_ SLIK_RECORD_performance"] = df_vis["last_ SLIK_RECORD_performance"].replace({0: "none", 1: "macet", 2: "diragukan", 3: "dalam perhatian khusus", 4: "kurang lancar", 5: "lancar"})
df_vis["last_ SLIK_RECORD_type_of_credit"] = df_vis["last_ SLIK_RECORD_type_of_credit"].replace({0: "none", 1: "Konsumsi Pribadi", 2: "Investasi", 3: "Peralatan Kerja", 4: "Persediaan", 5: "Peralatan Rumah Tangga", 6: "Hadiah/Pemberian", 7:"Kebutuhan Bisnis"})
df_vis["last_credit_survey_process"] = df_vis["last_credit_survey_process"].replace({0: "none", 1: "<1 bulan", 2: "1-3 bulan", 3: "3-6 bulan", 4: "6-9 bulan", 5: "9-12 bulan", 6: "> 12 bulan"})
df_vis["speed_of_credit_process"] = df_vis["speed_of_credit_process"].replace({0: "none", 1: "low", 2: "medium", 3: "high"})
df_vis["credit_scoring"] = df_vis["credit_scoring"].replace({0: "belum pernah", 1: "bad", 2: "poor", 3: "fair", 4: "good", 5: "excellent"})
df_vis["operating_system_type"] = df_vis["operating_system_type"].replace({0: "android", 1: "iOS", 2: "macOS", 3: "microsoft windows", 4: "Linux"})
df_vis["used_browser"] = df_vis["used_browser"].replace({0: "unknown", 1: "google chrome", 2: "Mozilla Firefox", 3: "Apple Safari", 4: "Opera", 5: "Brave"})
df_vis["credit_purpose"] = df_vis["credit_purpose"].replace({0: "Kredit rumah", 1: "Kredit mobil", 2: "Kredit konsumsi", 3: "Kredit usaha", 4: "Kredit investasi", 5: "Kredit kartu kredit"})
df_vis["means_of_communication"] = df_vis["means_of_communication"].replace({0: "phone", 1: "WA", 2: "surat", 3: "onsite", 4: "email"})
df_vis["still_have_existing_credit"] = df_vis["still_have_existing_credit"].replace({0: "none", 1: "sedang berjalan", 2: "lunas"})
df_vis["last_payment_type"] = df_vis["last_payment_type"].replace({0: "Tunai", 1: "Transfer bank", 2: "Kartu Kredit", 3: "Debit Kartu", 4: "Pembayaran Elektronik", 5: "Leasing", 6: "Bitcoin"})
df_vis["target"] = df_vis["target"].replace({0: "no fraud", 1: "suspect", 2: "medium risk", 3: "high risk", 4: "common fraud", 5: "mafia fraud", 6: "fraud"})
df_vis["type_of_business"] = df_vis["type_of_business"].replace({0: "Usaha Wholesale", 1: "Usaha Kecil", 2: "Usaha Menengah", 3: "Usaha Besar"})
df_vis["last_credit_history"] = df_vis["last_credit_history"].replace({0: "Lancar", 1: "Dalam Perhatian Khusus", 2: "Diragukan", 3: "Kurang Lancar", 4: "Tidak Lancar"})
df_vis["other_credit_history1"] = df_vis["other_credit_history1"].replace({0: "Lancar", 1: "Dalam Perhatian Khusus", 2: "Diragukan", 3: "Kurang Lancar", 4: "Tidak Lancar"})
df_vis["other_credit_history2"] = df_vis["other_credit_history2"].replace({0: "Lancar", 1: "Dalam Perhatian Khusus", 2: "Diragukan", 3: "Kurang Lancar", 4: "Tidak Lancar"})
df_vis["other_credit_history3"] = df_vis["other_credit_history3"].replace({0: "Lancar", 1: "Dalam Perhatian Khusus", 2: "Diragukan", 3: "Kurang Lancar", 4: "Tidak Lancar"})
df_vis["credit_purpose"] = df_vis["credit_purpose"].replace({0: "Modal Kerja", 1: "Modal Usaha", 2: "Ekspansi Bisnis", 3: "Biaya Operasional"})
df_vis["type_of_product"] = df_vis["type_of_product"].replace({0: "Baru", 1: "Bekas"})
df_vis["business_prospects"] = df_vis["business_prospects"].replace({0: "Low", 1: "Medium", 2: "High"})
df_vis["business_threat"] = df_vis["business_threat"].replace({0: "Low", 1: "Medium", 2: "High"})
df_vis["key_person_ID_existence"] = df_vis["key_person_ID_existence"].replace({0: "Real", 1: "Some Problem Exist", 2: "Undetected", 3: "Not Real"})
df_vis["existency_badan_usaha"] = df_vis["existency_badan_usaha"].replace({0: "Real", 1: "Some Problem Exist", 2: "Undetected", 3: "Not Real"})
df_vis["all_document_validity"] = df_vis["all_document_validity"].replace({0: "Valid", 1: "Unreadable", 2: "Not Include", 3: "Fraud", 4: "Expired"})
df_vis["falsification_of_signature_data_letter numbers"] = df_vis["falsification_of_signature_data_letter numbers"].replace({0: "Tidak Ada", 1: "Ada"})
df_vis["address_match"] = df_vis["address_match"].replace({0: "Sangat Sesuai", 1: "Cukup Sesuai", 2: "Tidak Sesuai"})
df_vis["suitability_of_the_size_of_the _company_with_transaction_value"] = df_vis["suitability_of_the_size_of_the _company_with_transaction_value"].replace({0: "Sangat Sesuai", 1: "Cukup Sesuai", 2: "Tidak Sesuai"})
df_vis["suitability_of_the_company_credit_type_with_ company_prospects"] = df_vis["suitability_of_the_company_credit_type_with_ company_prospects"].replace({0: "Sangat Sesuai", 1: "Cukup Sesuai", 2: "Tidak Sesuai"})
df_vis["company_reputation_crawl_results"] = df_vis["company_reputation_crawl_results"].replace({0: "Sangat Sesuai", 1: "Cukup Sesuai", 2: "Tidak Sesuai"})
df_vis["caller_ID_name_on_the_core_of_company"] = df_vis["caller_ID_name_on_the_core_of_company"].replace({0: "Sangat Sesuai", 1: "Cukup Sesuai", 2: "Tidak Sesuai"})
df_vis["number_of_company_related_articles_on_google"] = df_vis["number_of_company_related_articles_on_google"].replace({0: "Sangat Sesuai", 1: "Cukup Sesuai", 2: "Tidak Sesuai"})
df_vis["character_of_the_owner_key_figures_in_media"] = df_vis["character_of_the_owner_key_figures_in_media"].replace({0: "Sangat Sesuai", 1: "Cukup Sesuai", 2: "Tidak Sesuai"})


# In[20]:


df_vis.head()


# Untuk melihat banyaknya nilai unik pada masing-masing variabel pada dataset

# In[21]:


unique_values = {}
for col in df_vis.columns:
    unique_values[col] = df_vis[col].value_counts().shape[0]

pd.DataFrame(unique_values, index=["unique value count"]).transpose()


# In[22]:


df['income_user'] = df['income_user'].astype(int) 
df['asset_total'] = df['asset_total'].astype(int)
df['debt'] = df['debt'].astype(int)
df['location_criminality'] = df['location_criminality'].astype(int)
df['address_suitability'] = df['address_suitability'].astype(int)
df['amount_of_money_spent'] = df['amount_of_money_spent'].astype(int)
df['last_12_months_cost_of_collection'] = df['last_12_months_cost_of_collection'].astype(int)
df['credit_scoring'] = df['credit_scoring'].astype(int)
df['tenor'] = df['tenor'].astype(int)
df['ticket_size'] = df['ticket_size'].astype(int)
df['loan_to_value'] = df['loan_to_value'].astype(int)
df['besar_angsuran_pokok'] = df['besar_angsuran_pokok'].astype(int)
df['besar_angsuran_pokok_dan_bunga'] = df['besar_angsuran_pokok_dan_bunga'].astype(int)
df['last_contact_with_reply_or_mee_ in_days'] = df['last_contact_with_reply_or_mee_ in_days'].astype(int)	
df['still_have_existing_credit'] = df['still_have_existing_credit'].astype(int)
df['target'] = df['target'].astype(int)
df["type_of_business"] = df["type_of_business"].astype(int)
df["last_credit_history"] = df["last_credit_history"].astype(int)
df["other_credit_history1"] = df["other_credit_history1"].astype(int)
df["other_credit_history2"] = df["other_credit_history2"].astype(int)
df["other_credit_history3"] = df["other_credit_history3"].astype(int)
df["credit_purpose"] = df["credit_purpose"].astype(int)
df["type_of_product"] = df["type_of_product"].astype(int)
df["means_of_communication"] = df["means_of_communication"].astype(int)
df["still_have_existing_credit"] = df["still_have_existing_credit"].astype(int)
df["last_payment_type"] = df["last_payment_type"].astype(int)
df["last_user_behavior_status"] = df["last_user_behavior_status"].astype(int)
df["speed_of_credit_process"] = df["speed_of_credit_process"].astype(int)
df["business_prospects"] = df["business_prospects"].astype(int)
df["business_threat"] = df["business_threat"].astype(int)
df["key_person_ID_existence"] = df["key_person_ID_existence"].astype(int)
df["existency_badan_usaha"] = df["existency_badan_usaha"].astype(int)
df["all_document_validity"] = df["all_document_validity"].astype(int)
df["falsification_of_signature_data_letter numbers"] = df["falsification_of_signature_data_letter numbers"].astype(int)
df["address_match"] = df["address_match"].astype(int)
df["suitability_of_the_size_of_the _company_with_transaction_value"] = df["suitability_of_the_size_of_the _company_with_transaction_value"].astype(int)
df["suitability_of_the_company_credit_type_with_ company_prospects"] = df["suitability_of_the_company_credit_type_with_ company_prospects"].astype(int)
df["company_reputation_crawl_results"] = df["company_reputation_crawl_results"].astype(int)
df["caller_ID_name_on_the_core_of_company"] = df["caller_ID_name_on_the_core_of_company"].astype(int)
df["number_of_company_related_articles_on_google"] = df["number_of_company_related_articles_on_google"].astype(int)
df["character_of_the_owner_key_figures_in_media"] = df["character_of_the_owner_key_figures_in_media"].astype(int)


# In[23]:


df_2 = pd.DataFrame({
    'income_user': df['income_user'], 
    'asset_total': df['asset_total'], 
    'debt': df['debt'], 
    'road_type': df['road_type'], 
    'location_criminality': df['location_criminality'], 
    'address_suitability': df['address_suitability'], 
    'amount_of_money_spent': df['amount_of_money_spent'], 
    'transaction_purpose': df['transaction_purpose'], 
    'payment_method' : df['payment_method'], 
    'user_behavior' : df['user_behavior'], 
    'last_credit_user' : df['last_credit_user'], 
    'last_user_behavior_status' : df['last_user_behavior_status'], 
    'last_credit_history' : df['last_credit_history'], 
    'other_credit_history1' : df['other_credit_history1'], 
    'other_credit_history2' : df['other_credit_history2'], 
    'other_credit_history3' : df['other_credit_history3'], 
    'worst_credit_performance' : df['worst_credit_performance'], 
    'last_12_months_cost_of_collection' : df['last_12_months_cost_of_collection'], 
    'last_ SLIK_RECORD_performance' : df['last_ SLIK_RECORD_performance'],
    'last_ SLIK_RECORD_type_of_credit' : df['last_ SLIK_RECORD_type_of_credit'], 
    'last_credit_survey_process' : df['last_credit_survey_process'], 
    'speed_of_credit_process' : df['speed_of_credit_process'], 
    'credit_scoring' : df['credit_scoring'], 
    'operating_system_type' : df['operating_system_type'], 
    'used_browser' : df['used_browser'], 
    'credit_purpose' : df['credit_purpose'], 
    'tenor' : df['tenor'], 
    'ticket_size' : df['ticket_size'], 
    'loan_to_value' : df['loan_to_value'], 
    'besar_angsuran_pokok': df['besar_angsuran_pokok'], 
    'besar_angsuran_pokok_dan_bunga': df['besar_angsuran_pokok_dan_bunga'], 
    'last_contact_with_reply_or_mee_ in_days' : df['last_contact_with_reply_or_mee_ in_days'], 
    'means_of_communication' : df['means_of_communication'],
    'still_have_existing_credit' : df['still_have_existing_credit'], 
    'last_payment_type' : df['last_payment_type'], 
    'type_of_business': df['type_of_business'],   
    'credit_purpose': df['credit_purpose'], 
    'type_of_product': df['type_of_product'],     
    'business_prospects' : df['business_prospects'], 
    'business_threat' : df['business_threat'], 
    'key_person_ID_existence' : df['key_person_ID_existence'], 
    'existency_badan_usaha' : df['existency_badan_usaha'], 
    'all_document_validity' : df['all_document_validity'],
    'falsification_of_signature_data_letter numbers' : df['falsification_of_signature_data_letter numbers'],  
    'address_match' : df['address_match'],  
    'suitability_of_the_size_of_the _company_with_transaction_value' : df['suitability_of_the_size_of_the _company_with_transaction_value'], 
    'suitability_of_the_company_credit_type_with_ company_prospects' : df['suitability_of_the_company_credit_type_with_ company_prospects'], 
    'company_reputation_crawl_results' : df['company_reputation_crawl_results'], 
    'caller_ID_name_on_the_core_of_company' : df['caller_ID_name_on_the_core_of_company'], 
    'number_of_company_related_articles_on_google': df['number_of_company_related_articles_on_google'], 
    'character_of_the_owner_key_figures_in_media': df['character_of_the_owner_key_figures_in_media'],  
    'target': df['target'] 
})
df_2


# ## Exploratory Data Analysis

# In[24]:


df.info()


# In[25]:


df.shape


# In[26]:


df.target.value_counts()


# In[27]:


categorical_cols = ['road_type', 'transaction_purpose', 'payment_method', 'user_behavior', 'last_user_behavior_status', 'last_credit_history', 'other_credit_history1', 'other_credit_history2', 'other_credit_history3', 
                    'last_ SLIK_RECORD_type_of_credit', 'means_of_communication', 'last_payment_type', 'type_of_business', 'credit_purpose', 'type_of_product', 'business_prospects', 'business_threat', 
                    'company_reputation_crawl_results', 'character_of_the_owner_key_figures_in_media', 'nama', 'asset', 'address', 'type_of_product_purchased', 'last_ SLIK_RECORD_performance', 'hardware_used', 
                    'industry_business_name', 'worst_credit_performance', 'still_have_existing_credit', 'company_address', 'speed_of_credit_process', 'key_person_ID_existence', 
                    'existency_badan_usaha', 'all_document_validity', 'falsification_of_signature_data_letter numbers', 'address_match', 'suitability_of_the_size_of_the _company_with_transaction_value', 
                    'suitability_of_the_company_credit_type_with_ company_prospects', 'company_reputation_crawl_results', 'caller_ID_name_on_the_core_of_company', 'number_of_company_related_articles_on_google', 'asset_total']
numerical_cols = ['business_identification_number_industry', 'average_sales_not_profit', 'tenor', 'ticket_size', 'income_user', 'debt', 'RT', 'RW', 'zip_code', 'loan_to_value',
                  'amount_of_principal_installment', 'IP_address', 'address_suitability', 'amount_of_money_spent', 'amount_of_principal_installment_dan_principal_interest', 
                  'last_contact_with_reply_or_mee_ in_days', 'last_payment_time_days', 'last_user_behavior_recorded_date', 'last_24_months_credit_perfomance', 
                  'last_12_months_cost_of_collection', 'last_ SLIK_accessed_date', 'last_credit_survey_process', 'besar_angsuran_pokok_dan_bunga', 'besar_angsuran_pokok', 
                  'operating_system_type', 'location_criminality']
target_cols = 'credit_scoring'


# ### Categorical Data

# In[28]:


for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.set_style('whitegrid')
    plt.title(col)
    sns.barplot(x=col, y='count', data=df[col].value_counts().reset_index().rename(columns={'index':col, col:'count'}))
    plt.xticks(rotation=90)
    plt.show()


# ### Numerical Data

# In[29]:


# histogram
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.set_style('whitegrid')
    plt.title(col)
    sns.histplot(df[col], kde=True)
    plt.show()


# In[30]:


# boxplot for outlier detection
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.set_style('whitegrid')
    plt.title(col)
    sns.boxplot(df[col])
    plt.show()


# ### Exploratory Data Analysis (Multivariate Analysis)

# In[31]:


# pairplot
sns.pairplot(df[numerical_cols], diag_kind='kde')
plt.gcf().set_size_inches(10, 10)
plt.show()


# In[32]:


# melihat korelasi antar kolom
plt.figure(figsize=(16, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[33]:


df.corr()


# ### Encoding Fitur Categorical

# In[34]:


df_encoded = df.copy()


# In[35]:


from sklearn.preprocessing import LabelEncoder

# label encoding
le = LabelEncoder()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])
    
df_encoded.head()


# In[36]:


# korelasi antar kolom setelah label encoding
plt.figure(figsize=(20, 16))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[37]:


df_encoded.corr()


# In[38]:


# get variable with high correlation
high_corr = df_encoded.corr()[abs(df_encoded.corr()) > 0.4]
plt.figure(figsize=(20, 16))
sns.heatmap(high_corr, annot=True, cmap='coolwarm')
plt.show()


# In[39]:


#--- Menghilangkan variabel dengan nilai korelasi < 0.4

# calculate correlation between features and target
corr_with_target = df_encoded.corrwith(df['target'])

# get variables with correlation < 0.4
low_corr_vars = corr_with_target[abs(corr_with_target) < 0.4].index.tolist()

# drop variables with low correlation
df_encoded = df_encoded.drop(low_corr_vars, axis=1)


# In[40]:


# korelasi antar kolom setelah label encoding
plt.figure(figsize=(20, 16))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[41]:


df_encoded.corr()


# In[42]:


categorical = ['road_type', 'transaction_purpose', 'payment_method', 'user_behavior', 'last_user_behavior_status', 'last_credit_history', 'other_credit_history1', 'other_credit_history2', 'other_credit_history3', 
                    'last_ SLIK_RECORD_type_of_credit', 'means_of_communication', 'last_payment_type', 'type_of_business', 'credit_purpose', 'type_of_product', 'business_prospects', 'business_threat', 
                    'company_reputation_crawl_results', 'character_of_the_owner_key_figures_in_media', 'nama', 'asset', 'address', 'type_of_product_purchased', 'last_ SLIK_RECORD_performance', 'hardware_used', 
                    'industry_business_name', 'worst_credit_performance', 'still_have_existing_credit', 'company_address', 'speed_of_credit_process', 'key_person_ID_existence', 
                    'existency_badan_usaha', 'all_document_validity', 'falsification_of_signature_data_letter numbers', 'address_match', 'suitability_of_the_size_of_the _company_with_transaction_value', 
                    'suitability_of_the_company_credit_type_with_ company_prospects', 'company_reputation_crawl_results', 'caller_ID_name_on_the_core_of_company', 'number_of_company_related_articles_on_google', 'asset_total']
numerical = ['business_identification_number_industry', 'average_sales_not_profit', 'tenor', 'ticket_size', 'income_user', 'debt', 'RT', 'RW', 'zip_code', 'loan_to_value',
                  'amount_of_principal_installment', 'IP_address', 'address_suitability', 'amount_of_money_spent', 'amount_of_principal_installment_dan_principal_interest', 
                  'last_contact_with_reply_or_mee_ in_days', 'last_payment_time_days', 'last_user_behavior_recorded_date', 'last_24_months_credit_perfomance', 
                  'last_12_months_cost_of_collection', 'last_ SLIK_accessed_date', 'last_credit_survey_process', 'besar_angsuran_pokok_dan_bunga', 'besar_angsuran_pokok', 
                  'operating_system_type', 'location_criminality', 'credit_scoring']


# In[43]:


numeric_df = df[numerical]
categorical_df = df[categorical]


# In[44]:


# One-hot encode the categorical features
encoder = OneHotEncoder()
encoded_categorical_data = encoder.fit_transform(categorical_df)


# In[45]:


import joblib

joblib.dump(encoder, "onehot_encoder.joblib")


# ## Uji Korelasi

# In[46]:


unique_values = {}
for col in df_vis.columns:
    unique_values[col] = df_vis[col].value_counts().shape[0]

pd.DataFrame(unique_values, index=["unique value count"]).transpose()


# In[47]:


df


# In[48]:


# Create correlation matrix
correlation = df.corr()
corr_matrix_absolute = correlation.abs()

# Select upper triangle of correlation matrix
upper = corr_matrix_absolute.where(np.triu(np.ones(corr_matrix_absolute.shape), k=1).astype(bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.4)]
to_drop


# In[49]:


import seaborn as sns
plt.figure(figsize=(30,15))
sns.heatmap(correlation,xticklabels=correlation.columns,
            yticklabels=correlation.columns)


# In[50]:


plt.figure(figsize=(30,15))
sns.heatmap(correlation,cmap="BrBG",annot=True)


# In[51]:


df.columns


# In[52]:


# Step 1: Check the names of all the columns in the DataFrame
print(df.columns)

# Step 2: Compare the names of the columns in the DataFrame with the names in the `column_drop` list
columns_drop = ['asset_total','debt','road_type','transaction_purpose','payment_method', 'last_credit_user', 
               'last_user_behavior_status', 'last_credit_history','other_credit_history1',
               'other_credit_history2', 'other_credit_history3','last_ SLIK_RECORD_performance',
               'last_ SLIK_RECORD_type_of_credit', 'speed_of_credit_process','operating_system_type',
               'used_browser', 'credit_purpose', 'tenor/bulan', 'ticket_size', 'besar_angsuran_pokok',
               'means_of_communication', 'still_have_existing_credit', 'last_payment_type','last_ SLIK_accessed_date',
               'type_of_product_purchased', 'asset']

missing_columns = set(columns_drop) - set(df.columns)

if missing_columns:
    # Step 3: If some of the names in the `column_drop` list are not present in the DataFrame, remove them from the list
    print(f"The following columns are missing in the DataFrame: {missing_columns}")
    columns_drop = list(set(columns_drop) - missing_columns)

df.drop(columns_drop, axis=1, inplace=True)


# In[53]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),cmap="BrBG",annot=True)


# In[54]:


df.info()


# ## Membuat Model (Machine Learning)

# In[55]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[56]:


x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# In[57]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# ## Model Decision Tree Classifier

# In[58]:


from sklearn.tree import DecisionTreeClassifier


# In[59]:


dt = DecisionTreeClassifier()


# In[60]:


dt.fit(x_train, y_train)
pred_fraud_dt = dt.predict(x_test)
accuracy = accuracy_score(y_test, pred_fraud_dt)
print("Accuracy:", accuracy)


# In[61]:


pip install lazypredict


# In[62]:


import lazypredict
from lazypredict.Supervised import LazyClassifier
from lazypredict.Supervised import LazyRegressor


# In[63]:


clf = LazyClassifier()
models_clf, predictions_clf = clf.fit(x_train, x_test, y_train, y_test)
models_clf


# In[64]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier


# #### Decision Tree

# In[65]:


from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier()
BG = BaggingClassifier(base_estimator=DTC, n_estimators=100, random_state=42)


# In[66]:


dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
pred_fraud_dt = dt.predict(x_test)
accuracy = accuracy_score(y_test, pred_fraud_dt)
print("Accuracy:", accuracy)


# In[67]:


DTC_new = DecisionTreeClassifier(random_state=42, criterion='entropy', max_depth=None, max_features=None, min_samples_leaf=1, min_samples_split=5)
DTC_new.fit(x_train, y_train)

y_pred_DTC_new = DTC_new.predict(x_test)

# menampilkan hasil prediksi
score_train_DTC_new = DTC_new.score(x_train, y_train)
score_test_DTC_new = DTC_new.score(x_test, y_test)
print("Score Train:", score_train_DTC_new)
print("Score Test:", score_test_DTC_new)


# #### Bagging Classifier

# In[68]:


BGC = BaggingClassifier(base_estimator=DTC, n_estimators=100, random_state=42)
BGC = BGC.fit(x_train, y_train)

y_pred_BGC = BGC.predict(x_test)

# menampilkan hasil prediksi
score_train_BGC = BGC.score(x_train, y_train)
score_test_BGC = BGC.score(x_test, y_test)
print("Score Train:", score_train_BGC)
print("Score Test:", score_test_BGC)


# In[69]:


BGC_new = BaggingClassifier(base_estimator=DTC_new, n_estimators=100, random_state=42)
BGC_new = BGC_new.fit(x_train, y_train)

y_pred_BGC_new = BGC_new.predict(x_test)

# menampilkan hasil prediksi
score_train_BGC_new = BGC_new.score(x_train, y_train)
score_test_BGC_new = BGC_new.score(x_test, y_test)
print("Score Train:", score_train_BGC_new)
print("Score Test:", score_test_BGC_new)


# #### Random Forest Classifier

# In[70]:


RFC = RandomForestClassifier(random_state=42)
RFC = RFC.fit(x_train, y_train)

y_pred_RFC = RFC.predict(x_test)

# menampilkan hasil prediksi
score_train_RFC = RFC.score(x_train, y_train)
score_test_RFC = RFC.score(x_test, y_test)
print("Score Train:", score_train_RFC)
print("Score Test:", score_test_RFC)


# In[71]:


RFC_new = RandomForestClassifier(random_state=42, n_estimators=200, max_features='auto', max_depth=5, criterion='entropy')
RFC_new = RFC_new.fit(x_train, y_train)

y_pred_RFC_new = RFC_new.predict(x_test)

# menampilkan hasil prediksi
score_train_RFC_new = RFC_new.score(x_train, y_train)
score_test_RFC_new = RFC_new.score(x_test, y_test)
print("Score Train:", score_train_RFC_new)
print("Score Test:", score_test_RFC_new)


# #### Extra Trees Classifier

# In[72]:


pip install scikit-learn


# In[73]:


ETC = ExtraTreesClassifier(random_state=42)
ETC = ETC.fit(x_train, y_train)

y_pred_ETC = ETC.predict(x_test)

# menampilkan hasil prediksi
score_train_ETC = ETC.score(x_train, y_train)
score_test_ETC = ETC.score(x_test, y_test)
print("Score Train:", score_train_ETC)
print("Score Test:", score_test_ETC)


# In[74]:


ETC_new = ExtraTreesClassifier(random_state=42, n_estimators=200, max_features='auto', max_depth=5, criterion='entropy')
ETC_new = ETC_new.fit(x_train, y_train)

y_pred_ETC_new = ETC_new.predict(x_test)

# menampilkan hasil prediksi
score_train_ETC_new = ETC_new.score(x_train, y_train)
score_test_ETC_new = ETC_new.score(x_test, y_test)
print("Score Train:", score_train_ETC_new)
print("Score Test:", score_test_ETC_new)


# In[75]:


# mencoba model regresi
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models_reg, predictions_reg = reg.fit(x_train, x_test, y_train, y_test)
models_reg


# ## Evaluation

# In[78]:


# melakukan cross validation untuk mengecek overfitting
from sklearn.model_selection import cross_val_score

model_list = [("DTC", DTC), ("DTC_new", DTC_new), ("BGC", BGC), ("BGC_new", BGC_new), ("RFC", RFC), ("RFC_new", RFC_new), ("ETC", ETC), ("ETC_new", ETC_new)]
val_result = []

for model in model_list:
    scores = cross_val_score(model[1], x_train, y_train, cv=10, scoring='accuracy')
    val_result.append(scores)
    print(f"{model[0]}: {scores.mean():.2f} (+/- {scores.std():.2f})")


# In[80]:


# perbandingan hasil prediksi
acc = pd.DataFrame({
    'Model': ['DTC', 'DTC_new', 'BGC', 'BGC_new', 'RFC', 'RFC_new', 'ETC', 'ETC_new'],
    'Score Train': [score_train_DTC_new, score_train_DTC_new, score_train_BGC, score_train_BGC_new, score_train_RFC, score_train_RFC_new, score_train_ETC, score_train_ETC_new],
    'Score Test': [score_test_DTC_new, score_test_DTC_new, score_test_BGC, score_test_BGC_new, score_test_RFC, score_test_RFC_new, score_test_ETC, score_test_ETC_new],
    'CV Mean': [val_result[0].mean(), val_result[1].mean(), val_result[2].mean(), val_result[3].mean(), val_result[4].mean(), val_result[5].mean(), val_result[6].mean(), val_result[7].mean()],
    'CV Std': [val_result[0].std(), val_result[1].std(), val_result[2].std(), val_result[3].std(), val_result[4].std(), val_result[5].std(), val_result[6].std(), val_result[7].std()]
})
acc


# In[134]:


# confusion matrix
cm_DTC = confusion_matrix(y_test, y_pred_DTC_new)
cm_DTC_new = confusion_matrix(y_test, y_pred_DTC_new)
cm_BG = confusion_matrix(y_test, y_pred_BGC)
cm_BG_new = confusion_matrix(y_test, y_pred_BGC_new)
cm_RFC = confusion_matrix(y_test, y_pred_RFC)
cm_RFC_new = confusion_matrix(y_test, y_pred_RFC_new)
cm_ETC = confusion_matrix(y_test, y_pred_ETC)
cm_ETC_new = confusion_matrix(y_test, y_pred_ETC_new)

print("DTC:\n", cm_DTC)
print("\nDTC_new:\n", cm_DTC_new)
print("\nBG:\n", cm_BG)
print("\nBG_new:\n", cm_BG_new)
print("\nRFC:\n", cm_RFC)
print("\nRFC_new:\n", cm_RFC_new)
print("\nRFC:\n", cm_ETC)
print("\nRFC_new:\n", cm_ETC_new)


# ## Menyimpan Model

# In[161]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingClassifier
import pickle
import requests
import json


# In[162]:


# Saving the best model
pickle.dump(BG, open('BG_new_pickle.pkl', 'wb'))
pickle.dump(RFC, open('RFC_new_pickle.pkl', 'wb'))
pickle.dump(DTC, open('DTC_new_pickle.pkl', 'wb'))
pickle.dump(ETC, open('ETC_pickle.pkl', 'wb'))


# ## Test Model

# In[163]:


# Importing the saved best model
BGC_import = pickle.load(open('BGC_new_pickle.pkl', 'rb'))
RFC_import = pickle.load(open('RFC_new_pickle.pkl', 'rb'))
DTC_import = pickle.load(open('DTC_new_pickle.pkl', 'rb'))
ETC_import = pickle.load(open('ETC_pickle.pkl', 'rb'))


# In[164]:


BGC_import


# In[165]:


BGC_new_import.feature_names_in_


# In[ ]:





# In[ ]:





# In[ ]:





# In[145]:


'''import pickle

pickle.dump(BGC_new, open('BGC_new_pickle.pkl', 'wb'))'''


# In[146]:


BGC_new_import = pickle.load(open('BGC_new_pickle.pkl', 'rb'))


# In[147]:


BGC_new_import


# In[125]:


#BGC_new_import.__dict__


# In[126]:


#dir(BGC_new_import)


# In[127]:


BGC_new_import.feature_names_in_


# In[120]:


#print(dir(BaggingClassifier))                                                                                             


# In[159]:


x_test.iloc[0]


# In[115]:


np.array(x_test.iloc[0]).reshape(1, -1)


# In[116]:


BGC_new_import.predict(np.array(x_test.iloc[0]).reshape(1, -1))


# In[ ]:





# In[ ]:




