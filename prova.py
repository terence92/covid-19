import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import datetime
from datetime import date, timedelta
from sklearn.cluster import KMeans
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
import plotly.offline as py
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import streamlit as st
st.title('COVID 19')

st.subheader('SINTOMI COVID 19')
symptoms={'symptom':['Fever',
        'Dry cough',
        'Fatigue',
        'Sputum production',
        'Shortness of breath',
        'Muscle pain',
        'Sore throat',
        'Headache',
        'Chills',
        'Nausea or vomiting',
        'Nasal congestion',
        'Diarrhoea',
        'Haemoptysis',
        'Conjunctival congestion'],'percentage':[87.9,67.7,38.1,33.4,18.6,14.8,13.9,13.6,11.4,5.0,4.8,3.7,0.9,0.8]}

symptoms=pd.DataFrame(data=symptoms,index=range(14))
symptoms

fig = px.bar(symptoms[['symptom', 'percentage']].sort_values('percentage', ascending=False), 
             y="percentage", x="symptom", color='symptom', 
             log_y=True, template='ggplot2', title='Symptom of  Coronavirus')
fig.show()
plt.figure(figsize=(15,15))
plt.title('Symptoms of Coronavirus',fontsize=20)    
plt.pie(symptoms['percentage'],autopct='%1.1f%%')
plt.legend(symptoms['symptom'],loc='best')

st.pyplot()

from wordcloud import WordCloud, ImageColorGenerator
text = " ".join(str(each) for each in symptoms.symptom)
wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="white").generate(text)
plt.figure(figsize=(10,6))
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))

st.pyplot()


# **Reading Data**

# In[6]:


data=pd.read_csv("/Users/pasquale/Desktop/dati/covid19_italy_region.csv")
data_with_time = pd.read_csv("/Users/pasquale/Desktop/dati/covid19-ita-regions.csv")
data1= data_with_time


# In[7]:


an_data = pd.read_csv("/Users/pasquale/Desktop/dati/COVID19_open_line_list.csv")


# In[8]:


comp = pd.read_excel('/Users/pasquale/Desktop/dati/COVID-19-3.27-top30-500.xlsx')


# In[9]:


province = pd.read_csv("/Users/pasquale/Desktop/dati/covid19_italy_province.csv")


# In[10]:


nRowsRead = 1000
action=  pd.read_csv('/Users/pasquale/Desktop/dati/Dataset_Italy_COVID_19.csv'  , delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)


#  **Looking into data**

# In[11]:


action.head()


# In[12]:


action.shape


# In[13]:


action.dropna(how = 'all',inplace = True)


# In[14]:


plt.figure(figsize=(15, 5))
plt.title('Acttions')
action.Action.value_counts().plot.bar();


# In[15]:


from wordcloud import WordCloud, ImageColorGenerator
text = " ".join(str(each) for each in action.Action)
wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="white").generate(text)
plt.figure(figsize=(10,6))
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))

st.pyplot()



