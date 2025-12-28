import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
st.title('🏠House Price prediction using ML')
st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQrEZFyB3eYVFx4fiwANh-wSdkFGVBmth2Gsg&s')
df=pd.read_csv('house_data.csv')
X=df.iloc[:, :-3]
y=df.iloc[:, -1]
st.sidebar.title('🏠House Price prediction using ML')
st. sidebar.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQrEZFyB3eYVFx4fiwANh-wSdkFGVBmth2Gsg&s')
all_value=[]
for i in X:
    min_value = int (X[i].min())
    max_value = int (X[i].max())
    ans =st.sidebar.slider(f'select{i} value',min_value,max_value)
    all_value.append(ans)
# st.write(all_value)
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)
final_value = scaler.transform([all_value])
model = st.cache(RandomForestRegressor())
model.fit(X,y)
house_price = model.predict(final_value)
with st.spinner('Predicting House price'):
    time.sleep(1)
    msg= f'''House price is : $ (house_price*100000)'''
    st.success(msg)
    st.markdown('''**design and developed by:Anshika**''')









