import streamlit as st
import pandas as pd
import pyarrow.parquet as pq

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header         = st.container()
dataset        = st.container()
features       = st.container()
model_training = st.container()

st.markdown(
    """
    <style>
    .main {
    background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def get_data(filename):
    taxi_data = pq.read_table(filename)
    taxi_data = taxi_data.to_pandas()
    return taxi_data

with header:
    st.title('Welcome to my awesome data science project!')

with dataset:
    st.header('NYC taxi dataset')
    st.text("I found this dataset at: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page")


    taxi_data = get_data('Data/taxi_data.parquet')
    taxi_data = taxi_data.head(100000)
    st.write(taxi_data.head())

    st.subheader('Pick-up location ID distribution on the NYC dataset')
    pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts()).head(50)
    st.bar_chart(pulocation_dist)
    

with features:
    st.header('The features I created')

    st.markdown('* **first feature:** I created this feature because of this... I calculated it using this log...')
    st.markdown('* **second feature:** I created this feature because of this... I calculated it using this log...')

with model_training:
    st.header('Time to train the model!')
    st.text("Here you get to choose the hyperparameters of the model and see how the performance changes.")

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider('Whats should the max_depth of the model be?', min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('How many trees should there be?', options=[100,200,300,'No limit'], index=0)


    input_feature = sel_col.selectbox('Which feature should be used as the input feature', taxi_data.keys())

    if n_estimators == 'No limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    x = taxi_data[[input_feature]].values
    y = taxi_data[['trip_distance']].values

    regr.fit(x,y)
    prediction = regr.predict(y)

    disp_col.subheader('Mean absolute error of the model is: ')
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader('Mean squared error of the model is:')
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader('R squared score of the model is:')
    disp_col.write(r2_score(y, prediction))