"""This app creates an interactive app"""
import streamlit as st
import pandas as pd
import numpy as np


def title():
    st.title('Uber pickups in NYC')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/streamlit-demo-data/'
        'uber-raw-data-sep14.csv.gz')

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

def get_data(nrows):
    data_load_state = st.text('Loading data...')
    data = load_data(nrows)
    data_load_state.text('Loading data...done!')

def write_data():
    st.subheader('Raw data')
    st.write(load_data(10000))

def busiest_hour():
    st.subheader('NUmber of pickups by hour')
    hist_values = np.histogram(load_data(10000)[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
    st.bar_chart(hist_values)

def pickup_density(nrows):
    st.subheader("Map of all pickups")
    st.map(load_data(nrows))

def filtered_pickup_density(nrows, *hours):
    # TODO: Extend this to more hours
    filtered_data = load_data(nrows)[(load_data(nrows)[DATE_COLUMN].dt.hour).isin(hours)]
    st.subheader(f'Map of all pickups {hours}:00')
    st.map(filtered_data)

def slider_pickup_density(nrows):
    hour = st.slider('hour', 0, 23, 17)
    filtered_data = load_data(nrows)[load_data(nrows)[DATE_COLUMN].dt.hour == hour]
    st.subheader(f'Map of all pickups {hour}:00')
    st.map(filtered_data)       

def toggle_data(nrows):
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.dataframe(load_data(nrows).style.highlight_max(axis=0))
        st.table(load_data(10))

def get_sidebar():
    add_selectbox = st.sidebar.selectbox(
            'How would you like to be contacted?',
            ('Email', 'Home phone', 'Mobile phone')
            )
    add_slider = st.sidebar.slider(
            'Select a range of values',
            0.0, 100.0, (25.0, 75.0))

def get_columns():
    left_column, right_column = st.beta_columns(2)
    left_column.button('Press me')

    with right_column:
        chosen = st.radio(
                'Sorting hat',
                ("Gryffindor", "Ravenclaw", "Hufflepuff", "Syltherin"))
        st.write(f"You are in {chosen} house!")

if __name__=="__main__":
    data_rows = 10000
    title()
    get_data(data_rows)
    #write_data()
    toggle_data(data_rows)
    busiest_hour()
    pickup_density(data_rows)
    filtered_pickup_density(data_rows, 17)
    slider_pickup_density(data_rows)
    get_sidebar()
    get_columns()
