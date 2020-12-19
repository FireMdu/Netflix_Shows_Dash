"""This script contains all tutorials on streamlit from their docs site"""
import streamlit as st
import pandas as pd
import numpy as np

# add text
def add_text():
    st.title('My first app')

def add_df():
    st.write("Here's our first attempt at using data to create a table:")
    st.write(pd.DataFrame({
        'first column': [1, 2, 3, 4],
        'second column': [10, 20, 30, 40]
        }))

def add_line_chart():
    chart_data = pd.DataFrame(np.random.randn(20,3), columns=['a', 'b', 'c'])
    st.line_chart(chart_data)

def add_map():
    map_data = pd.DataFrame(
            np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
            columns=['lat', 'lon']
            )
    st.map(map_data)

def add_checkboxes():
    if st.checkbox('Show dataframe'):
        chart_data = pd.DataFrame(
                np.random.randn(20, 3),
                columns = ['a', 'b', 'c'])
        st.line_chart(chart_data)

def add_selectbox():
    option = st.selectbox(
            'Which number do you like best?',
            [1, 2, 3, 4])
    'You selected: ', option

def add_sidebar():
    option = st.sidebar.selectbox(
            'Which number do you like best?',
            [1,2,3,4])
    'You selected: ', option

def add_widget_columns():
    left_column, right_column = st.beta_columns(2)
    pressed = left_column.button("Press me?")
    if pressed:
        right_column.write("Woohoo!")

    expander = st.beta_expander("FAQ")
    expander.write("Here you could put in some really, really long explanations...")

def show_progress():
    import time
    'Starting a long computation...'

    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        latest_iteration.text(f"Iteration {i + 1}")
        bar.progress(i+1)
        time.sleep(0.1)

    '...and now we\'re done!'


if __name__=="__main__":
    st.set_page_config(
            page_title="Streamlit Tutorial", 
            page_icon=r'.\images\page_icon.svg'
    )
    add_text()  
    add_df()
    add_map()
    add_line_chart()
    add_checkboxes()
    #add_selectbox()
    add_sidebar()
    add_widget_columns()
    show_progress()
