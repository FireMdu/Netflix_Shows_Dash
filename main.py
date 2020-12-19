# Code for the streamlit app

import streamlit as st
import modeling

if __name__=='__main__':
    teams_df = modeling.EuropeanSoccerDatabase('.\data\database.sqlite')
    st.dataframe(teams_df.Team)
