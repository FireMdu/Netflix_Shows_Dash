"""This script produces a streamlit dashboard.
The script uses Netflix shows data from the .\data direvtory"""
import streamlit as st
import pandas as pd
import sqlite3 as sql
import matplotlib.pyplot as plt
import os
import pickle
import sys


class DataReader:
    def save_to_disk(self, filename):
        with open(filename, 'ab') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_disk(filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        return obj


class ReadSQLDatabase(DataReader):
    """Read sql data bases into pandas dataframe: Each pandas object read from sql 
    gets set as an attribute named with the table name"""

    def __init__(self, sql_database, *tab_names):
        for tab in tab_names:
            setattr(self, tab, self.read_sql_table(tab, sql.connect(sql_database)))      
    
    def read_sql_table(tab_name, conn):
        pandas_dataframe = pd.read_sql(
            f"""
            select *
            from {tab_name}
            """, 
            conn)
        return pandas_dataframe


class ReadExcelDatabase(DataReader):
    def __init__(self, pandas_excel_reader, file_obj, **kwargs):
        setattr(self, 'data', pandas_excel_reader(file_obj, **kwargs)) 


class EuropeanSoccerDatabase(ReadSQLDatabase):
    pass


@st.cache
class NetflixFilms(ReadExcelDatabase):
    pass


# Named data structures
class PivotTableArgsState:
    def __init__(self):
        pass


class CleaningRadioOpts:
    """Data structure for storing radio cleaning options"""
    def __init__(self):
        pass


data_file = " "
dash_data = None
cleaned_data = pd.DataFrame({})


def header_things():
    """Define header things like tittle, and other fancy things 
    I will find"""
    header = st.title("Its a Data App right?")
    text = st.text("Lets get some data then, shall we?")


def select_file():
    """Chose a file from a dropdown list of files"""

    global data_file
    #TODO: Set value to true when deploying
    chose_data_path = st.checkbox("Custom Data Path")
    database_extensions = ['csv', 'sqlite', 'xlsx', 'xlsm', 'txt']
    if chose_data_path:
        uploader = st.file_uploader("Upload data file",
                                    type=database_extensions
                                    )
        data_file = uploader
    else:
        data_folder = os.path.join(os.getcwd(), 'data')
        try:
            all_files = [f for f in os.listdir(data_folder)
                         if os.path.isfile(os.path.join(data_folder, f))
                         and (f.endswith(tuple(database_extensions)))]
            selected_file = st.selectbox('Select data file', all_files)
            data_file = os.path.join(data_folder, selected_file)
        except IOError:
            pass


def get_data(data_container):
    """Load chosen datafile"""
    global data_file

    data = pd.DataFrame({})
    try:
        data = data_container(pd.read_csv, data_file)
        data = data.data
    except pd.errors.ParserError:
        st.text("Oooopsie! Seems like you loaded a wrong "
                "file type:\nLet's try a .csv file")
        try:
            sys.exit(1)
        except SystemExit:
            pass
    return data


def raw_data_sidebar_objects():
    """Define objects to be placed by the sidebar"""
    st.sidebar.title("Tools")
    st.sidebar.markdown("### Raw Data")
    with st.sidebar.beta_expander("Data Input"):
        select_file()


def ploting_sidebar_things():
    st.sidebar.markdown("### Ploting")


def data_cleaning_main_page():
    global cleaned_data
    global dash_data
    # Useful for data that without missing values
    cleaned_data = dash_data

    if any(list(dash_data.isnull().any())):
        data_clean_expander = st.beta_expander("Data Cleaning: Chose what to "
                                               "do with missing values.")
        with data_clean_expander:
            dash_data_bak = dash_data.copy()
            empty_cols = dash_data_bak.columns[
                dash_data_bak.isnull().any()].to_list()
            empty_cols_cleaner = {key: " " for key in empty_cols}
            data_clean_cols = st.beta_columns(len(empty_cols))
            cleaning_radio_cols = CleaningRadioOpts()

            for ind, col in enumerate(list(data_clean_cols)):
                with col:
                    setattr(
                        cleaning_radio_cols, empty_cols[ind],
                        st.radio(f"{empty_cols[ind]}",
                                 ("Drop", "Fill with"),
                                 key=f"{empty_cols[ind]}"
                                 )
                    )
                    button_opt = getattr(cleaning_radio_cols, empty_cols[ind])
                    if button_opt == "Fill with":
                        setattr(cleaning_radio_cols,
                                empty_cols[ind] + '_input',
                                st.text_input("Sub", key=empty_cols[ind]))
                        empty_cols_cleaner.update(
                            {empty_cols[ind]: getattr(
                                cleaning_radio_cols,
                                empty_cols[ind] + '_input')}
                        )
                    else:
                        setattr(cleaning_radio_cols,
                                empty_cols[ind] + '_input',
                                st.empty())
                        empty_cols_cleaner.update({empty_cols[ind]: "Drop"})
            _, status, clean_tool_col = st.beta_columns([1, 0.17, 0.15])
            with clean_tool_col:
                collect_responses_but = st.button("Clean")
                collect_responses_but = True
                if collect_responses_but:
                    kwargs = empty_cols_cleaner
                    with st.spinner("Cleaning ..."):
                        cleaned_data = clean_data(dash_data_bak, **kwargs)
                        status.markdown("Success :white_check_mark:")
            if cleaned_data.empty:
                pass
            else:
                st.markdown('#### Clean Data')
                st.dataframe(cleaned_data.head())


@st.cache
def clean_data(data, **kwargs):
    data = data.copy()
    for key, val in kwargs.items():
        if val != "Drop":
            data[key].fillna(val, inplace=True)
        elif val == "Drop":
            data.dropna(subset=[key], inplace=True)
    return data


def generate_pivot(pivot_func, data, **kwargs):
    return pivot_func(data, **kwargs)


def plot_matplotlib():
    global cleaned_data

    permited_plots = ['plot', 'bar', 'barh', 'hist', 'box', 'kide', 'density',
                      'area', 'pie', 'scatter', 'hexbin']
    st.markdown("## Analysis\n***")

    pivot_table_arglist = ['values', 'index', 'columns', 'aggfunc', 'margins']
    pivot_arglist = ['values', 'index', 'columns']
    agg_functions = ['count', 'sum', 'mean', 'median', 'most_frequent']
    pivot_table_state = PivotTableArgsState()

    with st.sidebar.beta_expander("Plot Vars"):
        # TODO: Surely there must be another way to avoid duplicating entries
        #  across columns and index variables
        use_pivot = st.checkbox("Non numeric data")
        for opt in pivot_table_arglist:
            if opt in pivot_arglist:
                label = f"Set {opt}"
                setattr(
                    pivot_table_state,
                    opt,
                    st.multiselect(
                        label, cleaned_data.columns, key=opt
                    )
                )
            if use_pivot:
                continue
            elif opt == 'aggfunc':
                # TODO: Include multiple agg funcs
                setattr(
                    pivot_table_state,
                    opt,
                    st.selectbox(
                        "Agg Func", agg_functions, key=opt
                    )
                )
            elif opt == 'margins':
                setattr(pivot_table_state,
                        opt,
                        st.checkbox("Sub/Totals", value=False))
    if not use_pivot:
        pivot_table_kwargs = {key: getattr(pivot_table_state, key)
                              for key in pivot_table_arglist}
    pivot_kwargs = {key: getattr(pivot_table_state, key)
                    for key in pivot_arglist}

    num_params = sum([len(obj) > 0 for obj in pivot_kwargs.values()])
    pivot_data = pd.DataFrame({})
    if num_params >= 2:
        with st.beta_expander("Pivot Table"):
            if use_pivot:
                pivot_data = generate_pivot(pd.pivot,
                                            cleaned_data,
                                            **pivot_kwargs)
            else:
                pivot_data = generate_pivot(pd.pivot_table, cleaned_data,
                                            margins_name="Sub/Total(s)",
                                            fill_value=0, **pivot_table_kwargs)
            if not pivot_data.empty:
                st.dataframe(pivot_data)
            else:
                pass

    plot_selector_tab = st.sidebar.beta_expander("Plot")

    # sidebar for matplotlib tools
    with plot_selector_tab:
        plot_kwargs = {}
        select_plot_type = st.selectbox("Graph Type", permited_plots)
        color_selector_col, _ = st.beta_columns([0.9, 1])
        with color_selector_col:
            plot_color = st.color_picker("Plot color")
            x_label = st.text_input("x-axis label:")
            y_label = st.text_input("y-axis label:")

        with _:
            x_axis_var = st.multiselect("x-axis var",
                                        cleaned_data.columns.to_list()
                                        )
            y_axis_var = st.multiselect("y-axis var",
                                        cleaned_data.columns.to_list()
                                        )

        grouper_list = st.multiselect('Chose Grouper',
                                      cleaned_data.columns.to_list(),
                                      default=None
                                      )
        plot_kwargs.update({'color': plot_color})
        _, calculate_but_space = st.beta_columns([1, 0.3])
        with calculate_but_space:
            calculate_plot_but = st.button("Plot")
    try:
        plot_pivot_data(pivot_data.index.get_level_values(x_axis_var[0]), pivot_data[y_axis_var[0]].values, kind=select_plot_type, **plot_kwargs)
    except IndexError:
        pass


def plot_pivot_data(x, y, kind='plot', **kwargs):
    # breakpoint()
    with plt.xkcd():
        fig, ax = plt.subplots()
        getattr(plt, kind)(x, y, **kwargs)
        plt.xticks(rotation=80)
        st.pyplot(fig)


def get_pivot_table(data, **kwargs):
    return data.pivot_table(**kwargs)


def raw_data_main_page():
    """Define objects to be defined in the main page"""
    global dash_data
    dash_data = get_data(ReadExcelDatabase)
    if not dash_data.empty:
        st.dataframe(dash_data.head())

        # some statistics
        new_df = pd.DataFrame(dash_data.isnull().sum(),
                              columns=['Missing count']
                              ).reindex()
        new_df.sort_values('Missing count', ascending=False, inplace=True)
        # breakpoint()
        if new_df['Missing count'].any():
            st.markdown("### Data Statistics\n***")
            missing_data_tab, missing_data_plot = st.beta_columns((1, 1.7))
            with missing_data_tab:
                st.dataframe(new_df)

            with missing_data_plot:
                with plt.xkcd():
                    fig, ax = plt.subplots()
                    plt.bar(new_df.index.values, new_df['Missing count'])
                    plt.xticks(rotation=80)
                    st.pyplot(fig)
        else:
            st.text("Yaaaay, you don't have missing data!")
            st.balloons() # I think these are annoying
    else:
        pass


def streamlit_main():
    """Function organisation for the App"""
    header_things()
    raw_data_sidebar_objects()
    raw_data_main_page()
    data_cleaning_main_page()
    ploting_sidebar_things()
    plot_matplotlib()


if __name__ == "__main__":
    streamlit_main()
