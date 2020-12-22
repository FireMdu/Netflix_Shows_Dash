## Project Description
-----------------------------------
This project is a demonstration of building an **E**xploratory **D**ata **A**nalysis dashboard using the Python library ``streamlit``. The repository is currently set up in a way that permits different dashboards to be rendered from different database. In the sprit of creating a general tool, there might be some negative effects on the functionality.

### Objectives:
The main objectives of the project are to:  
- [ ] Build an EDA dashboard deriving insight on data.
- [ ] Build a model to predict the Movie/TV Show genre and incoporate the dashboard

### Setup
-------------
A `requirements.txt` file is included with this project; it contains all the required packages to succesfully run the project. If you have setup a Python virtual environment for the project(highly recommended) you can run: 
```Shell
pip install -r requirements.txt
```
from your favourite `Shell`.  
If you do not know how to setup `pip` or setting up the `virtual environment`, see the [pip installation](https://pip.pypa.io/en/stable/installing/) and the [setting up a virtual environment](https://docs.python.org/3/tutorial/venv.html) articles respectively.   	

#### Running Streamlit Scripts
When running streamlit scripts, the project presents two options:
1. Render the App to html for a single data source
```Shell
streamlit run .\streamlit_scripts\<filename_here.py> [--script_args]
```
2. Render the App to html for all data sources
```Shell
streamlit run main.py [--script_args]
```
The second option will present an option on the tools window to render all the scripts and present you with an option to individually select the analysis based on the input data. __Note:__ One can not show all dashboards for different data sources at once but can only do so for individual sources.

### Data and Storage
----------------------------
The raw data for the project is found under the `.\data` directory. Data files can be large in size resulting in a lsow process to clone the repository. Instead a `data_links.txt` file is located under the `.\data` directory with all the names and links to the datasets applicable in this project. This list will be kept updated as the project gets bigger and is also presented below for quick reference.

- [European Soccer Database](https://www.kaggle.com/hugomathien/soccer)
- [Netflix Movies and TV Shows](https://www.kaggle.com/shivamb/netflix-shows)

### Data Cleaning 
-----------------------

### Modeling
-----------------

### Analytics
-----------------