# ABCD Model Calibration App

## Description
The **ABCD Model Calibration App** is a web application designed to assist in the calibration of a simple hydrological model (ABCD model) using real-world climate and streamflow data. This app allows users to retrieve climate data (precipitation, temperature, etc.) from NOAA, along with observed streamflow data from USGS, and calibrate the model based on these inputs. Users can also explore future climate and hydrological scenarios based on the calibrated model.

## Features
- **Data Retrieval**: Fetch climate data (precipitation, temperature) and streamflow data for selected locations and periods.
- **Model Calibration**: Calibrate the ABCD model using observed data to minimize the Root Mean Square Error (RMSE).
- **Scenarios**: Explore the effects of future climate scenarios on hydrological behavior using the calibrated model.
- **Interactive Visualizations**: Interactive plots of observed vs simulated data and scenario results.

## Installation

### 1. Clone the repository
```
git clone https://github.com/your-username/abcd-model-calibration.git
```
### 2. Create a Conda environment (optional but recommended)
```
conda create --name ENV python=3.8
```
### 3. Activate the environment
```
conda activate ENV
```
### 4. Install the required dependencies
```
pip install -r requirements.txt
```
## Installation
### Running the App Locally
To start the app, run the following command in your terminal:
```
streamlit run app.py
```
This will start the Streamlit application and open it in your default web browser.
