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
This will start the Streamlit application and open it in your default web browse.
![1](https://github.com/user-attachments/assets/42fafa66-3963-4d44-92fa-94694a8480fb)
#![2](https://github.com/user-attachments/assets/ccd67b48-80ac-4d65-af12-417c91d1ec1b)
Sometimes the program gets stuck retrieving data from NOAA's API. If it is taking more than a minute to load, please refresh the page.

## Calibrate Model
Once climate data has been retrieved. Navigate to the Calibrate Model Tab.
![3](https://github.com/user-attachments/assets/805e7229-9e34-47d4-8407-6e3d52ff24bd)
![4](https://github.com/user-attachments/assets/b9052c48-2e6b-46e5-a229-1f70bc13d637)
![5](https://github.com/user-attachments/assets/a484ca35-3dd6-42d0-abcb-c2d0e448cbb9)


