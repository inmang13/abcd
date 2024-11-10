
from flask import Flask, render_template, request, jsonify
from dataretrieval import nwis  # For USGS data
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import time
import base64
import requests
from math import cos, sin, acos, pi, tan

app = Flask(__name__)

# Define the preset locations and their associated site IDs
PRESET_LOCATIONS = {
    "Bridgewater, MA": {
        "usgs_site_id": "01108000",  # Example USGS site ID
        "noaa_site_id": "GHCND:USW00054777"  # Example NOAA site ID (Central Park)
    },
    "Boston, MA": {
        "usgs_site_id": "USGS:01108001",  # Example USGS site ID for Boston
        "noaa_site_id": "GHCND:US1BOST001"  # Example NOAA site ID for Boston
    },
    "Providence, RI": {
        "usgs_site_id": "USGS:01108002",  # Example USGS site ID for Providence
        "noaa_site_id": "GHCND:US1PROV001"  # Example NOAA site ID for Providence
    }
}

def grab_usgs_data(site_no, start_date, end_date):
    """
    Retrieve flow data from USGS.
    :param site_no: USGS site number
    :param start_date: Start date
    :param end_date: End date
    :return: USGS DataFrame
    """
    print(f"Retrieving USGS data for USGS: {site_no} from {start_date} to {end_date}")
    
    try:
        df = nwis.get_dv(
            sites=site_no,
            parameterCd="00060",
            statCd="00003",
            start=start_date,
            end=end_date,
        )[0].reset_index()
        
        # Rename the datetime column to 'Date' and remove timezone
        df = df.rename(columns={'datetime': 'Date'})
        df['Date'] = df['Date'].dt.tz_convert(None)

        # Check if the DataFrame has data
        if df.empty:
            print("No data found for the given parameters.")
        else:
            print(f"Successfully retrieved {len(df)} rows of data.")
    
    except Exception as e:
        print(f"Error retrieving USGS data: {e}")
        df = pd.DataFrame()  # Return an empty DataFrame in case of error
    
    print(df.head())

    return df

def grab_noaa_data(site_id, start_date, end_date):
    """
    Retrieve climate data from NOAA.
    :param site_id: NOAA site ID
    :param start_date: Start date
    :param end_date: End date
    :return: NOAA Data
    """
    print(f"Retrieving NOAA data for {site_id} from {start_date} to {end_date}")
    # Use the NOAA Climate Data API (example URL)
    NOAA_API_TOKEN="lqbBOsoeMwxZbPbaOsZpjdSODFPAqQGG"

    url = f"https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    headers = {"token": NOAA_API_TOKEN}  # Replace with your NOAA API token
    params = {
        "datasetid": "GHCND",            # Global Historical Climatology Network - Daily
        "datatypeid": "TMAX,TMIN,PRCP",       # Request both TMAX and TMIN
        "stationid": site_id,       # Specific station ID
        "startdate": start_date,
        "enddate": end_date,
        "limit": 1000                    # Max records per request
    }
    print(f"Requesting NOAA data with parameters: {params}")
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
    else:
        print("Error:", response.status_code, response.text)

    df = pd.DataFrame(data['results'])
    if not df.empty:
        df = df[['date', 'datatype', 'value']]
        df = df.pivot(index='date', columns='datatype', values='value').reset_index()
        df.columns = ['Date', 'PRCP', 'TMAX','TMIN']
        df['TMAX_F'] = (df['TMAX'] / 10)*1.8+32
        df['TMIN_F'] = (df['TMIN'] / 10)*1.8+32
        df['TMAX'] = df['TMAX'] / 10
        df['TMIN'] = df['TMIN'] / 10
        df['PRCP'] = df['PRCP'] / 10 *0.03937008 # Convert precipitation to millimeters
        df['Date'] = pd.to_datetime(df['Date'])
        df['TAVE'] = (df['TMAX']+df['TMIN'])/2
        df['TAVE_F'] = (df['TMAX_F']+df['TMIN_F'])/2
    else:
        print('grab_noaa_data df is empty')
    return df


def merge_data(usgs_data, noaa_data):
    # Merge the USGS and NOAA data on 'Date'
    merged_df = pd.merge(usgs_data, noaa_data, on='Date', how='inner')
    merged_df['DOY'] = merged_df['Date'].dt.dayofyear
    return(merged_df)

def model(data, a, b, c, d, e, Tb, lat, S_init, G_init, Area):
    Tave = data['TAVE']
    Tmax = data['TMAX']
    Tmin = data['TMIN']
    Precip = data['PRCP']

    # Day Length Factor (dr)
    data['dr'] = 1 + 0.033 * np.cos(2 * np.pi * data['DOY'] / 365)

    # Solar Declination Angle (del)
    data['del'] = 0.4093 * np.sin((2 * np.pi * data['DOY'] / 365) - 1.405)

    # Solar Hour Angle (ws) in radians
    data['ws'] = np.arccos(-1 * np.tan(data['del']) * np.tan(lat))

    # Hours of sunshine (hrs)
    data['hrs'] = 24 / np.pi * data['ws']

    # Solar Radiation (Sol, in mm/day)
    data['Sol'] = 15.392 * data['dr'] * (
        data['ws'] * np.sin(lat) * np.sin(data['del']) +
        np.cos(lat) * np.cos(data['del']) * np.sin(data['ws'])
    )

    # Potential Evapotranspiration (PEt, in mm/day)
    data['PEt'] = 0.0023 * data['Sol'] * (Tave + 17.8) * np.sqrt(Tmax - Tmin)

    # Actual Evapotranspiration (PE, converted to inches per month)
    data['PE'] = np.where(Tmin < Tb, 0, data['PEt'] / 25.4 * 30.4)

    # Initialize arrays for snow accumulation, snow melt, available water, evapotranspiration opportunity,
    # soil moisture, and groundwater storage
    Snow_acc = np.zeros(len(Tave))
    Snow_melt = np.zeros(len(Tave))
    W = np.zeros(len(Tave))
    Y = np.zeros(len(Tave))
    S = np.zeros(len(Tave))
    GW = np.zeros(len(Tave))

    # Loop through each time step
    for t in range(len(Tave)):
        # Snow model calculations (as implemented before)
        if t == 0:
            Snow_acc[t] = 0
            Snow_melt[t] = min(Snow_acc[t], e * abs(Tave[t] - Tb)) if Tave[t] > Tb else 0
            W[t] = Precip[t] + S_init
            GW[t] = G_init
        elif t < len(Tave) - 1:
            Snow_acc[t] = (Snow_acc[t - 1] + Precip.iloc[t + 1] - Snow_melt[t - 1] 
                           if Tave.iloc[t + 1] < Tb 
                           else Snow_acc[t - 1] - Snow_melt[t - 1])
            Snow_melt[t] = min(Snow_acc[t], e * abs(Tave[t] - Tb)) if Tave[t] > Tb else 0
            W[t] = Precip[t] + S[t - 1]
        else:
            Snow_acc[t] = Snow_acc[t - 1] - Snow_melt[t - 1]
            Snow_melt[t] = min(Snow_acc[t], e * abs(Tave[t] - Tb)) if Tave[t] > Tb else 0
            W[t] = Precip[t] + S[t - 1]

        # Calculate intermediate values w1 and w2
        w1 = (W[t] + b) / (2 * a)
        w2 = W[t] * b / a

        # Calculate Y (evapotranspiration opportunity)
        Y[t] = w1 - np.sqrt((w1 ** 2) - w2)

        # Calculate S (soil moisture)
        S[t] = Y[t] * np.exp(-1 * data['PE'].iloc[t] / b)

        # Calculate GW (groundwater storage)
        if t > 0:
            GW[t] = (c * (W[t] - Y[t]) + GW[t - 1]) / (1 + d)

    # Add calculated columns to the DataFrame
    data['W'] = W
    data['Y'] = Y
    data['S'] = S
    data['GW'] = GW

    # Runoff and Baseflow calculations
    data['Runoff'] = ((1 - c) * (W - Y)) / 12 * Area * 43560 * 7.48 / 1000000  # MG
    data['Baseflow'] = (GW * d) / 12 * Area * 43560 * 7.48 / 1000000  # MG
    data['SimFlow_MG'] = data['Runoff'] + data['Baseflow']
    data['SimFlow_CFS'] = data['SimFlow_MG'] / 0.646 / 30.4  # CFS
    print(data.head(3))
    return data


def climate_plot(data):
    # Create a figure with two subplots (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot USGS discharge data on the first subplot (ax1)
    ax1.plot(data['Date'], data['00060_Mean'], label='USGS Discharge (cfs)', color='blue', linestyle='--')
    ax1.set_ylabel('Discharge (cfs)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('USGS Discharge Data')
    ax1.grid()

    # Create a second y-axis for precipitation
    ax1_precip = ax1.twinx()
    ax1_precip.bar(data['Date'], data['PRCP'], color='green', alpha=0.5, label='Precipitation (inches)')
    ax1_precip.set_ylabel('Precipitation (inches)', color='green')
    ax1_precip.tick_params(axis='y', labelcolor='green')

    # Reverse the precipitation axis
    ax1_precip.invert_yaxis()

    # Add legends for discharge and precipitation
    ax1.legend(loc='upper left')
    ax1_precip.legend(loc='upper right')

    # Plot NOAA temperature data on the second subplot (ax2)
    ax2.plot(data['Date'], data['TMAX_F'], color='gray',linestyle='--', label='_nolegend_')
    ax2.plot(data['Date'], data['TMIN_F'], color='gray',linestyle='--', label='_nolegend_')

    # Fill the area between TMIN and TMAX
    ax2.fill_between(data['Date'], data['TMIN_F'], data['TMAX_F'], color='gray', alpha=0.3, label='Temperature Range (TMIN to TMAX)')

    # Plot TAVE on the same subplot
    ax2.plot(data['Date'], data['TAVE_F'], label='NOAA TAVE (°F)', color='black')

    ax2.set_ylabel('Temperature (°F)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_title('NOAA Temperature Data')

    # Add legends for both subplots
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')

    # Set the x-axis label
    ax2.set_xlabel('Date')

    # Rotate the x-axis labels for better visibility
    plt.xticks(rotation=45)

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf8')

def calibration_plot(data):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual discharge data
    ax.plot(data['Date'], data['00060_Mean'], label='Observed', color='blue', linewidth=1.5)

    # Plot simulated flow data
    ax.plot(data['Date'], data['SimFlow_CFS'], label='Simulated', color='red', linestyle='--', linewidth=1.5)

    # Set plot title and labels
    ax.set_title('Calibration Plot: Observed vs Simulated Flow', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Flow (cfs)', fontsize=14)

    # Format the x-axis for better readability

    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Add legend to differentiate between observed and simulated data
    ax.legend(loc='upper left', fontsize=12)

    # Adjust layout for better spacing
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf8')

def run_scenario_analysis(a, b, c, d):
    print(a,b,c,d)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        location = request.form['location']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        # Check if the location is in the preset locations
        if location in PRESET_LOCATIONS:
            usgs_site_id = PRESET_LOCATIONS[location]['usgs_site_id']
            noaa_site_id = PRESET_LOCATIONS[location]['noaa_site_id']
        else:
            # If no location is selected or not in the preset list, show an error
            return render_template('index.html', error="Please select a valid location.", preset_locations=PRESET_LOCATIONS)

        # Fetch data from USGS and NOAA using the preset site IDs
        usgs_data = grab_usgs_data(usgs_site_id, start_date, end_date)
        noaa_data = grab_noaa_data(noaa_site_id, start_date, end_date)

        # Merge the data from USGS and NOAA
        merged_df = merge_data(usgs_data, noaa_data)

        a = request.form.get('a', type=float, default=0.98)
        b = request.form.get('b', type=float, default=5.05)
        c = request.form.get('c', type=float, default=0.71)
        d = request.form.get('d', type=float, default=0.9)
        e = request.form.get('e', type=float, default=50)
        Tb = request.form.get('Tb', type=float, default=-4.81)
        lat = request.form.get('lat', type=float, default=0.732)
        S_init = request.form.get('S_init', type=float, default=10)
        G_init = request.form.get('G_init', type=float, default=2)
        Area = request.form.get('Area', type=float, default=167040)

        if merged_df is not None and not merged_df.empty:
            print("Merging was successful!")
            print(merged_df.head())

            # Create the plot
            plot_url = climate_plot(merged_df)

            model_result = model(merged_df, a, b, c, d, e, Tb, lat, S_init, G_init, Area)

            return render_template(
                'index.html', 
                plot_url=plot_url, 
                model_result=model_result,
                location=location,
                start_date=start_date,
                end_date=end_date,
                a=a, b=b, c=c, d=d, e=e, Tb=Tb, lat=lat, 
                S_init=S_init, G_init=G_init, Area=Area,
                preset_locations=PRESET_LOCATIONS
            )

        else:
            print("Merging failed. merged_df is None or empty.")
            error_message = 'No data found for the specified parameters.'
            return render_template('index.html', error=error_message, preset_locations=PRESET_LOCATIONS)
            

    # For GET request, render the page directly
    return render_template('index.html', preset_locations=PRESET_LOCATIONS)

if __name__ == '__main__':
    app.run(debug=True)