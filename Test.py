
from flask import Flask, render_template, request, jsonify
from dataretrieval import nwis  # For USGS data
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import time
import base64
import requests

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
    
    return df

def grab_noaa_data(site_id, start_date, end_date,retries=3, delay=5):
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
        "datatypeid": "TMAX,TMIN",       # Request both TMAX and TMIN
        "stationid": site_id,       # Specific station ID
        "startdate": start_date,
        "enddate": end_date,
        "limit": 1000                    # Max records per request
    }
    for attempt in range(retries):
        try:
            print(f"Requesting NOAA data with parameters: {params}")
            response = requests.get(url, headers=headers, params=params, timeout=30)
            print(f"Response status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    print(f"Successfully retrieved noaa data")
                    df = pd.DataFrame(data['results'])
                    df = df[['date', 'datatype', 'value']]
                    df = df.pivot(index='date', columns='datatype', values='value').reset_index()
                    df.columns = ['Date', 'TMAX', 'TMIN']
                    df['TMAX'] = df['TMAX'] / 10
                    df['TMIN'] = df['TMIN'] / 10
                    df['TAVE'] = (df['TMAX']+df['TMIN'])/2
                    df['Date'] = pd.to_datetime(df['Date'])
                    return df
                else:
                    print("No data found for this query.")
                    return pd.DataFrame()
            else:
                print(f"Error: {response.status_code}, {response.text}")
        except requests.RequestException as e:
            print(f"Error occurred during the request: {e}")
        
        # Wait before retrying
        print(f"Retrying in {delay} seconds...")
        time.sleep(delay)

    return df


def merge_data(usgs_data,noaa_data):

    merged_df = pd.merge(usgs_data, noaa_data, on='Date', how='inner')
    return(merged_df)


def create_plot(data):
    # Create a figure with two subplots (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot USGS discharge data on the first subplot (ax1)
    ax1.plot(data['Date'], data['00060_Mean'], label='USGS Discharge (cfs)', color='blue', linestyle='--')
    ax1.set_ylabel('Discharge (cfs)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('USGS Discharge Data')
    ax1.grid()

    # Plot NOAA temperature data on the second subplot (ax2)
    ax2.plot(data['Date'], data['TMAX'], color='gray',linestyle='--', label='_nolegend_')
    ax2.plot(data['Date'], data['TMIN'], color='gray',linestyle='--', label='_nolegend_')

    # Fill the area between TMIN and TMAX
    ax2.fill_between(data['Date'], data['TMIN'], data['TMAX'], color='gray', alpha=0.3, label='Temperature Range (TMIN to TMAX)')

    # Plot TAVE on the same subplot
    ax2.plot(data['Date'], data['TAVE'], label='NOAA TAVE (°C)', color='black')

    ax2.set_ylabel('Temperature (°C)', color='red')
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

        if merged_df is not None and not merged_df.empty:
            print("Merging was successful!")
            print(merged_df.head())

            # Create the plot
            plot_url = create_plot(merged_df)
            return render_template('index.html', plot_url=plot_url, location=location, preset_locations=PRESET_LOCATIONS)

        else:
            print("Merging failed. merged_df is None or empty.")
            error_message = 'No data found for the specified parameters.'
            return render_template('index.html', error=error_message, preset_locations=PRESET_LOCATIONS)

    # For GET request, render the page directly
    return render_template('index.html', preset_locations=PRESET_LOCATIONS)

if __name__ == '__main__':
    app.run(debug=True)