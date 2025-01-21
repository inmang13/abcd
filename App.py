import streamlit as st
from dataretrieval import nwis  # For USGS data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import pi
from datetime import datetime, timedelta, date
from scipy.optimize import minimize


Location_dict = {
    "Bridgewater, MA": {
        "usgs_site_id": "01108000",  # Example USGS site ID
        "noaa_site_id": "GHCND:USW00054777"  # Example NOAA site ID (Central Park)
    }
}

def grab_usgs_data(site_no, start_date, end_date):
    """
    Retrieve daily average flow data from USGS.
    :param site_no: USGS site number
    :param start_date: Start date
    :param end_date: End date
    :return: USGS DataFrame with monthly Ave
    """
    print(f"Retrieving USGS data for {site_no} from {start_date} to {end_date}")
    df = nwis.get_dv(
        sites=site_no,
        parameterCd="00060",
        statCd="00003",
        start=start_date,
        end=end_date,
    )[0].reset_index()

    site_info = nwis.get_record(site_no,service='site')
    area=site_info.loc[0,'drain_area_va']*640
    lat=site_info.loc[0,'dec_lat_va']
    long=site_info.loc[0,'dec_long_va']
    lat_radians= lat*(pi / 180)
    print(lat,long)

    df = df.rename(columns={'datetime': 'Date'})
    df = df.rename(columns={'datetime': 'Date', '00060_Mean': 'Flow'})
    df['Date'] = df['Date'].dt.tz_convert(None)
    df = df[['Date', 'Flow']] 
    df['Flow'] = df['Flow'].astype(float) 
    df.set_index('Date', inplace=True)
    monthly_avg = df.resample('M').mean().reset_index()
    monthly_avg['Date'] = monthly_avg['Date'].apply(lambda x: x.replace(day=1))
    return monthly_avg, lat, long, lat_radians, area

def grab_noaa_data(site_id, start_date, end_date):
    """
    Retrieve climate data from NOAA for multiple years.
    :param site_id: NOAA site ID
    :param start_date: Start date (YYYY-MM-DD)
    :param end_date: End date (YYYY-MM-DD)
    :return: DataFrame containing monthly Precipitation and Temperature data
    """
    print(f"Retrieving NOAA data for {site_id} from {start_date} to {end_date}")
    
    NOAA_API_TOKEN = "lqbBOsoeMwxZbPbaOsZpjdSODFPAqQGG"
    url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    headers = {"token": NOAA_API_TOKEN}
    all_data = []
    
    # Convert start_date and end_date to datetime objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Loop through each year and make requests
    current_start = start_date
    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=364), end_date)  # NOAA API has a limit of 1 year per request
        
        params = {
            "datasetid": "GSOM",  # Global Historical Climatology Network - Daily
            "datatypeid": "TMAX,TMIN,PRCP",  # Request TMAX, TMIN, PRCP
            "stationid": site_id,
            "startdate": current_start.strftime("%Y-%m-%d"),
            "enddate": current_end.strftime("%Y-%m-%d"),
            "limit": 1000  # Maximum records per request
        }
        
        print(f"Requesting data from {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}...")
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if "results" in data:
                all_data.extend(data["results"])
        else:
            print(f"Error {response.status_code}: {response.text}")
        
        # Move to the next year
        current_start = current_end + timedelta(days=1)
    
    # Convert the collected data into a DataFrame
    if all_data:
        df = pd.DataFrame(all_data)
        df = df[['date', 'datatype', 'value']]
        df = df.pivot(index='date', columns='datatype', values='value').reset_index()
        df.columns = ['Date', 'PRCP', 'TMAX', 'TMIN']
        df['TMAX'] = df['TMAX'].interpolate(method='linear') # change to sin interpolate at some point
        df['TMIN'] = df['TMIN'].interpolate(method='linear')
        df['PRCP'] = df['PRCP'].fillna(0)
        df['TMAX_F'] = (df['TMAX']) * 1.8 + 32  # Convert Celsius to Fahrenheit
        df['TMIN_F'] = (df['TMIN']) * 1.8 + 32  # Convert Celsius to Fahrenheit
        df['PRCP'] = df['PRCP'] * 0.03937008  # Convert precipitation to inches
        df['Date'] = pd.to_datetime(df['Date'])
        df['TAVE_F'] = (df['TMAX_F'] + df['TMIN_F']) / 2
        df['TAVE'] = (df['TMAX'] + df['TMIN']) / 2
    else:
        print("No data retrieved.")
        df = pd.DataFrame()  # Return empty DataFrame if no data
    
    return df


def merge_data(usgs_data, noaa_data):
    # Merge the USGS and NOAA data on 'Date'
    merged_df = pd.merge(usgs_data, noaa_data, on='Date', how='inner')
    merged_df['DOY'] = merged_df['Date'].dt.dayofyear
    return(merged_df)

def climate_plot(data):
    # Create a figure with two subplots (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot USGS discharge data on the first subplot (ax1)
    ax1.plot(data['Date'], data['Flow'], label='USGS Discharge (cfs)', color='blue', linestyle='--')
    ax1.set_ylabel('Discharge (cfs)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('USGS Discharge Data')
    ax1.grid()

    # Create a second y-axis for precipitation
    ax1_precip = ax1.twinx()
    ax1_precip.bar(data['Date'], data['PRCP'], color='green', alpha=0.5, label='Precipitation (inch)')
    ax1_precip.set_ylabel('Precipitation (inch)', color='green')
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
  
    return fig

    
def model(data, a, b, c, d, e, Tb, lat, S_init, G_init, Area):
    data=data.dropna()
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
    data['PE'] = (np.where(Tmin < Tb, 0, data['PEt'] / 25.4)) * 30.4

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
    return data


def calibration(data, a, b, c, d, e, Tb, lat, S_init, G_init, Area):
    # Define the objective function to minimize (RMSE)
    def objective(params):
        a, b, c, d, e = params
        model_result = model(data, a, b, c, d, e, Tb, lat, S_init, G_init, Area)
        obs = model_result["Flow"]
        sim = model_result["SimFlow_CFS"]
        rmse = np.sqrt(np.mean((obs - sim) ** 2))
        return rmse

    # Calculate original RMSE before minimizing
    model_result = model(data, a, b, c, d, e, Tb, lat, S_init, G_init, Area)
    obs = model_result["Flow"]
    sim = model_result["SimFlow_CFS"]
    original_rmse = np.sqrt(np.mean((obs - sim) ** 2))
    print("Original RMSE:", original_rmse)


    # Initial guess for the parameters
    initial_guess = [a, b, c, d, e]

    # Define the bounds for each parameter
    bounds = [
        (1e-6, 1),        # a_lims
        (10, 50),      # b_lims
        (0.10, 0.90),  # c_lims
        (0.1, 0.90),   # d_lims
        (0, 75)        # e_lims
    ]

    # Perform the optimization to minimize RMSE
    result = minimize(objective, initial_guess, bounds=bounds)

    # Check if the optimization was successful
    if result.success:
        optimized_params = result.x
        print("Optimization successful.")
        print("Optimized parameters:", optimized_params)
        print("Minimum RMSE:", result.fun)
    else:
        print("Optimization failed.")
        optimized_params = None

    return optimized_params
        


#raw = merge_data(usgs_data, noaa_data)
st.set_page_config(page_title="ABCD MODEL", layout="wide")
a_init = 0.984
b_init = 5.05
c_init = 0.710
d_init = 0.9
e_init= 50
Tb_init = -4.81
G_init = 2
S_init = 10

st.markdown(
    """
    <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px; color: navy;">
        🌟 ABCD Water Balance Model Calibration App 🌟
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    '''
    <style>
        /* Style for the first tab */
        .stTabs [data-baseweb="tab-list"] button:nth-child(1) {
            background-color: #808080;  /* Grey background */
            color: white;               /* White text */
            font-size: 36px;            /* Larger font size */
            padding: 5px 10px;          /* Add some padding */
            border-radius: 6px;         /* Round corners */
            border: 2px solid #555555; /* Dark grey border */
        }

        /* Style for the second tab */
        .stTabs [data-baseweb="tab-list"] button:nth-child(2) {
            background-color: #4CAF50;  /* Green background */
            color: white;               /* White text */
            font-size: 36px;            /* Larger font size */
            padding: 5px 10px;          /* Add some padding */
            border-radius: 6px;         /* Round corners */
            border: 2px solid #388E3C;  /* Dark green border */
        }

        /* Style for the third tab */
        .stTabs [data-baseweb="tab-list"] button:nth-child(3) {
            background-color: #2196F3;  /* Blue background */
            color: white;               /* White text */
            font-size: 36px;            /* Larger font size */
            padding: 5px 10px;          /* Add some padding */
            border-radius: 6px;         /* Round corners */
            border: 2px solid #1976D2;  /* Dark blue border */
        }

        /* Style for the fourth tab */
        .stTabs [data-baseweb="tab-list"] button:nth-child(4) {
            background-color: #FF9800;  /* Orange background */
            color: white;               /* White text */
            font-size: 36px;            /* Larger font size */
            padding: 5px 10px;          /* Add some padding */
            border-radius: 6px;         /* Round corners */
            border: 2px solid #F57C00;  /* Dark orange border */
        }
    </style>
    ''', 
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs(['Retrieve Climate Data','Model Calibration','Scenarios','Documentation'])  

with tab1:
    st.title("🔍 Retrieve Climate Data")
    st.write("Select a location and time period to calibrate your model. This app will automatically retrieve discharge data from a nearby USGS gauge, along with precipitation and temperature data from NOAA for the selected region.")
    form = st.form(key='Retrieve')
    location = form.selectbox("Pick a Location",("Bridgewater, MA", "Example 2", "Example 3"))

    usgs_site_id = Location_dict[location]['usgs_site_id']
    noaa_site_id = Location_dict[location]['noaa_site_id']


    start_date = form.date_input("Start Date", date(2010, 1, 1)).strftime("%Y-%m-%d")
    end_date = form.date_input("End Date", date(2020, 1, 1)).strftime("%Y-%m-%d")
    submit = form.form_submit_button('Retrieve Data')

    if submit:
        with st.spinner('Processing...'):
            usgs_data ,lat, long, lat_radians,area = grab_usgs_data(usgs_site_id, start_date, end_date)
            noaa_data = grab_noaa_data(noaa_site_id, start_date, end_date)

            st.write("**USGS Site ID:** USGS", usgs_site_id)
            st.write(f"**USGS Site Coordinates:** ({lat}, {long})")
            st.write(f"**USGS Site Drainage Area:** {area} acres")
            st.write("**NOAA Site ID:**  " ,noaa_site_id)

            raw = merge_data(usgs_data, noaa_data)
            st.session_state.raw_data = raw
            st.session_state.location = location
            st.session_state.lat_radians = lat_radians
            st.session_state.area = area

            fig1 = go.Figure()

            fig1.add_trace(go.Scatter(x=raw['Date'], y=raw['Flow'], mode='lines', name='USGS Discharge (cfs)', line=dict(color='blue')))
            fig1.add_trace(go.Bar(x=raw['Date'], y=raw['PRCP'], name='Precipitation (inch)', 
                                marker=dict(color='green', opacity=0.5), 
                                yaxis="y2"))  # Assign to secondary Y-axis

            fig1.update_layout(
                title="Precipitation and Discharge",
                xaxis_title="Date",
                yaxis_title="Discharge (cfs)",
                yaxis2=dict(
                    title="Precipitation (inch)",
                    overlaying='y',  # This makes the secondary Y-axis overlay the primary one
                    side='right',    # Place it on the right side
                    showgrid=False,  # Optional: hides the grid lines for the second Y-axis
                    autorange='reversed'
                ),
                showlegend=True,
                legend=dict(
                    orientation='h',  # Horizontal legend
                    y=-0.2,           # Position it below the plot
                    x=0.5,            # Center the legend horizontally
                    xanchor='center', # Align the legend to the center horizontally
                    yanchor='top'     # Align the legend to the top vertically
                )
            )

            st.plotly_chart(fig1)

            fig2 = go.Figure()

            fig2.add_trace(go.Scatter(x=raw['Date'], y=raw['TMAX_F'], mode='lines', name='TMAX', line=dict(color='red',dash='dash')))
            fig2.add_trace(go.Scatter(x=raw['Date'], y=raw['TMIN_F'], mode='lines', name='TMIN', line=dict(color='blue',dash='dash')))

            # Fill the area between TMAX and TMIN
            fig2.add_trace(go.Scatter(x=raw['Date'], y=raw['TMAX_F'], mode='lines', name='Temperature Range (TMIN to TMAX)', 
                                    fill='tonexty', fillcolor='rgba(169,169,169,0.3)', line=dict(color='red',dash='dash')))

            # Plot TAVE on the same subplot
            fig2.add_trace(go.Scatter(x=raw['Date'], y=raw['TAVE_F'], mode='lines', name='TAVE (°F)', line=dict(color='black')))

            # Update layout for the second plot
            fig2.update_layout(
                title="Temperature Data",
                xaxis_title="Date",
                yaxis_title="Temperature (°F)",
                showlegend=True,
                legend=dict(
                    orientation='h',  # Horizontal legend
                    y=-0.2,           # Position it below the plot
                    x=0.5,            # Center the legend horizontally
                    xanchor='center', # Align the legend to the center horizontally
                    yanchor='top'     # Align the legend to the top vertically
                )
            )

            # Show the second plot in Streamlit
            st.plotly_chart(fig2)

with tab2:
    st.title("🔧 Model Calibration 🧰")

    st.write("""
        🛠️ **Step 1**: Adjust the initial parameters with the sliders.  
        📊 **Step 2**: Click **Calibrate** to run the model and refine the parameters.  
        🚀 **Step 3**: Watch the sliders adjust as the model fine-tunes your settings!
    """)
    st.write("""
        There are four parameters governing the model behavior:

        - **a** controls the amount of runoff and recharge that occurs when the soils are under-saturated.
        - **b** controls the saturation level of the soils.
        - **c** defines the ratio of groundwater recharge to surface runoff.
        - **d** controls the rate of groundwater discharge.
        """)
    if 'raw_data' not in st.session_state: 
        st.write('Missing Climate Data')
    else:
        raw = st.session_state.raw_data  # Retrieving the raw data from session state
        location = st.session_state.location
        lat = st.session_state.lat_radians
        area = st.session_state.area
        st.write(f"**Location:** {location}")
        st.write(f"**Area:** {area}")
        st.write(f"**Lat (rads)**: {lat}")

        abcd_model = raw.copy()  # Make a copy of the raw data for model processing

        col1, col2 = st.columns(2)

        with col1:
            # Ensure session_state values are float for sliders
            a = st.slider("a:", min_value=0.0, max_value=1.0, value=float(a_init), step=0.0001,key='a_slider')
            b = st.slider("b:", min_value=1.0, max_value=50.0, value=float(b_init), step=0.0001,key='b_slider')
            c = st.slider("c:", min_value=0.1, max_value=0.9, value=float(c_init), step=0.0001,key='c_slider')
            d = st.slider("d:", min_value=0.1, max_value=0.9, value=float(d_init), step=0.0001,key='d_slider')

        with col2:
            # For 'e' slider, ensure it's a float if required
            e = st.slider("e:", min_value=0.0, max_value=75.0, value=float(e_init), step=0.0001,key='e_slider')
            Tb = st.slider("Tb:", min_value=-5.0, max_value=5.0, value=float(Tb_init), step=0.0001,key='Tb_slider')
            G_init = st.slider("G_init:", min_value=1.0, max_value=2.0, value=float(G_init), step=0.0001,key='g_slider')
            S_init = st.slider("S_init:", min_value=1, max_value=10, value=int(S_init), step=1,key='s_slider')

        # Running the model with the parameters from session_state
        model1 = model(abcd_model, a, b, c, d, e, Tb, lat, S_init,G_init, area)
        st.write(model1.head())
        rmse = np.sqrt(np.mean((model1['Flow'] - model1['SimFlow_CFS'])**2))

        sorted_true_values = np.sort(model1['Flow'])
        sorted_predicted_values = np.sort(model1['SimFlow_CFS'])
        cdf_true_values = np.arange(1, len(sorted_true_values) + 1) / len(sorted_true_values)
        cdf_predicted_values = np.arange(1, len(sorted_predicted_values) + 1) / len(sorted_predicted_values)




        # Create a plot comparing observed and simulated flow
        Flow_TS = go.Figure()

        # Plot observed discharge data
        Flow_TS.add_trace(go.Scatter(x=model1['Date'], 
                                y=model1['Flow'], 
                                mode='lines', 
                                name='Observed', 
                                line=dict(color='blue', width=2)))

        # Plot simulated flow data
        Flow_TS.add_trace(go.Scatter(x=model1['Date'], 
                                y=model1['SimFlow_CFS'], 
                                mode='lines', 
                                name='Simulated', 
                                line=dict(color='red', dash='dash', width=2)))

        # Update layout to set title, axis labels, and grid
        Flow_TS.update_layout(
            title='Calibration Plot: Observed vs Simulated Flow',
            xaxis_title='Date',
            yaxis_title='Flow (cfs)',
            template='plotly_white',  # Light theme
            xaxis=dict(
                showgrid=True, 
                gridcolor='lightgray', 
                tickformat="%Y-%m",  # Format x-axis as year-month
            ),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
            legend=dict(x=0, y=1, traceorder='normal', font=dict(size=12)),
            plot_bgcolor='white',  # Background color
        )

        # Display the plot in Streamlit
        st.plotly_chart(Flow_TS, use_container_width=True)

        calibration_plots = make_subplots(rows=1, cols=2,
        column_widths=[0.5, 0.5]  # Give more space to the calibration plot
    )

        calibration_plots.add_trace(go.Scatter(
            x=model1['Flow'], 
            y=model1['SimFlow_CFS'], 
            mode='markers',
            marker=dict(color='rgba(0, 0, 0, 0.5)', size=8),
            name='Calibration Data',
            showlegend=False,
            text=f'RMSE: {rmse:.2f}',
            textposition="top right"), row=1, col=1)

        calibration_plots.add_trace(go.Scatter(
            x=[min(model1['Flow']), max(model1['Flow'])],
            y=[min(model1['Flow']), max(model1['Flow'])],
            mode='lines',
            showlegend=False,
            line=dict(color='red', dash='dash')), row=1, col=1)

        calibration_plots.add_annotation(
            text=f'RMSE: {rmse:.2f}',
            x=0.95,  # Right side of the plot
            y=0.05,  # Top of the plot
            xref='x domain',
            yref='y domain',
            showarrow=False,
            font=dict(size=14, color="black"),
            align="center", row=1, col=1)

        calibration_plots.update_xaxes(title_text="OBSERVED",row=1, col=1)
        calibration_plots.update_yaxes(title_text="MODELED",row=1, col=1)

        calibration_plots.add_trace(go.Scatter(
            y=sorted_true_values, 
            x=cdf_true_values, 
            mode='lines', 
            line=dict(color='black', width=2),
            name='OBSERVED'), row=1, col=2)

        calibration_plots.add_trace(go.Scatter(
            y=sorted_predicted_values, 
            x=cdf_predicted_values, 
            mode='lines', 
            line=dict(color='red', width=2),
            name='MODELED'), row=1, col=2)

        calibration_plots.update_xaxes(title_text="Cumulative Frequency",row=1, col=2)
        calibration_plots.update_yaxes(title_text="Flow",row=1, col=2)
   

        st.plotly_chart(calibration_plots)

        def action():
            [a_updated, b_updated, c_updated, d_updated, e_updated] = calibration(abcd_model, a, b, c, d, e, Tb, lat, S_init, G_init, area)
            # Update session state with new slider values
            st.session_state.a_slider = a_updated
            st.session_state.b_slider = b_updated
            st.session_state.c_slider = c_updated
            st.session_state.d_slider = d_updated
            st.session_state.e_slider = e_updated
            # Indicate calibration is complete
            st.session_state.calibration_done = True

        # Add a "calibration_done" flag to session state if not present
        if 'calibration_done' not in st.session_state:
            st.session_state.calibration_done = False

        # Add the calibration button
        calibration_button = st.button("Calibrate", on_click=action, key='calibration_button')

        if st.session_state.calibration_done:
            st.markdown(
                f"""
                <div style="text-align: center; font-size: 24px; font-weight: bold; color: green;">
                    🎉🎉 Calibration Complete! 🎉🎉<br>
                    Updated Parameters:<br>
                    <b>a</b> = {round(st.session_state.a_slider, 3)}, 
                    <b>b</b> = {round(st.session_state.b_slider, 3)}, 
                    <b>c</b> = {round(st.session_state.c_slider, 3)}, 
                    <b>d</b> = {round(st.session_state.d_slider, 3)}, 
                    <b>e</b> = {round(st.session_state.e_slider, 3)}
                </div>
                """,
                unsafe_allow_html=True
            )

with tab3:
    st.title("❄️ Future Climate Scenarios🌞")
    st.write('Coming Soon')

with tab4:
    st.title("📖 Documentation")
    st.write("Welcome to the documentation for the ABCD Model Calibration App! This guide will walk you through the app's functionality and provide answers to common questions.")
    
    st.header("Overview")
    st.write("This app allows users to retrieve climate and streamflow data, calibrate a hydrological model, and explore future climate scenarios. Here's an overview of the app's key features:")

    st.header("Sections")
    st.write("""
1. **Retrieve Climate Data**: 
   - The model operates on a monthly time step and requires timeseries data for precipitation, minimum and maximum air temperature, and observed streamflow. These data sets are sourced from NOAA and USGS.

2. **Model Calibration**: 
   - The model is calibrated by minimizing the Root Mean Squared Error (RMSE) between observed and simulated streamflow using the **L-BFGS-B** optimization algorithm. This helps adjust model parameters for better accuracy.

3. **Scenarios**: 
   - After calibration, the model can be used to explore future climate and hydrological scenarios based on adjusted parameters. This feature allows for forward-looking analysis of potential changes in streamflow under different conditions.
""")

    st.header("How to Use")
    st.write("""
    - **Navigate between tabs** to access various features of the app, such as retrieving data, calibrating the model, and running future scenarios.
    - **Adjust parameters** using sliders in the **Model Calibration** tab. You can manually adjust them or use the calibration feature to automatically find optimal values.
    - **Visualizations**: Interact with the visualizations to explore streamflow data and calibration results. You can hover over elements for additional information and export data as needed.
    """)

    st.header("FAQs")
    st.write("""
    - **What is the ABCD Model?**
      The ABCD model is a conceptual hydrological model used to estimate streamflow based on precipitation, temperature, and other environmental variables. It simulates streamflow response to precipitation and potential evapotranspiration. The model was developed by Thomas (1981) and is based on two primary storage compartments: soil moisture and groundwater. The model computes streamflow by accounting for soil moisture, surface runoff, groundwater recharge, and discharge.

    - **What are the components of the ABCD model?**
      The model includes the following components:
      - **Soil Moisture**: Gains water from precipitation and loses water to evapotranspiration (ET), surface runoff, and groundwater recharge.
      - **Groundwater**: Gains water from recharge and loses water as discharge.
      - **Runoff**: The surface runoff generated by soil moisture.
      - **Recharge**: The amount of water moving from soil moisture into groundwater.
      - **Discharge**: Groundwater that is discharged into the streamflow.

    - **How can I adjust parameters?**
      You can manually adjust the model parameters in the **Model Calibration** tab using the sliders. Alternatively, the app will calibrate the parameters automatically by minimizing the RMSE between observed and simulated streamflow.

    - **What is the purpose of the calibration?**
      Calibration is essential for optimizing the model's accuracy. By adjusting the parameters based on observed data, the model can provide more realistic simulations of streamflow under varying conditions.

    - **What input data is required?**
      The model runs on a daily time step and requires the following input timeseries:
      - **Precipitation**
      - **Minimum and Maximum Air Temperature**
      - **Observed Streamflow**
      Additionally, air temperature data is used to compute potential evapotranspiration (PET) based on the method described by Shuttleworth (1993).

    - **Can I use this app for future climate scenarios?**
      Yes, once the model is calibrated, you can use it to explore future climate scenarios by adjusting the model's input data for different climate conditions. This helps in understanding how streamflow might change under future environmental changes.
    """)

    st.header("Contact")
    st.write("For questions or support, please contact [inmang2000@gmail.com](mailto:inmang2000@gmail.com).")
