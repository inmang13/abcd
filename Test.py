import pandas as pd
from flask import Flask, render_template, request
from dataretrieval import nwis
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def grabQ(site_no, start_date, end_date):
    """
    Purpose: Retrieve Flow Data from USGS
    :param site_no: USGS site number
    :param start_date: Start date for the data retrieval (YYYY-MM-DD)
    :param end_date: End date for the data retrieval (YYYY-MM-DD)
    :return: A cleaned DataFrame with Discharge data
    """

    print(site_no)
    print(start_date)
    df=nwis.get_dv(
                sites=site_no,
                parameterCd="00060",
                statCd="00003",
                start=start_date,
                end=end_date,
                )[0]

    return df

def create_plot(df):
    """
    Create a plot from the DataFrame.
    :param df: DataFrame containing the data
    :return: Base64 encoded PNG image
    """
    # Make sure the dateTime column is the index for plotting
    #df[0].set_index('dateTime', inplace=True)

    ax = df['00060_Mean'].plot(figsize=(10, 5), marker='o', linestyle='-', color='b')
    ax.set_title('Time Series of Discharge Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Discharge (cubic feet per second)')
    ax.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()  # Close the plot to avoid displaying it inline
    return base64.b64encode(buf.getvalue()).decode('utf8')  # Return the image as base64


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        site_number = request.form['site_number'].strip()  # Get user-entered site number
        preset_site_number = request.form['preset_site_number']  # Get selected preset site number
        
        # If a preset site number was selected, use it; otherwise, use the custom site number
        if preset_site_number:
            site_number = preset_site_number
        
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        
        # Ensure a site number was entered or selected
        if not site_number:
            return render_template('index.html', error='Please enter or select a valid USGS site number.')

        data = grabQ(site_number, start_date, end_date)

        if data is not None and not data.empty:
            plot_url = create_plot(data)
            return render_template('index.html', plot_url=plot_url, site_number=site_number)
        else:
            return render_template('index.html', error='No data found for the specified parameters.', site_number=site_number)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)