from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import subprocess
import os
import shutil
import re
from markdown import markdown

app = Flask(__name__)

# Load CSV at startup
CSV_PATH = '../project_dataset_all_states_7_years.csv'
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]
for c in ['State','District','Market','Commodity','Variety']:
    df[c] = df[c].astype(str).str.strip()

@app.route('/')
def index():
    # Get initial options
    states = sorted(df['State'].unique())
    return render_template('index.html', states=states, csv_path=CSV_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    csv_path = request.form.get('csv_path', CSV_PATH)
    state = request.form['state']
    district = request.form['district']
    market = request.form['market']
    commodity = request.form['commodity']
    variety = request.form['variety']
    days = int(request.form['days'])

    # Run app.py
    cmd = [
        'python', '../app.py',
        '--csv_path', csv_path,
        '--state', state,
        '--district', district,
        '--market', market,
        '--commodity', commodity,
        '--variety', variety,
        '--days', str(days)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return render_template('results.html', error=result.stderr)

        stdout = result.stdout

        # Parse predictions
        predictions = []
        lines = stdout.split('\n')
        in_predictions = False
        for line in lines:
            if line.startswith('PREDICTIONS:'):
                in_predictions = True
                continue
            if in_predictions:
                if line.strip() == '':
                    break
                if ' -> ' in line:
                    date, price = line.split(' -> ')
                    predictions.append(f"{date.strip()} -> {price.strip()}")

        # Parse table
        table_html = ''
        in_table = False
        table_lines = []
        for line in lines:
            if line.startswith('ANALYSIS SUMMARY:'):
                in_table = True
                continue
            if in_table:
                table_lines.append(line)
        table_md = '\n'.join(table_lines).strip()
        if table_md:
            table_html = markdown(table_md, extensions=['tables'])

        # Copy image
        image_src = 'prediction_with_buffer_analysis.png'
        if os.path.exists(f'../{image_src}'):
            shutil.copy(f'../{image_src}', 'static/images/prediction.png')
            image_url = '/static/images/prediction.png'
        else:
            image_url = None

        return render_template('results.html', predictions=predictions, table_html=table_html, image_url=image_url)

    except Exception as e:
        return render_template('results.html', error=str(e))

# AJAX routes
@app.route('/get_states')
def get_states():
    csv_path = request.args.get('csv_path', CSV_PATH)
    # For now, assume same CSV
    states = sorted(df['State'].unique())
    return jsonify(states)

@app.route('/get_districts/<state>')
def get_districts(state):
    districts = sorted(df[df['State'] == state]['District'].unique())
    return jsonify(districts)

@app.route('/get_markets/<state>/<district>')
def get_markets(state, district):
    markets = sorted(df[(df['State'] == state) & (df['District'] == district)]['Market'].unique())
    return jsonify(markets)

@app.route('/get_commodities/<state>/<district>/<market>')
def get_commodities(state, district, market):
    commodities = sorted(df[(df['State'] == state) & (df['District'] == district) & (df['Market'] == market)]['Commodity'].unique())
    return jsonify(commodities)

@app.route('/get_varieties/<state>/<district>/<market>/<commodity>')
def get_varieties(state, district, market, commodity):
    varieties = sorted(df[(df['State'] == state) & (df['District'] == district) & (df['Market'] == market) & (df['Commodity'] == commodity)]['Variety'].unique())
    return jsonify(varieties)

if __name__ == '__main__':
    app.run(debug=True)