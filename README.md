# Agricultural Commodity Price Forecasting Framework

## Description

This project is an AI/ML-based framework designed for forecasting agricultural commodity prices and providing buffer stock recommendations. It utilizes a Random Forest regression model to predict future prices based on historical data and offers actionable advice on buffer stock management (increase, reduce, or maintain). The framework includes a Flask web application interface that allows users to input parameters via dynamic dropdowns and view predictions, analysis summaries, and visual plots.

The backend handles data preprocessing, model training, and forecasting, while the frontend provides an interactive web interface for seamless user interaction.

## Features

- **Price Forecasting**: Predicts future commodity prices using historical data and a Random Forest regressor.
- **Buffer Stock Recommendations**: Provides advice on buffer stock levels based on forecasted versus current prices.
- **Web Interface**: Interactive Flask-based web app with dynamic dropdowns for state, district, market, commodity, and variety selections.
- **Results Display**: Shows predictions, tabulated analysis summaries, and generated plot images.
- **Integration**: Seamlessly integrates ML backend with web frontend via subprocess calls.
- **Data Handling**: Processes historical price data from a CSV file (project_dataset_all_states_7_years.csv).

## Installation

1. Clone or download the project repository to your local machine.
2. Navigate to the project root directory.
3. Install the required dependencies using pip:

   ```
   pip install -r requirements.txt
   ```

4. Ensure Python 3.8+ is installed on your system.
5. (Optional) If RDKit is required for any additional chemical data processing, install it separately as per its documentation.

## Usage

1. Place the historical data file `project_dataset_all_states_7_years.csv` in the project root directory.
2. Run the Flask web application:

   ```
   python flask_app/webapp.py
   ```

3. Open a web browser and navigate to `http://127.0.0.1:5000/` (or the displayed URL).
4. Use the web interface to select parameters (state, district, market, commodity, variety) and submit for forecasting.
5. View the results, including predictions, summary table, and plot image.

For backend-only testing, run:

```
python app.py
```

This will output predictions to the console, generate a tabulated summary, and save a plot image.

## Dependencies

- pandas==1.5.3
- numpy==1.23.5
- scikit-learn==1.2.2
- matplotlib==3.7.1
- joblib==1.2.0
- tabulate==0.9.0
- Flask==2.3.3
- Flask-WTF==1.1.1
- WTForms==3.0.1
- Markdown==3.4.1

Note: RDKit may be required if chemical data processing is involved; install separately if needed.

## Project Structure

```
ML_Buffer/
├── app.py                              # Backend script for data loading, preprocessing, model training, forecasting, and output generation
├── test.py                             # Test script
├── requirements.txt                    # Python dependencies
├── project_dataset_all_states_7_years.csv  # Historical price data
├── prediction_with_buffer_analysis.png # Generated plot image
├── commodity_price_report.md           # Report documentation
├── flask_webapp_spec.md                # Web app specifications
├── flask_app/                          # Flask web application directory
│   ├── webapp.py                       # Flask routes for form handling, AJAX dropdowns, and result rendering
│   ├── prediction_with_buffer_analysis.png  # Plot image for web display
│   ├── static/                         # Static assets
│   │   ├── css/                        # Stylesheets
│   │   └── images/                     # Static images
│   │       └── prediction.png          # Prediction-related images
│   └── templates/                      # HTML templates
│       ├── base.html                   # Base template
│       ├── index.html                  # Home page template
│       └── results.html                # Results page template
└── .idea/                              # IDE configuration (optional)
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and ensure tests pass.
4. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.