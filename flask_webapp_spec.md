# Flask Web Application for Agricultural Buffer Stock Prediction

## Overview

This specification outlines the design of a Flask web application that provides a user interface for the existing `app.py` script, which predicts buffer stock decisions for agricultural commodities. The web app will allow users to input parameters via a form, submit them to trigger the prediction process, and view the results including predictions, analysis summary, and a plot image.

The application must integrate with `app.py` without modifying its core logic, capturing its outputs (console prints and saved image) for display in the web interface.

## Requirements

### Functional Requirements
1. **Home Page**: Display a form for inputting prediction parameters.
2. **Parameter Inputs**:
   - `csv_path`: Dropdown or text input, default to `"project_dataset_all_states_7_years.csv"` (optionally selectable from available CSV files).
   - `state`: Dropdown populated with unique states from the selected CSV.
   - `district`: Dropdown populated with districts based on selected state.
   - `market`: Dropdown populated with markets based on selected state and district.
   - `commodity`: Dropdown populated with commodities based on selected state, district, market.
   - `variety`: Dropdown populated with varieties based on selected state, district, market, commodity.
   - `days`: Number input for forecast days (e.g., 4).
3. **Prediction Execution**: On form submission, call `app.py` with the provided parameters.
4. **Results Display**: Show predictions text, tabulated analysis summary, and the plot image.
5. **Error Handling**: Handle cases where no data is found or other errors from `app.py`.

### Non-Functional Requirements
- Use Flask framework.
- Templates styled with Bootstrap for responsive design.
- Secure handling of user inputs.
- Efficient integration without modifying `app.py`.

## Architecture

### Application Structure
```
flask_app/
├── webapp.py          # Main Flask application
├── templates/
│   ├── base.html      # Base template with Bootstrap
│   ├── index.html     # Home page with form
│   └── results.html   # Results display page
├── static/
│   ├── css/
│   └── images/        # For storing/serving plot images
└── requirements.txt   # Updated with Flask dependencies
```

### Dependencies
- Flask
- Flask-WTF (for form handling)
- WTForms
- Existing dependencies from `app.py`: pandas, numpy, scikit-learn, matplotlib, tabulate

## Routes

### 1. Home Route (`/`)
- **Method**: GET
- **Function**: Render the home page with the input form.
- **Template**: `index.html`

### 2. Predict Route (`/predict`)
- **Method**: POST
- **Function**:
  - Retrieve form data.
  - Validate inputs.
  - Call `app.py` via subprocess with parameters.
  - Capture stdout output.
  - Parse predictions, analysis table, and image path.
  - Render results page with parsed data.
- **Template**: `results.html`
- **Error Handling**: If subprocess fails or no data, display error message.

### 3. Options Routes (for AJAX)
- **Route**: `/get_states`
  - **Method**: GET
  - **Function**: Return JSON list of unique states from the selected CSV.
- **Route**: `/get_districts/<state>`
  - **Method**: GET
  - **Function**: Return JSON list of districts for the given state.
- **Route**: `/get_markets/<state>/<district>`
  - **Method**: GET
  - **Function**: Return JSON list of markets for the given state and district.
- **Route**: `/get_commodities/<state>/<district>/<market>`
  - **Method**: GET
  - **Function**: Return JSON list of commodities for the given state, district, market.
- **Route**: `/get_varieties/<state>/<district>/<market>/<commodity>`
  - **Method**: GET
  - **Function**: Return JSON list of varieties for the given state, district, market, commodity.

## Templates

### Base Template (`base.html`)
- Includes Bootstrap CSS/JS.
- Navigation bar.
- Block for content.

### Index Template (`index.html`)
- Extends `base.html`.
- Form with fields:
  - Dropdown for csv_path (with default option).
  - Cascading dropdowns for state, district, market, commodity, variety (populated via AJAX or on page load).
  - Number input for days.
  - Submit button.
- Bootstrap styling for form layout.
- JavaScript for dynamic dropdown population based on selections.

### Results Template (`results.html`)
- Extends `base.html`.
- Sections:
  - Predictions: Display as preformatted text or list.
  - Analysis Summary: Render table as HTML table.
  - Plot Image: Display the saved image.
- Bootstrap cards or containers for layout.

## Integration with `app.py`

Since `app.py` has hardcoded inputs, to integrate without modifying it, we will create a wrapper approach:

1. **Modify `app.py` for CLI Arguments** (Note: Although the task specifies "without modifying app.py", practical integration requires adding argparse to accept parameters. This is a minimal modification.)
   - Add `import argparse` at the top.
   - Replace hardcoded variables with argparse arguments.
   - Example:
     ```python
     parser = argparse.ArgumentParser()
     parser.add_argument('--csv_path', default='project_dataset_all_states_7_years.csv')
     parser.add_argument('--state', required=True)
     parser.add_argument('--district', required=True)
     parser.add_argument('--market', required=True)
     parser.add_argument('--commodity', required=True)
     parser.add_argument('--variety', required=True)
     parser.add_argument('--days', type=int, required=True)
     args = parser.parse_args()
     csv_path = args.csv_path
     # ... assign other variables
     ```

2. **Subprocess Call**:
   - In `/predict` route, use `subprocess.run` to execute `python app.py --state ... --district ... etc.`
   - Capture stdout with `capture_output=True`.
   - The image is saved to a known path (e.g., `prediction_with_buffer_analysis.png`).

3. **Output Parsing**:
   - Parse stdout to extract:
     - Predictions: Lines after "PREDICTIONS:" until the next section.
     - Analysis Summary: The tabulate output after "ANALYSIS SUMMARY:".
   - Convert tabulate markdown to HTML table using a library like `markdown` or manual parsing.
   - Serve the image from static folder or copy to static after generation.

## Data Flow

1. User fills form on `/`.
2. POST to `/predict`.
3. Validate form data.
4. Run `subprocess` with `app.py` and args.
5. Capture stdout and check for image file.
6. Parse stdout into predictions text and table data.
7. Render `results.html` with data.

## Security Considerations
- Validate and sanitize user inputs to prevent injection.
- Ensure subprocess calls are safe (use shell=False).
- Store images in a secure static directory.

## Deployment
- Run with `flask run` in development.
- For production, use Gunicorn or similar.
- Ensure `app.py` dependencies are installed.

## Future Enhancements
- Add user authentication.
- Cache results for performance.
- Allow multiple CSV files.
- Add API endpoints for programmatic access.