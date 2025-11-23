import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from tabulate import tabulate  # <— import tabulate
import argparse

# ============================
# COMMAND LINE ARGUMENTS
# ============================
parser = argparse.ArgumentParser(description='Predict buffer stock decisions for agricultural commodities.')
parser.add_argument('--csv_path', default='project_dataset_all_states_7_years.csv', help='Path to the CSV dataset file')
parser.add_argument('--state', required=True, help='State name')
parser.add_argument('--district', required=True, help='District name')
parser.add_argument('--market', required=True, help='Market name')
parser.add_argument('--commodity', required=True, help='Commodity name')
parser.add_argument('--variety', required=True, help='Variety name')
parser.add_argument('--days', type=int, required=True, help='Number of days to forecast')

args = parser.parse_args()

csv_path = args.csv_path
state = args.state
district = args.district
market = args.market
commodity = args.commodity
variety = args.variety
days = args.days

# ============================
# LOAD DATA
# ============================
df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]
df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], errors='coerce')
df = df.dropna(subset=['Arrival_Date', 'Modal_Price'])

for c in ['State','District','Market','Commodity','Variety']:
    df[c] = df[c].astype(str).str.strip()

# ============================
# VALIDATE COMBINATION
# ============================
combo_filter = (
    (df['State'] == state) &
    (df['District'] == district) &
    (df['Market'] == market) &
    (df['Commodity'] == commodity) &
    (df['Variety'] == variety)
)
df_combo = df[combo_filter].sort_values('Arrival_Date')
if df_combo.empty:
    print("ERROR: No data found for given combination.")
    raise SystemExit

# ============================
# PREPARE TRAINING FEATURES
# ============================
df = df.sort_values('Arrival_Date')
df['day'] = df['Arrival_Date'].dt.day
df['month'] = df['Arrival_Date'].dt.month
df['year'] = df['Arrival_Date'].dt.year

df['lag_1'] = df.groupby(['State','District','Market','Commodity','Variety'])['Modal_Price'].shift(1)
df = df.dropna(subset=['lag_1'])

X = df[['lag_1','day','month','year']]
y = df['Modal_Price']

# ============================
# TRAIN MODEL
# ============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ============================
# FORECAST NEXT DAYS
# ============================
last = df_combo.iloc[-1]
last_price = last['Modal_Price']
last_date = last['Arrival_Date']

preds = []
for i in range(1, days + 1):
    next_date = last_date + pd.Timedelta(days=i)
    X_future = np.array([[last_price, next_date.day, next_date.month, next_date.year]])
    pred = model.predict(X_future)[0]
    preds.append((next_date.date(), float(pred)))
    last_price = pred

print("PREDICTIONS:")
for d, p in preds:
    print(f"{d} -> {p:.2f}")

# ============================
# BUFFER STOCK DECISION
# ============================
avg_pred_price = np.mean([p for _, p in preds])
current_price = df_combo.iloc[-1]['Modal_Price']
change_pct = (avg_pred_price - current_price) / current_price * 100

THRESHOLD_UP = 2.0
THRESHOLD_DOWN = -2.0

if change_pct > THRESHOLD_UP:
    buffer_advice = "Increase buffer stock"
elif change_pct < THRESHOLD_DOWN:
    buffer_advice = "Reduce buffer stock"
else:
    buffer_advice = "Maintain buffer stock"

# ============================
# DEEP ANALYSIS TABLE
# ============================
table = [
    ["Current Modal Price", f"{current_price:.2f}", ""],
    ["Average Forecast Price", f"{avg_pred_price:.2f}", ""],
    ["% Change (forecast vs current)", f"{change_pct:.2f}%", ""],
    ["Buffer Recommendation", buffer_advice, ""],
    ["Inflation / Volatility Risk",
     ("High risk of price increase, buffer helps" if change_pct > 0 else
      "Risk of price decrease, buffer liquidation risk"),
     ""],
    ["Holding Cost Risk",
     "Storage cost + possible spoilage for perishable stocks",
     ""],
    ["Policy Recommendation",
     ("Consider building buffer" if buffer_advice == "Increase buffer stock" else
      "Consider releasing or maintaining buffer"),
     ""]
]

print("\nANALYSIS SUMMARY:")
print(tabulate(table, headers=["Metric", "Value", "Notes"], tablefmt="github"))

# ============================
# PLOT
# ============================
hist_dates = df_combo['Arrival_Date']
hist_prices = df_combo['Modal_Price']
pred_dates = [d for d, _ in preds]
pred_values = [p for _, p in preds]

plt.figure(figsize=(10,5))
plt.plot(hist_dates, hist_prices, marker='o', label='Historical')
plt.plot(pred_dates, pred_values, marker='x', linestyle='--', label='Predicted')
plt.legend()
plt.title('Price Prediction with Buffer Advice')
plt.xlabel('Date')
plt.ylabel('Modal Price')

plt.annotate(buffer_advice,
             xy=(pred_dates[-1], pred_values[-1]),
             xytext=(0, 20),
             textcoords='offset points',
             fontsize=12,
             arrowprops=dict(arrowstyle='->'))

out_file = 'prediction_with_buffer_analysis.png'
plt.savefig(out_file)
print(f"Plot saved: {out_file}")