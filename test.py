import  pandas as pd

df = pd.read_csv("project_dataset_all_states_7_years.csv")
print(df["State"].unique())