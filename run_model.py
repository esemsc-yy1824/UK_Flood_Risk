import tools.model_tools as mt
from tools.model_tools import Models
from sklearn.metrics import r2_score
import pandas as pd

tool = mt.Tool()
models = Models()

# Load the data
print("Loading the data...")
X_train = tool._postcodedb.drop(columns=["riskLabel"])
X_train[["easting", "northing", "elevation"]] = X_train[["easting", "northing", "elevation"]].astype('int64')
y_train = tool._postcodedb["riskLabel"].astype(int)

# Train the model
print("Training the model...")
risk_pipe = models.flood_risk_model()
risk_pipe.fit(X_train, y_train)

# Evaluate the model
print("Evaluating the model...")
prediction = risk_pipe.predict(X_train)
r2 = r2_score(y_train, prediction)
print(f"R2 score of the model: {r2}")

# Save the model
df = pd.read_csv("./example_data/postcodes_unlabelled.csv")
df.drop_duplicates(inplace=True)

df_impute_missing = tool.impute_missing_values(df)

df_drop_noRecord = tool.drop_noRecord_cat_row(df_impute_missing)
df_drop_noRecord[["easting", "northing", "elevation"]] = df_drop_noRecord[["easting", "northing", "elevation"]].astype('int64')

prediction = risk_pipe.predict(df_drop_noRecord)
prediction = pd.DataFrame(prediction, columns=["riskLabel"])


df_drop_noRecord = df_drop_noRecord.reset_index() 
df_pre = pd.concat([df_drop_noRecord, prediction], axis=1)

df_pre.drop_duplicates(inplace=True)   # TODO:there are two duplicate processes in the data

df_latlon = tool.easting_northing_to_lat_lon(df_pre, "easting", "northing")

df_output = pd.concat([df_pre, df_latlon], axis=1)
df_output.to_csv("./output/postcodes_unlabelled_predicted.csv", index=False)
print("Model trained and evaluated successfully!")
print("The predicted result file is already saved in the output folder.")