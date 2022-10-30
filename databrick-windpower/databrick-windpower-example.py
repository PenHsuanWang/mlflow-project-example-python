# This example is provided by databrick official document
# https://docs.databricks.com/mlflow/model-registry-example.html
# This example illustrates how to use MLflow Model Registry to build a machine learning application that forecasts the daily power output of wind farm.abs
# The pipeline included
# 1. Tracking and log model with MLFlow
# 2. Register models with the Model Registry
# 3. Describe models and make model version stage transitions
# 4. Integrate registered models with production application.
# 5. Achive and delete models.



# load dataset

#%%

import pandas as pd
wind_farm_data = pd.read_csv("https://github.com/dbczumar/model-registry-demo-notebook/raw/master/dataset/windfarm_data.csv", index_col=0)
 
def get_training_data():
    training_data = pd.DataFrame(wind_farm_data["2014-01-01":"2018-01-01"])
    X = training_data.drop(columns="power")
    y = training_data["power"]
    return X, y
 
def get_validation_data():
    validation_data = pd.DataFrame(wind_farm_data["2018-01-01":"2019-01-01"])
    X = validation_data.drop(columns="power")
    y = validation_data["power"]
    return X, y
 
def get_weather_and_forecast():
    format_date = lambda pd_date : pd_date.date().strftime("%Y-%m-%d")
    today = pd.Timestamp('today').normalize()
    week_ago = today - pd.Timedelta(days=5)
    week_later = today + pd.Timedelta(days=5)
    
    past_power_output = pd.DataFrame(wind_farm_data)[format_date(week_ago):format_date(today)]
    weather_and_forecast = pd.DataFrame(wind_farm_data)[format_date(week_ago):format_date(week_later)]
    if len(weather_and_forecast) < 10:
      past_power_output = pd.DataFrame(wind_farm_data).iloc[-10:-5]
      weather_and_forecast = pd.DataFrame(wind_farm_data).iloc[-10:]
  
    return weather_and_forecast.drop(columns="power"), past_power_output["power"]

wind_farm_data["2019-01-01":"2019-01-14"]

# %%

# Defind the tensor keras model and start to train
# Using MLflow autologging function to tracking the training details.

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def train_keras_model(X_train, y_train):
      
    model = Sequential()
    model.add(Dense(100, input_shape=(X_train.shape[-1],), activation="relu", name="hidden_layer"))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
  
    model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=.2)
    return model


import mlflow
import mlflow.keras
import mlflow.tensorflow

mlflow.set_tracking_uri("http://127.0.0.1:5010")
 
X_train, y_train = get_training_data()
 
with mlflow.start_run():
    # Automatically capture the model's parameters, metrics, artifacts,
    # and source code with the `autolog()` function
    mlflow.tensorflow.autolog()
    
    train_keras_model(X_train, y_train)
    run_id = mlflow.active_run().info.run_id
    
# %%

# Registry the model
model_name = "power-forecasting-model"

# The default path where the MLflow autologging function stores the model
artifact_path = "model"
model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
 
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# %%


import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
 
def wait_until_ready(model_name, model_version):
  client = MlflowClient()
  for _ in range(10):
    model_version_details = client.get_model_version(
      name=model_name,
      version=model_version,
    )
    status = ModelVersionStatus.from_string(model_version_details.status)
    print("Model status: %s" % ModelVersionStatus.to_string(status))
    if status == ModelVersionStatus.READY:
      break
    time.sleep(1)
  
wait_until_ready(model_details.name, model_details.version)
# %%

from mlflow.tracking.client import MlflowClient
 
client = MlflowClient()
client.update_registered_model(
  name=model_details.name,
  description="This model forecasts the power output of a wind farm based on weather data. The weather data consists of three features: wind speed, wind direction, and air temperature."
)


# %%


client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version was built using TensorFlow Keras. It is a feed-forward neural network with one hidden layer."
)
# %%


client.transition_model_version_stage(
  name=model_details.name,
  version=model_details.version,
  stage='Production',
)
# %%


model_version_details = client.get_model_version(
  name=model_details.name,
  version=model_details.version,
)
print("The current model stage is: '{stage}'".format(stage=model_version_details.current_stage))
# %%

# Intergrate the model with the forecasting application
# Load the model by pyfunc module, for start to run application.

import mlflow.pyfunc
 
model_version_uri = "models:/{model_name}/1".format(model_name=model_name)
 
print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_version_uri))
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)


# %%
# desing the prediction result plot 

def plot(model_name, model_stage, model_version, power_predictions, past_power_output):
      
    import pandas as pd
    import matplotlib.dates as mdates
    from matplotlib import pyplot as plt
    index = power_predictions.index
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Date", size=20, labelpad=20)
    ax.set_ylabel("Power\noutput\n(MW)", size=20, labelpad=60, rotation=0)
    ax.tick_params(axis='both', which='major', labelsize=17)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.plot(index[:len(past_power_output)], past_power_output, label="True", color="red", alpha=0.5, linewidth=4)
    ax.plot(index, power_predictions.squeeze(), "--", label="Predicted by '%s'\nin stage '%s' (Version %d)" % (model_name, model_stage, model_version), color="blue", linewidth=3)
    ax.set_ylim(ymin=0, ymax=max(3500, int(max(power_predictions.values) * 1.3)))
    ax.legend(fontsize=14)
    plt.title("Wind farm power output and projections", size=24, pad=20)
    plt.tight_layout()
    display(plt.show())
  
def forecast_power(model_name, model_stage):
    from mlflow.tracking.client import MlflowClient
    client = MlflowClient()
    model_version = client.get_latest_versions(model_name, stages=[model_stage])[0].version
    model_uri = "models:/{model_name}/{model_stage}".format(model_name=model_name, model_stage=model_stage)
    model = mlflow.pyfunc.load_model(model_uri)
    weather_data, past_power_output = get_weather_and_forecast()
    power_predictions = pd.DataFrame(model.predict(weather_data))
    power_predictions.index = pd.to_datetime(weather_data.index)
    print(power_predictions)
    plot(model_name, model_stage, int(model_version), power_predictions, past_power_output)
    
#%%
forecast_power(model_name, "Production")

# %%
# Create and deploy the new model.

import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
 
with mlflow.start_run():
    n_estimators = 300
    mlflow.log_param("n_estimators", n_estimators)
    
    rand_forest = RandomForestRegressor(n_estimators=n_estimators)
    rand_forest.fit(X_train, y_train)
  
    val_x, val_y = get_validation_data()
    mse = mean_squared_error(rand_forest.predict(val_x), val_y)
    print("Validation MSE: %d" % mse)
    mlflow.log_metric("mse", mse)
    
    # Specify the `registered_model_name` parameter of the `mlflow.sklearn.log_model()`
    # function to register the model with the MLflow Model Registry. This automatically
    # creates a new model version
    mlflow.sklearn.log_model(
      sk_model=rand_forest,
      artifact_path="sklearn-model",
      registered_model_name=model_name,
    )
# %%

# Fetch the model ID using MLflow Model Registry Search
from mlflow.tracking.client import MlflowClient
client = MlflowClient()
 
model_version_infos = client.search_model_versions("name = '%s'" % model_name)
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])


# %%

wait_until_ready(model_name, new_model_version)

# %%

client.update_model_version(
  name=model_name,
  version=new_model_version,
  description="This model version is a random forest containing 100 decision trees that was trained in scikit-learn."
)


#%%

# Transition the new model version to staging
# Before deploying the model to production application, it is often best practice to test it in a staging environment.

client.transition_model_version_stage(
  name=model_name,
  version=new_model_version,
  stage="Staging",
)


# %%

# Check the staging model performance
forecast_power(model_name, "Staging")
# %%

# if the model performance is good, send the staging model to go to production.
client.transition_model_version_stage(
  name=model_name,
  version=new_model_version,
  stage="Production",
)

# %%

# The final production model 

forecast_power(model_name, "Production")

# %%

# Archive the model, version 1

# When a model version is no longer being used, you can archive it or delete it. You can delete entire registered model, this removes all its associated model versions.

# Archive version_1 windpower-model
from mlflow.tracking.client import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=1,
  stage="Archived",
)

# %%

# Delete version 1 model

client.delete_model_version(
  name=model_name,
  version=1,
)

# %%

# Delete the power forecasting model

client.transition_model_version_stage(
  name=model_name,
  version=2,
  stage="Archived"
)

client.delete_registered_model(name=model_name)

# %%
