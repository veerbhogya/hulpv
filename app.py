import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
from bayes_opt import BayesianOptimization

# Load the XGBoost model
model = xgb.XGBRegressor()
model.load_model('xgboost_model.json')

# Load X_train to get the column names
X_train = pd.read_csv('X_train.csv')  # Update with the actual path to your X_train

st.set_page_config(page_title="Unilever LLPL PV Predictor", layout="wide")

st.markdown("""
<style>
    body {
        background-color: #f5f5f5;
    }
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #0033cc;
        margin-top: 20px;
        font-family: 'Arial', sans-serif;
    }
    .logo {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .main-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 20px rgba(0,0,0,0.1);
        margin-top: 20px;
        font-family: 'Arial', sans-serif;
    }
    .parameter-title {
        font-size: 24px;
        font-weight: bold;
        color: #333333;
        margin-bottom: 20px;
    }
    .prediction {
        font-size: 28px;
        font-weight: bold;
        color: #0033cc;
        text-align: center;
        margin-top: 20px;
        padding: 10px;
        border-radius: 5px;
        border: 2px solid #0033cc;
        background-color: #e6f0ff;
    }
    .stButton button {
        background-color: #0033cc;
        color: white;
    }
    .display-inputs-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# st.markdown("<div class='logo'><img src='Downloads\logo.png' width='100'></div>", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Unilever LLPL PV Prediction Dashboard</div>", unsafe_allow_html=True)

# st.markdown("<div class='main-container'>", unsafe_allow_html=True)

st.markdown("<div class='parameter-title'>Input Parameters</div>", unsafe_allow_html=True)

input_params = []
cols = st.columns(7)

# Define min and max values for each parameter
param_min_max = {
    'turbo_final_in_temp': (5.0, 12.0),
    'FPLDR BAR FLOW RATE': (1000.0, 5000.0),
    'FPLDR MOUTH TEMP': (40.0, 55.0),
    'FPLDR PRESSURE': (10.0, 30.0),
    'FPLDR SOAP TEMP': (35.0, 55.0),
    'GLYCERINE': (0.0, 20.0),
    'NDLR MASS EXIT TEMP': (35.0, 50.0),
    'NOODLE': (350.0, 500.0),
    'PERFUME': (5.0, 12.0),
    'recycle per': (40.0, 100.0),
    'MIXER OUT TEMP': (35.0, 50.0),
    'PSM NDLR TEMP': (25.0, 60.0),
    'RM STARCH': (50.0, 110.0),
    'TURBO FINAL OUT TEMP': (5.0, 35.0),
    'TURBO MIXER OUT TEMP': (5.0, 40.0),
    'TURBO NOODLER OUT TEMP': (5.0, 35.0),
    'TURBO PRE OUT TEMP': (5.0, 35.0),
    'VCM CHAMBER': (-600.0, -500.0),
    'Moisture metre.': (15.0, 25.0),
    'IV.': (38.0, 42.0),
    'salt values.': (0.75, 0.95),
    'FA': (0.01, 1.0),
    'BATCH TIME.': (5.0, 20.0),
}

# Create sliders for each parameter
for i, col in enumerate(X_train.columns):
    min_val, max_val = param_min_max.get(col, (0.0, 100.0))  # Default to (0.0, 100.0) if not specified
    with cols[i % 7]:
        param_value = st.slider(col, min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2, step=0.1)
        input_params.append(param_value)

param_names = list(X_train.columns)

input_data = pd.DataFrame({
    'Parameter': param_names,
    'Value': input_params
})

turbo_final_in_temp = input_data.loc[input_data['Parameter'] == 'turbo_final_in_temp', 'Value'].values[0]
FPLDR_BAR_FLOW_RATE = input_data.loc[input_data['Parameter'] == 'FPLDR BAR FLOW RATE', 'Value'].values[0]
FPLDR_MOUTH_TEMP = input_data.loc[input_data['Parameter'] == 'FPLDR MOUTH TEMP', 'Value'].values[0]
FPLDR_PRESSURE = input_data.loc[input_data['Parameter'] == 'FPLDR PRESSURE', 'Value'].values[0]
FPLDR_SOAP_TEMP = input_data.loc[input_data['Parameter'] == 'FPLDR SOAP TEMP', 'Value'].values[0]
GLYCERINE = input_data.loc[input_data['Parameter'] == 'GLYCERINE', 'Value'].values[0]
NDLR_MASS_EXIT_TEMP = input_data.loc[input_data['Parameter'] == 'NDLR MASS EXIT TEMP', 'Value'].values[0]
NOODLE = input_data.loc[input_data['Parameter'] == 'NOODLE', 'Value'].values[0]
PERFUME = input_data.loc[input_data['Parameter'] == 'PERFUME', 'Value'].values[0]
recycle_per = input_data.loc[input_data['Parameter'] == 'recycle per', 'Value'].values[0]
MIXER_OUT_TEMP = input_data.loc[input_data['Parameter'] == 'MIXER OUT TEMP', 'Value'].values[0]
PSM_NDLR_TEMP = input_data.loc[input_data['Parameter'] == 'PSM NDLR TEMP', 'Value'].values[0]
RM_STARCH = input_data.loc[input_data['Parameter'] == 'RM STARCH', 'Value'].values[0]
TURBO_FINAL_OUT_TEMP = input_data.loc[input_data['Parameter'] == 'TURBO FINAL OUT TEMP', 'Value'].values[0]
TURBO_MIXER_OUT_TEMP = input_data.loc[input_data['Parameter'] == 'TURBO MIXER OUT TEMP', 'Value'].values[0]
TURBO_NOODLER_OUT_TEMP = input_data.loc[input_data['Parameter'] == 'TURBO NOODLER OUT TEMP', 'Value'].values[0]
TURBO_PRE_OUT_TEMP = input_data.loc[input_data['Parameter'] == 'TURBO PRE OUT TEMP', 'Value'].values[0]
VCM_CHAMBER = input_data.loc[input_data['Parameter'] == 'VCM CHAMBER', 'Value'].values[0]
Moisture_metre = input_data.loc[input_data['Parameter'] == 'Moisture metre.', 'Value'].values[0]
IV = input_data.loc[input_data['Parameter'] == 'IV.', 'Value'].values[0]
salt_values = input_data.loc[input_data['Parameter'] == 'salt values.', 'Value'].values[0]
FA = input_data.loc[input_data['Parameter'] == 'FA', 'Value'].values[0]
BATCH_TIME = input_data.loc[input_data['Parameter'] == 'BATCH TIME.', 'Value'].values[0]

if st.button('Predict'):
    input_array = np.array(input_params).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    st.markdown(f"<div class='prediction'>Predicted PV: {prediction.round(2)}</div>", unsafe_allow_html=True)
    def target_function(param1, param2, param3, param4, param5):
        # Create a copy of the dataset
        X_opt = X_train.copy()
        
        X_opt['FPLDR BAR FLOW RATE'] = FPLDR_BAR_FLOW_RATE
        X_opt['FPLDR MOUTH TEMP'] = FPLDR_MOUTH_TEMP
        X_opt['FPLDR PRESSURE'] = FPLDR_PRESSURE
        X_opt['FPLDR SOAP TEMP'] =FPLDR_SOAP_TEMP 
        X_opt['GLYCERINE'] = GLYCERINE
        X_opt['NDLR MASS EXIT TEMP'] = NDLR_MASS_EXIT_TEMP
        X_opt['NOODLE'] = NOODLE
        X_opt['PERFUME'] = PERFUME
        X_opt['recycle per'] = recycle_per
        X_opt['MIXER OUT TEMP'] = MIXER_OUT_TEMP
        X_opt['RM STARCH'] = RM_STARCH
        X_opt['VCM CHAMBER'] = VCM_CHAMBER
        X_opt['Moisture metre.'] = Moisture_metre
        X_opt['IV.'] = IV
        X_opt['salt values.'] = salt_values
        X_opt['FA'] = FA
        X_opt['PSM NDLR TEMP'] = PSM_NDLR_TEMP
        X_opt['turbo_final_in_temp'] = turbo_final_in_temp

        # Set the 3 parameters to be optimized
        X_opt['TURBO FINAL OUT TEMP'] = param1
        X_opt['TURBO MIXER OUT TEMP'] = param2
        X_opt['TURBO NOODLER OUT TEMP'] = param3
        X_opt['TURBO PRE OUT TEMP'] = param4
        X_opt['BATCH TIME.'] = param5

        # Predict the output
        y_pred = model.predict(X_opt)
        
        # Calculate how close the prediction is to the desired range [3.8, 4.0]
        if np.all((3.6 <= y_pred) & (y_pred <= 3.8)):
            return 0  # Perfect fit
        else:
            return -np.mean(np.abs(y_pred - 3.7))   # Penalize deviations from the target range

    # Define the parameter bounds (replace with actual min and max values for each parameter)
    pbounds = {
        'param1': (10, 20),
        'param2': (10, 25),
        'param3': (10, 20),
        'param4': (10, 20),
        'param5': (11, 15),
        # 'param6': (8, 20),
        # 'param7': (20, 35),
        # 'param8': (-600, -550),
        # 'param9': (45, 50),
        # 'param10': (6,10)
    }

    # Perform Bayesian Optimization
    optimizer = BayesianOptimization(
        f=target_function,
        pbounds=pbounds,
        random_state=42,
        verbose=1
    )

    optimizer.maximize(
        init_points=10,
        n_iter=100,
    )

    # Collect the results
    results = optimizer.res

    # Extract the valid points
    valid_params = [res['params'] for res in results if -res['target'] <= 0.1]

    # Extract ranges
    param1_values = [params['param1'] for params in valid_params]
    param2_values = [params['param2'] for params in valid_params]
    param3_values = [params['param3'] for params in valid_params]
    param4_values = [params['param4'] for params in valid_params]
    param5_values = [params['param5'] for params in valid_params]

    optimal_param1_range = (round(np.mean(param1_values) - np.std(param1_values) / 2,2), np.mean(param1_values) + np.std(param1_values) / 2)
    optimal_param2_range = (np.mean(param2_values) - np.std(param2_values) / 2, np.mean(param2_values) + np.std(param2_values) / 2)
    optimal_param3_range = (np.mean(param3_values) - np.std(param3_values) / 2, np.mean(param3_values) + np.std(param3_values) / 2)
    optimal_param4_range = (np.mean(param4_values) - np.std(param4_values) / 2, np.mean(param4_values) + np.std(param4_values) / 2)
    optimal_param5_range = (np.mean(param5_values) - np.std(param5_values) / 2, np.mean(param5_values) + np.std(param5_values) / 2)

    st.markdown("<div class='main-title'>Optimal Ranges</div>", unsafe_allow_html=True)
    st.write(f"Optimal Range of turbo final plodder out temp: { optimal_param1_range}")
    st.write(f"Optimal Range of turbo mixer out temp: { optimal_param2_range}")
    st.write(f"Optimal Range of turbo noodler out temp: { optimal_param3_range}")
    st.write(f"Optimal Range of turbo pre plodder out temp: { optimal_param4_range}")
    st.write(f"Optimal Range of batch time: { optimal_param5_range}")


else:
    st.markdown("<div class='prediction'>Enter values and click Predict</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
