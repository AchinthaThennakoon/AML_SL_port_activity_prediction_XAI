# Colombo Port Activity Forecast

A Streamlit front end for the Colombo daily vessel-call prediction model in
`SL_port_activity_prediction.ipynb`. The notebook selects the Poisson regression
regularization using 2024 validation data, fits the final model, and exports it
with its feature names, recent history, and SHAP background data. The app loads
that artifact; it does not train or refit the model.

## Run locally

From PowerShell in the repository root, create a project-local virtual
environment and install the app dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the notebook cells through the final model-export code cell to create
`artifacts/colombo_portcalls_model.joblib`, then start the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

The app's default inputs use the latest recorded activity. You can edit the
previous-day calls, calls from seven days earlier, and preceding 7-day average
before generating the forecast. The selected forecast date supplies the month
feature.
