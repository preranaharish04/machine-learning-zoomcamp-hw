# ⚡ Smart Grid "Dunkelflaute" Shortfall Predictor (Germany)

## Description of the Problem

**The Challenge:** As electricity grids transition to Variable Renewable Energy (VRE), a prolonged *Dunkelflaute* — a German term for a "dark wind lull" where solar irradiance collapses and wind speeds drop simultaneously — creates catastrophic gaps between grid demand and green supply. During winter 2016–2017, Germany experienced multi-day Dunkelflaute episodes that required emergency activation of coal and gas reserves.

**The Objective:** Build a Supervised ML pipeline that ingests **Spatio-Temporal Weather Covariates** (temperature, wind speed at 10m and 100m, direct and diffuse solar radiation) to accurately forecast the **24-Hour-Ahead Residual Load** (Total Load minus Wind Generation minus Solar Generation) for the German power grid.

**Business / Grid Utility:** Transmission System Operators (TSOs) such as 50Hertz, Amprion, TenneT, and TransnetBW use these predictions to:
- Calculate reserve margins hours in advance
- Accurately procure secondary balancing power, avoiding day-ahead price spikes (€600+/MWh)
- Prevent cascading blackouts during sustained renewable shortfall events

## Dataset

| Source | File | Description |
|--------|------|-------------|
| [OPSD](https://open-power-system-data.org/) | `Data/time_series_60min_singleindex.csv` | Hourly load, solar, and wind generation for Germany (2015–2020) |
| [Open-Meteo](https://open-meteo.com/) | `Data/open_meteo_weather.csv` | Hourly weather: temperature, wind speeds (10m & 100m), direct & diffuse radiation at 51.49°N, 10.43°E |

## Project Structure

```
dunkelflaute-predictor/
├── Data/
│   ├── open_meteo_weather.csv
│   └── time_series_60min_singleindex.csv
├── .gitignore
├── Pipfile
├── Pipfile.lock
├── Dockerfile
├── README.md
├── notebook.ipynb        # EDA, feature engineering, model tuning
├── train.py              # Production training pipeline
└── predict.py            # Flask model-serving microservice
```
## 📊 Dataset Access & Reproducibility

To maintain a clean repository footprint, the heavy raw datasets are hosted externally. You can access and download the exact historical source files used to train and validate this model here:

*   **Google Drive Data Repository:** [Download Project Datasets (Google Drive) 🌐] https://drive.google.com/drive/folders/1jFir3GnEgczzHK1XOgdk6i30J4lwqAN5?usp=sharing

### Dataset Contents:
1.  `time_series_60min_singleindex.csv` — Raw hourly consumption load, solar generation, and wind generation profiles for Germany (extracted via Open Power System Data).
2.  `open_meteo_weather.csv` — Hourly historical weather parameters (100m wind speeds, direct/diffuse radiation matrices) corresponding to the exact grid target timeline (extracted via Open-Meteo ERA5 API).

To replicate the execution pipeline, download these files and place them directly into the `/Data` directory within your local workspace root before executing `notebook.ipynb` or `train.py`.


## Setup & Execution Instructions

### 1. Environment Installation (pipenv)

```bash
# Install pipenv if not already available
pip install pipenv

# Install all dependencies from Pipfile.lock
pipenv install

# Activate the virtual environment
pipenv shell
```

### 2. Train the Model

```bash
python train.py
```

This runs the full pipeline: data loading → merging → feature engineering → chronological train/test split → LightGBM training → serialization to `dunkelflaute_model.bin`.

### 3. Run the Prediction Service Locally

```bash
# On Windows (uses Flask's built-in dev server)
python predict.py

# On Linux/macOS (production-grade via gunicorn)
gunicorn --bind=0.0.0.0:9696 predict:app
```

The service will start at `http://localhost:9696`. Test it with:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": -2.0,
    "wind_speed_100m": 8.5,
    "direct_radiation": 0.0,
    "diffuse_radiation": 5.0,
    "hour": 18,
    "month": 1,
    "is_winter": 1,
    "wind_speed_10m": 4.2,
    "total_radiation": 5.0,
    "wind_chill_index": -0.84
  }'
```

**Example response:**
```json
{
  "predicted_residual_load_mw": 52134.67,
  "dunkelflaute_risk_alert": true
}
```

### 4. Containerization (Docker)

```bash
# Build the Docker image
docker build -t dunkelflaute-service .

# Run the container
docker run -p 9696:9696 dunkelflaute-service
```

The containerized service is now accessible at `http://localhost:9696/predict`.

## 🚀 Local Deployment and Verification

To verify the predictive service locally without a container engine, run the Flask web application using your configured Python environment.

### 1. Launch the Prediction Service
Run the script to start the localized server on port `9696`:
```bash
python predict.py
```

### 2. Test the API Interface
Open a separate terminal window. If using Windows, open a standard Command Prompt (`cmd`) and execute the following `curl` script to send test weather parameters:
```bash
curl -X POST http://localhost:9696/predict -H "Content-Type: application/json" -d "{\"temperature\": -2, \"wind_speed_100m\": 8.5, \"direct_radiation\": 0, \"diffuse_radiation\": 5, \"hour\": 18, \"month\": 1}"
```

### 3. Expected Response Pipeline
The API interface evaluates the feature conditions and responds with a JSON payload indicating tomorrow's forecast residual grid stress load along with an automated risk warning evaluation:
```json
{
  "dunkelflaute_risk_alert": true,
  "predicted_residual_load_mw": 60269.41
}
```
images:

![alt text](<Screenshot (920)-1.png>)
![alt text](<Screenshot (921)-1.png>)

### 5. Cloud Deployment (AWS Elastic Beanstalk)

```bash
# Install the EB CLI
pip install awsebcli

# Initialize an EB application
eb init -p docker dunkelflaute-predictor --region eu-central-1

# Create and deploy the environment
eb create dunkelflaute-env --single --instance-type t3.small

# Open in browser
eb open

# Check health
eb status
```

**Alternative: Local Kubernetes via Minikube**

```bash
# Start Minikube
minikube start

# Point Docker to Minikube's daemon
eval $(minikube docker-env)

# Build image inside Minikube
docker build -t dunkelflaute-service .

# Deploy
kubectl create deployment dunkelflaute --image=dunkelflaute-service --port=9696
kubectl expose deployment dunkelflaute --type=NodePort --port=9696

# Get the URL
minikube service dunkelflaute --url
```

## Model Summary

| Model | Test MAE (MW) | Test RMSE (MW) |
|-------|--------------|----------------|
| Ridge Regression (baseline) | ~6,500 | ~8,200 |
| **LightGBM (selected)** | **~3,800** | **~5,100** |
| XGBoost | ~3,900 | ~5,200 |

The **LightGBM** regressor was selected as the production model based on the lowest MAE on the chronologically held-out 2019+ test period. Key predictive features include wind speed at 100m, hour of day, temperature, and direct solar radiation — all physically consistent with Dunkelflaute dynamics.



