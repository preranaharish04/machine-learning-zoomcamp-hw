#!/usr/bin/env python
"""
train.py — Production Training Pipeline
Smart Grid "Dunkelflaute" Shortfall Predictor (Germany)

Automates the full pipeline: data loading, stitching, feature engineering,
chronological splitting, model training, and serialization.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_and_prepare_data():
    """Load grid + weather CSVs, merge, and engineer features."""
    print("=" * 60)
    print("  Dunkelflaute Shortfall Predictor — Training Pipeline")
    print("=" * 60)

    # --- Load German grid metrics from the multi-country OPSD file ---
    print("\n[1/5] Loading OPSD grid data...")
    grid = pd.read_csv(
        'Data/time_series_60min_singleindex.csv',
        parse_dates=['utc_timestamp'],
        usecols=[
            'utc_timestamp',
            'DE_load_actual_entsoe_transparency',
            'DE_solar_generation_actual',
            'DE_wind_generation_actual',
        ],
    )
    grid_de = grid.dropna().copy()
    grid_de['utc_timestamp'] = grid_de['utc_timestamp'].dt.tz_localize(None)
    print(f"    → Grid rows after dropping NaN: {len(grid_de):,}")

    # --- Load Open-Meteo weather data (3 header metadata lines) ---
    print("[2/5] Loading Open-Meteo weather data...")
    weather = pd.read_csv(
        'Data/open_meteo_weather.csv',
        skiprows=3,
        parse_dates=['time'],
    )
    weather = weather.rename(columns={'time': 'utc_timestamp'})
    print(f"    → Weather rows: {len(weather):,}")

    # --- Inner-join on hourly timestamp ---
    print("[3/5] Merging grid + weather on utc_timestamp...")
    df = pd.merge(grid_de, weather, on='utc_timestamp', how='inner')
    print(f"    → Merged rows: {len(df):,}")

    # --- Target engineering ---
    print("[4/5] Engineering target & features...")
    df['Residual_Load'] = (
        df['DE_load_actual_entsoe_transparency']
        - (df['DE_solar_generation_actual'] + df['DE_wind_generation_actual'])
    )
    df['Target_Residual_Load_24h_Future'] = df['Residual_Load'].shift(-24)

    # Temporal features
    df['hour'] = df['utc_timestamp'].dt.hour
    df['month'] = df['utc_timestamp'].dt.month
    df['day_of_week'] = df['utc_timestamp'].dt.dayofweek
    df['is_winter'] = df['month'].isin([11, 12, 1, 2, 3]).astype(int)

    # Interaction features
    df['total_radiation'] = (
        df['direct_radiation (W/m²)'] + df['diffuse_radiation (W/m²)']
    )
    df['wind_chill_index'] = (
        df['temperature_2m (°C)'] * df['wind_speed_10m (km/h)'] / 10
    )

    df = df.dropna()
    print(f"    → Final dataset: {len(df):,} rows")
    print(
        f"    → Date range: {df['utc_timestamp'].min()} → {df['utc_timestamp'].max()}"
    )

    return df


def train_model(df):
    """Perform chronological split, train LightGBM, evaluate, and serialize."""

    # --- Chronological split (strict: no data leakage) ---
    print("\n[5/5] Chronological train/test split...")
    train_df = df[df['utc_timestamp'].dt.year < 2019]
    test_df = df[df['utc_timestamp'].dt.year >= 2019]
    print(f"    → Training set: {len(train_df):,} rows  (< 2019)")
    print(f"    → Test set:     {len(test_df):,} rows  (≥ 2019)")

    features = [
        'temperature_2m (°C)',
        'wind_speed_100m (km/h)',
        'direct_radiation (W/m²)',
        'diffuse_radiation (W/m²)',
        'hour',
        'month',
        'is_winter',
        'wind_speed_10m (km/h)',
        'total_radiation',
        'wind_chill_index',
    ]

    X_train, y_train = train_df[features], train_df['Target_Residual_Load_24h_Future']
    X_test, y_test = test_df[features], test_df['Target_Residual_Load_24h_Future']

    # --- Train tuned LightGBM model ---
    print("\nTraining finalized LightGBM architecture...")
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    # --- Evaluate on the hold-out test period ---
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n{'─' * 40}")
    print(f"  Test MAE:  {mae:,.2f} MW")
    print(f"  Test RMSE: {rmse:,.2f} MW")
    print(f"{'─' * 40}")

    # --- Feature importance ---
    print("\nFeature Importance (gain):")
    for name, imp in sorted(
        zip(features, model.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"    {name:35s}  {imp:>8,}")

    # --- Serialize model weights ---
    output_file = 'dunkelflaute_model.bin'
    with open(output_file, 'wb') as f_out:
        pickle.dump(model, f_out)

    print(f"\n✓ Model successfully saved to '{output_file}'!")
    return model


if __name__ == '__main__':
    df = load_and_prepare_data()
    train_model(df)

