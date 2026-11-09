FROM python:3.13-slim

# Install system dependencies required for LightGBM (OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pipenv

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

# Copy application code and model weights
COPY ["predict.py", "dunkelflaute_model.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]

