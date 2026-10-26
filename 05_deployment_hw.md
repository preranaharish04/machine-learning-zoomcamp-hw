#	Question	            Answer
1	uv version	            0.13.5
2	Scikit-Learn version	1.7.2
3	Conversion probability 
    (loaded model)	        0.533
4	Conversion probability 
    (API)	                0.770
5	Python base image	    python:3.11.15-slim-bookworm

Setup and checks
uv --version -> uv 0.13.5

uv sync --locked -> installed 18 packages using CPython 3.11.15, including scikit-learn==1.7.2

uv run python smoke_test.py -> {'conversion_probability': 0.769799, 'conversion': True}
Checksum of pipeline.bin (SHA256) matches the expected value: 1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1

Question 3: load the model

Script: q3.py. It loads pipeline.bin with pickle and calls predict_proba on the lead:

json
{
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}

Run with uv run python q3.py. 
Output: 0.533

Question 4: serve the model

Started the API:

bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696

Sent the second lead (organic_search, technology, employed, europe, 4 courses viewed, income 80304.0, 7 interactions, lead score 0.74) with PowerShell's Invoke-RestMethod to POST /predict.

Response:

text
conversion_probability conversion
---------------------- ----------
              0.769799       True

Rounded to three decimals: 0.770

Question 5: container configuration

The Dockerfile declares the base image python:3.11.15-slim-bookworm

