# California Housing Analysis and Prediction

This project includes an interactive data-analysis dashboard and a FastAPI
prediction service. The dashboard explores the California Housing dataset;
the API provides single-record and CSV-based predictions using a trained
Random Forest model.

## Data analyst dashboard

The Streamlit dashboard supports filtering by median household income and
median house age, and includes:

- Summary metrics for the selected block groups
- Home-value distribution and income/value relationship charts
- Average value by income quartile
- A geographic view of block-group values
- Analyst notes and limitations for interpreting the dataset

The dataset represents California census block groups from the 1990 census.
Home values are capped at $500,000, so the charts should not be interpreted as
current listings or individual sale prices.

## Setup on Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first dashboard run may need an internet connection to download the
California Housing dataset if it is not already cached by scikit-learn.

## Run the dashboard

```powershell
streamlit run dashboard.py
```

## Run the prediction API

The saved model files must be present in the project directory.

```powershell
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` to try the API endpoints.

## Run with Docker

Build the image from the project directory:

```powershell
docker build -t california-housing .
```

Run the Streamlit dashboard at `http://localhost:8501`:

```powershell
docker run --rm -p 8501:8501 california-housing
```

To run the FastAPI service from the same image instead, use:

```powershell
docker run --rm -p 8000:8000 california-housing uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/docs` for the API documentation. The dashboard
downloads the California Housing dataset on its first run, so the container
needs network access unless the dataset is already cached.
