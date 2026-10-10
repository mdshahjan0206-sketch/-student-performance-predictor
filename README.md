# EduPredict AI — Student Performance Predictor

An end-to-end machine learning workspace for analysing student data, training models and
predicting academic performance.

This is the React + FastAPI version of the project. It replaces the original Streamlit
application. Streamlit is not installed, imported or required anywhere.

```
Upload  ->  Analyse  ->  Train  ->  Predict  ->  Export
```

---

## Running it

Two processes: a Python API and a Vite dev server. Start the backend first.

### Backend

Windows:

```bat
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

macOS / Linux:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API listens on `http://127.0.0.1:8000`. Interactive docs are at `/api/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The dev server proxies `/api` to the backend, so both run on
one origin in the browser.

To build and serve the production bundle:

```bash
npm run build
npm run preview
```

---

## Requirements

* Python 3.10 or newer
* Node.js 18 or newer

---

## Project layout

```
student-performance-predictor/
├── backend/
│   ├── app/
│   │   ├── main.py              FastAPI app, CORS, error handlers
│   │   ├── api/                 dataset, analysis, model, prediction, jobs routes
│   │   ├── core/                config (paths, model registry, keywords), errors
│   │   ├── ml/                  helpers, preprocessing, training, prediction, eda
│   │   ├── schemas/             request bodies
│   │   ├── services/            store (state), dataset, analysis, model, prediction
│   │   └── utils/               file handling, JSON serialisation
│   ├── tests/                   API tests, equivalence tests, benchmark
│   ├── uploads/                 uploaded files (git-ignored)
│   ├── models/                  fitted pipelines (git-ignored)
│   ├── outputs/                 prediction exports (git-ignored)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── charts/              Plotly wrapper and every chart
│   │   ├── components/          UI primitives, data table, dropzone
│   │   ├── hooks/               workspace state, async loader
│   │   ├── layouts/             sidebar, status header, workflow rail
│   │   ├── pages/               Home, Upload, Analysis, Train, Predict, About
│   │   ├── services/            typed API client
│   │   ├── types/               API payload types
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
├── MIGRATION.md                 old-vs-new feature comparison and measurements
└── README.md
```

---

## What it does

### Upload

CSV and Excel files, validated on arrival: unsupported formats, empty files, unreadable
files and duplicate column names are all rejected with a readable message. Row and column
counts, file size, per-column dtypes, missing-value counts and duplicate-row counts are
computed once and cached.

Target columns are detected by name against a list of student-performance keywords
(`exam_score`, `grade`, `pass_fail`, `performance`, `gpa` and others). If nothing matches,
you pick a column manually on the training page. This keeps the tool focused on student
performance rather than turning into a general-purpose model builder.

### Analysis

Dataset overview, distribution (histogram or box plot), correlation heatmap, missing-value
chart, and scatter comparison between two numeric columns. Charts that do not apply say so:
a box plot on a text column, a heatmap with fewer than two numeric columns, a scatter with
the same column on both axes.

Filters are exploratory. They preview a subset and export it as CSV, and they never replace
the dataset used for training or prediction.

### Training

The problem type is detected from the target column and can be overridden. A model is
recommended based on dataset size. Six models are available:

| Classification | Regression |
| --- | --- |
| Logistic Regression | Linear Regression |
| Decision Tree | Decision Tree Regressor |
| Random Forest | Random Forest Regressor |

Before any fitting happens, the dataset is checked: target present and not all null, enough
rows, enough classes, problem type compatible with the target, plus non-blocking warnings
for class imbalance, high dimensionality, small samples and missing values. An unsuitable
combination is refused with an explanation rather than a scikit-learn exception.

Training runs as a background job, so the interface stays responsive. The status shown
(`queued`, `processing`, `training`, `evaluating`, `completed`, `failed`) is set by the
backend as each stage actually happens.

Results: accuracy, precision, recall and F1 for classification; MAE, MSE, RMSE and R² for
regression. Plus a confusion matrix, a ROC curve for binary targets with probability
scores, a residual plot for regression, and feature importance for tree-based models.

### Prediction

Single-student prediction builds its form from the trained model's own feature columns,
with ranges and defaults taken from the dataset. Batch prediction validates the uploaded
file against the required columns, blocks on missing ones, warns about extra ones, coerces
compatible types, then adds `Prediction` and, where the model supports it, `Confidence` and
`Predicted Class`.

Results are paged from the server and exported as CSV.

---

## State and model invalidation

Streamlit's `session_state` is replaced by backend-managed state behind identifiers:
`dataset_id`, `model_id`, `job_id`, `prediction_id`.

Uploading a new dataset clears the active model, because a model fitted on the previous
data is not valid for the new data. Every model also records the dataset it was trained on,
and the prediction endpoints refuse a model whose dataset is not the active one. A model
trained on dataset A cannot be used after dataset B is uploaded, by any route.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/dataset/upload` | Upload and profile a dataset |
| GET | `/api/dataset/{id}/summary` | Metadata, column profiles, overview |
| GET | `/api/dataset/{id}/preview` | One page of rows, with search |
| GET | `/api/dataset/{id}/statistics` | Descriptive statistics |
| GET | `/api/dataset/{id}/filter-options` | Ranges and category values |
| POST | `/api/dataset/{id}/filter` | Filtered preview (exploratory only) |
| POST | `/api/dataset/{id}/filter/download` | Filtered rows as CSV |
| GET | `/api/analysis/{id}/overview` | Six overview figures |
| GET | `/api/analysis/{id}/histogram` | Distribution of one column |
| GET | `/api/analysis/{id}/boxplot` | Five-number summary and outliers |
| GET | `/api/analysis/{id}/correlation` | Correlation matrix |
| GET | `/api/analysis/{id}/missing` | Missing values per column |
| GET | `/api/analysis/{id}/scatter` | Two numeric columns, sampled |
| GET | `/api/model/config/{dataset_id}` | Targets, columns, model options |
| POST | `/api/model/target-analysis` | Problem type and recommendation |
| POST | `/api/model/preflight` | Validation without training |
| POST | `/api/model/train` | Queue a training job |
| GET | `/api/model/history/{dataset_id}` | Models trained on this dataset |
| GET | `/api/model/{model_id}` | Metadata, metrics and chart data |
| GET | `/api/model/{model_id}/metrics` | Metrics only |
| GET | `/api/model/{model_id}/feature-schema` | Fields for the prediction form |
| POST | `/api/predict/single` | Score one record |
| POST | `/api/predict/batch` | Score an uploaded file |
| GET | `/api/predictions/{id}` | One page of results |
| GET | `/api/predictions/{id}/download` | All results as CSV |
| GET | `/api/jobs/{job_id}` | Training job status |
| GET | `/api/session/status` | Active dataset, model, stage |
| GET | `/api/health` | Liveness |

Failures return a consistent envelope and never include a traceback, stack trace, internal
path or raw Python exception text:

```json
{ "success": false, "error": { "code": "INVALID_DATASET", "message": "Required columns are missing." } }
```

---

## Tests

```bash
cd backend
python -m pytest tests -q
```

56 tests: API coverage for upload, validation, analysis, filtering, training, prediction,
downloads and model invalidation, plus 20 tests that compare this implementation against
the original Streamlit modules numerically. See `MIGRATION.md`.

Measured performance on a 120,000-row dataset:

```bash
uvicorn app.main:app          # in one terminal
python tests/benchmark.py     # in another
```

---

## Privacy

Uploaded datasets stay on the machine running the backend. Files are parsed, analysed,
trained and scored in that process. Nothing is sent to a third-party service, and the only
things written to disk are the uploaded file, the fitted model and prediction output, all
inside `backend/`.

---

Built by Mohamed Shahjan. B.E. Computer Science Engineering.
