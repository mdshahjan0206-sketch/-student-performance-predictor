# Migration report

Streamlit to React + FastAPI, for EduPredict AI.

Every row below was checked against the running application, not read off the source. The
equivalence tests referenced throughout live in `backend/tests/test_equivalence.py` and run
the original `utils/*.py` modules in-process (behind a small Streamlit stub) so both
implementations can be handed the same data and compared exactly.

---

## 1. Feature comparison

### Home

| Feature | Original Streamlit | React + FastAPI | Result |
| --- | --- | --- | --- |
| EduPredict AI branding | `hero()` with badge and icon | Hero card, sidebar brand | Kept |
| Welcome and introduction | Static markdown | Hero copy | Kept |
| Workflow explanation | Six feature cards | Five numbered steps, linked | Kept |
| Upload Dataset action | `st.switch_page` button | Router link | Kept |
| About action | `st.switch_page` button | Router link | Kept |
| Statistics strip | Four static `stat_html` figures | Four live metric cards | Improved: reflects real state |
| Visual cards | Streamlit columns | Card grid | Kept |

### Upload

| Feature | Original Streamlit | React + FastAPI | Result |
| --- | --- | --- | --- |
| CSV upload | `st.file_uploader` | `POST /api/dataset/upload` | Kept |
| XLSX upload | Same | Same | Kept |
| File validation | Extension, empty, duplicate columns | Same checks, same messages | Kept |
| File size | `st.metric` in KB | Metric card, B/KB/MB | Kept |
| Row count | `st.metric` | Metric card | Kept |
| Column count | `st.metric` | Metric card | Kept |
| Column names | `column_info_cards` | Column profile grid | Kept |
| Data types | Per-column dtype | Per-column dtype | Kept |
| Missing-value summary | Table plus warning | Warning plus per-column list | Kept |
| Duplicate-row count | `st.metric` plus warning | Metric card plus warning | Kept |
| Detected target columns | Success or warning message | Same, with column chips | Kept |
| Dataset preview | First 10 rows | Paged, searchable, server-side | Improved |
| Success/error messages | `st.success` / `st.error` | Alert components | Kept |
| Navigation to analysis | `st.switch_page` | Router link | Kept |
| Drag and drop | Streamlit built-in | Custom dropzone, keyboard reachable | Improved |
| Upload progress | None | Real XHR progress | Added |

### Target detection

| Feature | Original Streamlit | React + FastAPI | Result |
| --- | --- | --- | --- |
| Keyword list | 18 keywords, priority ordered | Identical list and order | Kept |
| Matching rule | Normalise, substring both ways | Identical | Kept, verified |
| Candidate ordering | By keyword priority | Identical | Kept, verified |
| Manual selection fallback | Checkbox plus selectbox | Checkbox plus select | Kept |
| Stays performance-specific | Not generic AutoML | Same | Kept |

Verified by `test_target_detection_agrees`, including the quirk where a single-character
column name matches because `"a"` is a substring of `"label"`. That behaviour is preserved
rather than silently corrected.

### Validation

| Feature | Original Streamlit | React + FastAPI | Result |
| --- | --- | --- | --- |
| Missing target column | `validate_schema` message | Identical message | Kept, verified |
| Completely null target | `validate_schema` message | Identical message | Kept, verified |
| Empty dataset | `st.error` | `EMPTY_DATASET` error | Kept |
| Invalid files | `st.error` | `INVALID_FILE` / `INVALID_CSV` | Kept |
| Unsupported formats | `st.error` | `UNSUPPORTED_FORMAT` | Kept |
| Invalid schema | Pre-flight validation | Identical | Kept, verified |
| No raw tracebacks | Logged, templated alerts | Logged, error envelope | Kept |

### Analysis

| Feature | Original Streamlit | React + FastAPI | Result |
| --- | --- | --- | --- |
| Total rows / columns | `st.metric` | Metric cards | Kept |
| Numeric / categorical counts | `st.metric` | Metric cards | Kept |
| Missing values / duplicates | `st.metric` | Metric cards | Kept |
| Column selection | `st.selectbox` | Select | Kept |
| Chart type | Histogram, Box Plot | Same two | Kept |
| Histogram | `px.histogram` | Server-side bins, Plotly bar | Kept |
| Box plot | `px.box` | Server-side quartiles plus outliers | Kept |
| Applicability checks | Info message on non-numeric | Same message | Kept |
| Correlation heatmap | `px.imshow` | Server-side matrix, Plotly heatmap | Kept |
| Fewer than two numeric columns | Info message | Same message | Kept |
| Missing-data chart | `px.bar` | Plotly bar plus percentages | Kept |
| No missing values | Success message | Same message | Kept |
| Scatter, X and Y numeric | Two selectboxes | Two selects | Kept |
| X and Y cannot match | Y options exclude X | Same, plus server check | Kept |
| Numeric range filter | `st.slider` | From/to number inputs | Kept |
| Categorical filter | `st.multiselect` | Checkbox list | Kept |
| Remove missing rows | `st.checkbox` | Checkbox | Kept |
| Filtered-row count | Info message | Badge plus count | Kept |
| Filtered preview | First 50 rows | First 50 rows | Kept |
| Filtered CSV download | `premium_table` download | `POST .../filter/download` | Kept |
| Filters never touch training data | Page-local `filtered_df` | Backend never writes back | Kept, verified |
| Numeric summary table | Not present | `describe()` table | Added |

`test_filters_do_not_mutate_the_stored_dataset` filters a 608-row dataset down, then
re-reads the summary and overview and asserts both still report 608 rows.

### Training

| Feature | Original Streamlit | React + FastAPI | Result |
| --- | --- | --- | --- |
| Problem type detection | Bool/non-numeric, `nunique <= 20`, ratio < 0.05 | Identical function | Kept, verified |
| Manual problem override | Radio: Auto, Classification, Regression | Segmented control, same three | Kept |
| Model recommendation | <200 rows linear, else Random Forest | Identical function | Kept, verified |
| Logistic Regression | `max_iter=1000, random_state=42` | Identical | Kept, verified |
| Decision Tree | `random_state=42` | Identical | Kept, verified |
| Random Forest | `random_state=42` | Identical | Kept, verified |
| Linear Regression | Default | Identical | Kept, verified |
| Decision Tree Regressor | `random_state=42` | Identical | Kept, verified |
| Random Forest Regressor | `random_state=42` | Identical | Kept, verified |
| Off-recommendation warning | `st.warning` | Alert | Kept |
| Pre-flight validation | `validate_before_training` | Identical function | Kept, verified |
| Blocking vs warning | `ok` flag plus warnings list | Identical | Kept, verified |
| Recommended models on failure | Listed in alert | Listed in alert | Kept |
| SimpleImputer | Median / constant | Identical | Kept, verified |
| StandardScaler | Numeric | Identical | Kept, verified |
| OneHotEncoder | `handle_unknown='ignore'`, dense | Identical | Kept, verified |
| ColumnTransformer | num plus cat | Identical | Kept, verified |
| Pipeline | preprocessor plus model | Identical | Kept, verified |
| Train/test split | `test_size=0.2, random_state=42` | Identical | Kept, verified |
| Stratification | When classes >1 and min count >=2 | Identical | Kept, verified |
| Accuracy, Precision, Recall, F1 | Weighted, `zero_division=0` | Identical | Kept, verified |
| MAE, MSE, RMSE, R² | As written | Identical | Kept, verified |
| Confusion matrix | `px.imshow`, labels sorted by str | Same data, Plotly heatmap | Kept, verified |
| ROC curve and AUC | Binary plus `predict_proba` only | Identical condition | Kept, verified |
| Residual plot | Regression only | Identical condition | Kept, verified |
| Feature importance | Tree models, top 10, one-hot names | Identical | Kept, verified |
| Results persist across navigation | Session state re-render | Workspace state plus model id | Kept |
| Training does not freeze the UI | Blocking spinner | Background job plus polling | Improved |

`test_classification_training_is_identical` and `test_regression_training_is_identical` run
all six models and assert equal predictions, equal metrics to 1e-12, equal confusion
matrices, equal ROC arrays and equal feature importances.

### Prediction

| Feature | Original Streamlit | React + FastAPI | Result |
| --- | --- | --- | --- |
| Form built from trained features | `feature_columns`, never raw columns | Identical | Kept, verified |
| Numeric inputs | `st.number_input`, step 0.1 | Number input, step 0.1 | Kept |
| Integer inputs | Step 1, rounded mean default | Step 1, rounded mean default | Kept |
| Categorical inputs | `st.selectbox`, sorted values | Select, sorted values | Kept |
| Dataset-derived ranges | Margin `max((max-min)*0.2, 1.0)` | Identical | Kept, verified |
| Default values | Column mean | Column mean | Kept |
| Feature ordering | Training order | Training order | Kept, verified |
| Prediction | `model.predict(...)[0]` | Identical | Kept |
| Confidence | `max(predict_proba)` | Identical | Kept |
| Predicted class | `classes_[argmax]` | Identical | Kept |
| Batch upload | CSV and Excel | Same | Kept |
| Required column check | Blocking error | Blocking error, same message | Kept, verified |
| Extra column check | Warning, kept in output | Identical | Kept, verified |
| Data type check | Numeric vs categorical | Identical | Kept, verified |
| Value coercion | `coerce_batch_dtypes` | Identical function | Kept, verified |
| `Prediction` column | Appended | Appended | Kept, verified |
| `Confidence` column | When available | When available | Kept, verified |
| `Predicted Class` column | When available | When available | Kept, verified |
| Results table | Full frame rendered | Paged from the server | Improved |
| Prediction distribution | Bar or 20-bin histogram | Same, same bin count | Kept |
| CSV download | `st.download_button` | `GET .../download` | Kept |

`test_batch_output_structure_is_identical` reproduces the original page's batch code inline
and asserts the ported path yields the same columns in the same order with the same values.

### About

| Feature | Original Streamlit | React + FastAPI | Result |
| --- | --- | --- | --- |
| About section | Card | Card | Kept |
| Mission | Feature card | Card | Kept |
| Vision | Feature card | Card | Kept |
| Technology stack | Streamlit, Pandas, scikit-learn, Plotly | React, Vite, TypeScript, Tailwind, FastAPI, pandas, NumPy, scikit-learn, Plotly.js | Updated, as required |
| Architecture | Session-manager handoff | Client, API, services, pipeline | Updated |
| Timeline | Four milestones | Five, migration added | Extended |
| Developer | Profile card | Profile card | Kept |
| Contact | Card | Card | Kept |

### State

| Behaviour | Original Streamlit | React + FastAPI | Result |
| --- | --- | --- | --- |
| Active dataset | `st.session_state["dataset"]` | `dataset_id` on the backend | Replaced |
| Trained model | `st.session_state["model"]` | `model_id` on the backend | Replaced |
| Training metadata | Stored in session | Stored on the model record | Replaced |
| Feature columns, target, problem type | Stored in session | Stored on the model record | Replaced |
| Training results | Stored in session | Stored on the model record | Replaced |
| New dataset clears the model | `set_dataset` calls `clear_trained_model` | `set_active_dataset` clears the model, plus a dataset-id check on every prediction | Kept and strengthened |

`test_new_dataset_invalidates_the_previous_model` trains a model, uploads a different
dataset, and asserts the active model is cleared and the old model id is refused with HTTP
409.

---

## 2. Streamlit removal

| Removed | Replaced by |
| --- | --- |
| `streamlit` dependency | Not in `requirements.txt` |
| `st.session_state` | Backend state behind ids |
| `st.file_uploader` | Dropzone plus multipart endpoint |
| `st.button` | Semantic `<button>` |
| `st.selectbox` | `Select` |
| `st.multiselect` | Checkbox list |
| `st.checkbox` | `Checkbox` |
| `st.slider` | From/to number inputs |
| `st.number_input` | Number input |
| `st.text_input` | Text input |
| `st.tabs` | Tab buttons with `role="tab"` |
| `st.metric` | `MetricCard` |
| `st.dataframe` | `DataTable` |
| `st.plotly_chart` | Plotly.js in the browser |
| `st.download_button` | Download endpoints |
| `st.spinner` | Job status and skeletons |
| `st.container` | `Card` |
| `st.expander` | Disclosure with `aria-expanded` |
| `st.switch_page` | React Router |
| `pages/*.py` | `src/pages/*.tsx` |
| `utils/ui.py` (745 lines) | Tailwind design system and components |
| `utils/session_manager.py` | `app/services/store.py` |
| `utils/cache.py` (`st.cache_data`) | Per-dataset cache in the store |
| `assets/css/*.css` (1,987 lines) | `tailwind.config.js` and `index.css` |

No file in the new project imports Streamlit, and no dead Streamlit code was carried over.
The one place the name appears is `backend/tests/test_equivalence.py`, which stubs it
deliberately so the original code can be executed for comparison.

---

## 3. Measured performance

120,000 rows, 2.7 MB, on this machine. Median of three requests unless marked otherwise.
These are wall-clock HTTP round trips to uvicorn, not estimates.

| Operation | Time |
| --- | --- |
| Upload, parse and profile (cold) | 175 ms |
| Dataset summary (cached) | 0.9 ms |
| Preview page, 25 rows | 1.3 ms |
| Preview search across all columns | 278 ms |
| Correlation heatmap (cold) | 9.6 ms |
| Correlation heatmap (cached) | 0.9 ms |
| Histogram (cold) | 4.5 ms |
| Histogram (cached) | 1.0 ms |
| Scatter, sampled to 5,000 points (cold) | 17 ms |
| Box plot (cached) | 1.1 ms |
| Overview stats (cached) | 0.7 ms |
| Filter preview, numeric plus drop-missing | 8.1 ms |
| Linear Regression, end to end | 220 ms (fit 81 ms, R² 0.8875) |
| Random Forest Regressor, end to end | 27.2 s (fit 25.5 s, R² 0.8727) |
| Feature schema for the form | 17 ms |
| Single prediction | 34 ms |
| Batch prediction, 120,000 rows | 4.2 s |
| Results page, 25 of 120,000 rows | 43 ms |
| CSV export of all rows | 312 ms |

Frontend bundle, production build:

| Asset | Raw | Gzipped |
| --- | --- | --- |
| `index.js` | 243 kB | 75 kB |
| `index.css` | 25 kB | 5 kB |
| `plotly.js` (lazy) | 1,385 kB | 474 kB |

Plotly is a separate chunk fetched when the first chart mounts, so it is not part of the
initial load. The cartesian build is used rather than the full distribution: it carries the
bar, box, heatmap and scatter traces this app draws, and it renders as SVG, which keeps
charts working without WebGL — the same concern the original addressed with
`render_mode="svg"`.

**No comparative speed claim is made against the original.** The two applications were not
benchmarked head to head, so stating a multiplier would not be honest. What can be stated
is structural: Streamlit re-executed the whole page script on every widget interaction,
whereas here changing a column or switching a tab issues one request whose result is
cached, and changing pages issues none.

### Where the numbers come from

* Analysis results are cached per dataset under keys that include the chart's parameters,
  which is the gap between the cold and cached rows above.
* Aggregate charts are computed over every row. Only row-level charts sample, with a fixed
  seed: scatter and residual plots cap at 5,000 points.
* Training never samples. The Random Forest figure is 25.5 s of genuine `fit` on all
  120,000 rows, and it is why training runs as a background job.
* Preview search is the slowest read because it scans every cell of every column as text.
  It is a deliberate trade: the alternative is restricting search to one column.

---

## 4. Test coverage

56 tests, all passing.

```
backend/tests/test_api.py           36 tests   upload, validation, analysis, filtering,
                                               training, prediction, downloads,
                                               invalidation, session status
backend/tests/test_equivalence.py   20 tests   numerical equivalence with the original
```

Covered from the brief's checklist: CSV upload, XLSX upload, invalid file, empty file,
missing target, missing values, duplicate detection, overview, histogram, box plot,
heatmap, missing-value chart, scatter, numeric filtering, categorical filtering,
missing-row filtering, filtered preview, target auto-detection, manual target,
classification detection, regression detection, model selection, invalid dataset handling,
training, metrics, confusion matrix, ROC and AUC, residual plot, feature importance, single
prediction, invalid input, batch prediction, missing columns, extra columns, datatype
handling, confidence, results and CSV download.

---

## 5. Known behaviour worth stating

* **Duplicate column names.** pandas renames a repeated CSV header to `grade.1` while
  reading, so the duplicate-column guard rarely fires for CSV input. The check is kept
  because it still catches frames that genuinely carry duplicate labels. This matches the
  original exactly; it was not introduced here.
* **Single-character column names.** `detect_target_candidates` matches a column named `a`
  because `"a"` is a substring of the keyword `"label"`. This is original behaviour and is
  preserved. Changing it would have been a silent behavioural change outside the scope of a
  migration.
* **Sessions.** The client generates an id so two browser tabs do not share an active
  dataset. It is not authentication: it carries no identity and the server uses it only as
  a dictionary key.
* **Restart.** Uploaded files, models and prediction outputs persist on disk under
  `backend/`. In-memory caches rebuild from those files on demand. If the files are removed,
  the session status reports no active dataset rather than pointing at something missing.

---

## 6. Enhancement phase (post-migration)

Work is underway to add 16 requested features on top of this base without touching the
existing ML/training/prediction logic. Delivered so far:

* **Student Performance Dashboard** — `GET /api/dashboard/{dataset_id}` aggregates real
  dataset, model and prediction figures (`app/services/dashboard_service.py`). A
  pass/fail split is only shown when the trained target's two class labels are
  recognisably pass/fail-shaped; otherwise a plain class distribution is returned rather
  than guessing which class is "passing".
* **Model Comparison & Training History** — reuses the existing
  `GET /api/model/history/{dataset_id}` endpoint; the frontend now presents it as both a
  side-by-side comparison (table + bar chart) and an expandable history list. No model is
  ever ranked as "best".
* **Prediction History** — new `GET /api/predictions/history/{dataset_id}`, built from the
  `PredictionRecord`s already stored; the full result frame is never duplicated.
* **Performance / Processing-Time Monitor** — `app/core/timing.py` wraps upload, EDA
  overview, correlation, training and single/batch prediction with real
  `time.perf_counter()` measurements, logged via `Store.log_performance()` and exposed at
  `GET /api/performance`. Nothing here is a synthetic benchmark.
* **Dark/Light Theme** — CSS custom properties (`--color-*`) drive every Tailwind colour
  token; `hooks/useTheme.tsx` toggles a `data-theme` attribute and persists the choice.
  Plotly charts read the same variables at render time so they follow the theme too.
* **Quick Actions** — dashboard shortcuts that disable themselves based on real workspace
  state (no dataset → Analyze/Train disabled; no model → Predict disabled).

New tests: `backend/tests/test_dashboard_performance_history.py` (self-contained, does not
require the external `DATA_DIR` fixtures).

Still to come: At-Risk Student Detection, Individual Student Profile, Interactive
Prediction Explorer, AI-generated Dataset Insights, Explainable Predictions/SHAP, Advanced
Dataset Explorer, Student Risk Heatmap, Performance Distribution Dashboard, and one-click
PDF/HTML Report Generation.
