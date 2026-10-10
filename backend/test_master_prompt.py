import io
import pandas as pd
from starlette.testclient import TestClient
from app.main import app

client = TestClient(app)
session_id = 'test_master_session'
headers = {'X-Session-Id': session_id}

print("Running Master Prompt Test Suite...")

# 1. Test clean dataset upload
df_clean = pd.DataFrame({'hours': [2, 4, 6, 8], 'score': [50, 70, 80, 95]})
csv_buf = io.BytesIO()
df_clean.to_csv(csv_buf, index=False)
csv_buf.seek(0)
res = client.post('/api/dataset/upload', files={'file': ('clean.csv', csv_buf.getvalue(), 'text/csv')}, headers=headers)
assert res.status_code == 200, res.text
clean_id = res.json()['dataset']['dataset_id']
assert res.json()['dataset']['total_missing'] == 0, 'Clean dataset has missing values!'
print("[PASS] Clean dataset upload (0 missing)")

# 2. Test dataset with missing values
with open(r'C:\Users\Mohamed shahjan s\Downloads\student_performance_missing_10000.csv', 'rb') as f:
    res = client.post('/api/dataset/upload', files={'file': ('student_performance_missing_10000.csv', f.read(), 'text/csv')}, headers=headers)
assert res.status_code == 200, res.text
missing_id = res.json()['dataset']['dataset_id']
total_missing = res.json()['dataset']['total_missing']
num_rows = res.json()['dataset']['num_rows']
num_cols = res.json()['dataset']['num_cols']
assert total_missing == 12028, f'Expected 12028 missing, got {total_missing}'
assert num_rows == 10000, f'Expected 10000 rows, got {num_rows}'
assert num_cols == 16, f'Expected 16 cols, got {num_cols}'
print(f"[PASS] Missing dataset upload (10,000 rows x 16 cols, {total_missing} missing cells)")

# 3. Before refinement: download cleaned must fail with 400
res_dl_fail = client.get(f'/api/dataset/{missing_id}/download-cleaned', headers=headers)
assert res_dl_fail.status_code == 400, f'Expected 400 before refinement, got {res_dl_fail.status_code}'
print("[PASS] Safety check: Cleaned dataset download blocked before refinement (400)")

# 4. Training blocked before refinement
res_train_block = client.post('/api/model/train', json={
    'dataset_id': missing_id,
    'target_column': 'final_exam_score',
    'problem_type': 'regression',
    'model_choice': 'Random Forest'
}, headers=headers)
assert res_train_block.status_code == 400, f'Expected 400 training before refinement, got {res_train_block.status_code}'
print("[PASS] Training gate: Blocked before refinement (400)")

# 5. Refinement
res_refine = client.post(f'/api/dataset/{missing_id}/refine', headers=headers)
assert res_refine.status_code == 200, res_refine.text
refine_data = res_refine.json()['refinement']
assert res_refine.json()['success'] is True
assert refine_data['total_missing'] == 12028
print(f"[PASS] Automatic refinement complete: {refine_data['total_missing']} missing values handled")

# 6. Download cleaned dataset
res_dl = client.get(f'/api/dataset/{missing_id}/download-cleaned', headers=headers)
assert res_dl.status_code == 200
disposition = res_dl.headers.get('content-disposition', '')
assert 'attachment; filename=' in disposition
cleaned_df = pd.read_csv(io.StringIO(res_dl.text))
assert len(cleaned_df) == 10000, f'Cleaned df rows mismatch: {len(cleaned_df)}'
assert len(cleaned_df.columns) == 16, f'Cleaned df cols mismatch: {len(cleaned_df.columns)}'
feature_cols = [c for c in cleaned_df.columns if c != 'final_exam_score']
remaining_missing = cleaned_df[feature_cols].isna().sum().sum()
assert remaining_missing == 0, f'Cleaned features still have {remaining_missing} missing values!'
print(f"[PASS] Cleaned dataset downloaded successfully: 10,000 rows x 16 cols, 0 feature missing cells!")

# 7. Dashboard summary verification
res_dash = client.get(f'/api/dashboard/{missing_id}', headers=headers)
assert res_dash.status_code == 200
dash = res_dash.json()['dashboard']
assert dash['total_records'] == 10000
assert dash['total_columns'] == 16
assert dash['missing_values'] == 12028
print("[PASS] Dashboard summary provides exact real metrics!")

print("\n>>> ALL MASTER PROMPT TEST CASES PASSED PERFECTLY! <<<")
