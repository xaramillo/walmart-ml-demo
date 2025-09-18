
import os
import pandas as pd
import mlflow
from functools import reduce


def etl_processing(raw_dir='data/raw', output_path='data/dataset.csv', dataset_version='v1'):
	"""
	Uses PdM_telemetry.csv as the base and joins with PdM_failures.csv on 'datetime' and 'machineID'.
	Fills missing values in 'failure' with 'normal'.
	Saves the result and logs it in MLflow.
	"""
	telemetry_path = os.path.join(raw_dir, 'PdM_telemetry.csv')
	failures_path = os.path.join(raw_dir, 'PdM_failures.csv')

	df_telemetry = pd.read_csv(telemetry_path)
	df_failures = pd.read_csv(failures_path)

	# Join on 'datetime' and 'machineID'
	df = pd.merge(df_telemetry, df_failures, on=['datetime', 'machineID'], how='left')
	df['failure'] = df['failure'].fillna('normal')

	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	df.to_csv(output_path, index=False)

	# Log in MLflow
	mlflow.set_experiment("dataset_unification")
	with mlflow.start_run(run_name="unify_telemetry_failures"):
		mlflow.log_param("input_files", ['PdM_telemetry.csv', 'PdM_failures.csv'])
		mlflow.log_param("output_dataset", os.path.basename(output_path))
		mlflow.log_param("dataset_version", dataset_version)
		mlflow.log_artifact(output_path)

# If run as a script, execute the function
if __name__ == "__main__":
	etl_processing()
