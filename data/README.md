# Dataset Source and Download

This project uses the Telco Customer Churn CSV published in IBM's
telco-customer-churn-on-icp4d sample repository.

- Source repository: https://github.com/IBM/telco-customer-churn-on-icp4d
- Source CSV: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- Repository license: https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/LICENSE

The source repository contains an Apache-2.0 license. This records the
repository-level license; a separate dataset-specific license has not
been verified. The raw CSV is not redistributed in this project's
repository or source ZIP. Consult the upstream source for applicable terms.

## Download

After installing the project dependencies, run:

```powershell
Set-Location "C:\Mydata\Interns\Enliven solutions LLC\project 6\customer-churn-prediction-system"

.\.venv\Scripts\python.exe -m src.download_data
```

The command saves the CSV to data/raw/Telco-Customer-Churn.csv.

It verifies the SHA-256 fingerprint before writing a downloaded file.
An existing file is checked and preserved. A mismatch stops execution.

Expected SHA-256:
16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91

The source URL references a branch. The checksum ensures that a changed
upstream file will not silently replace the dataset used for these results.

## Data handling

- Keep the raw CSV unchanged.
- Generate training/test files using python -m src.split.
- Processed data is excluded from Git.
- Customer IDs are excluded from model inputs.
- The saved Streamlit application does not require the raw CSV.