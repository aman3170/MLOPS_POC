# MLflow + DVC Project Template

A reproducible machine learning project setup using **MLflow** for experiment tracking and **DVC** for data and model versioning.

---

## Project Setup (macOS/Linux)

### 1. Install Prerequisites

# Install Homebrew (if not already installed)
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
# Install Python & Git
```
brew install python git
```
### 2. Create and Initialize the Project
```
mkdir mlflow-dvc-project && cd mlflow-dvc-project
git init
```
### 3. Create a Virtual Environment
```
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Required Packages
pip install mlflow dvc scikit-learn pandas matplotlib joblib

### 5. Initialize DVC
```
dvc init
git add .dvc .gitignore
git commit -m "Initialize DVC"
```
### 6. Add and Track Dataset
```
mkdir data
```
Move your dataset here (e.g., data.csv)
```
mv /path/to/data.csv data/
dvc add data/data.csv
git add data/data.csv.dvc .gitignore
git commit -m "Add dataset to DVC"
```
🧠 Model Training and MLflow Logging
### 8. Create the Training Script
```
mkdir src && touch src/train.py
src/train.py
```
python
```
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

mlflow.set_experiment("random-forest-experiment")

with mlflow.start_run():
    df = pd.read_csv("data/data.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    joblib.dump(model, "models/model.joblib")
```
### 9. Run the Script
```
python3 src/train.py
```
MLflow UI
### 10. Start MLflow UI
```
mlflow ui
```
Navigate to http://localhost:5000 to view experiment results.

🛠️ Build DVC Pipeline
### 11. Create params.yaml
```
train:
  n_estimators: 100
```
### 12. Create Pipeline with DVC
```
dvc run -n train_model \
  -d src/train.py -d data/data.csv \
  -p train.n_estimators \
  -o models/model.joblib \
  -m metrics.json \
  python3 src/train.py
git add dvc.yaml dvc.lock
git commit -m "Add DVC pipeline"
```
### 13. Reproduce Pipeline
```
dvc repro
```
### 14. Push Data and Model to Remote
```
dvc push
```
