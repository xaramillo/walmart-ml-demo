
# walmart-ml-demo
Implementation for Code Challenge.

## Description

This repo contains a complete and modular pipeline for the development, training, evaluation, interpretation, and deployment of machine learning models, with a focus on engineering best practices and experiment traceability using MLflow. It includes scripts for preprocessing, training, evaluation, interpretability , and Streamlit application.

---

## Installation and Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/xaramillo/walmart-ml-demo.git
cd walmart-ml-demo
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the data

Run the script `kaggle_data_download.sh` and make sure the CSV files are extracted in `data/raw/`.

---

## Pipeline Execution

### Training and Evaluation

```bash
python main.py
```

### Experiment Tracking

The pipeline uses [MLflow](https://mlflow.org/) for tracking and versioning. To launch the MLflow UI:

```bash
mlflow ui
```
Open the browser at [http://localhost:5000](http://localhost:5000).

---

## Deployment and Prediction with Streamlit

### 1. Train and save the models (`main.py` saves them in `models/`).

### 2. Launch the Streamlit app locally

```bash
streamlit run app/streamlit_serve.py
```

### 3. Usage with Docker

Build the image:

```bash
docker build -t walmart-ml-demo .
```

Run the app:

```bash
docker run -p 8501:8501 walmart-ml-demo bash run_streamlit.sh
```

Access [http://localhost:8501](http://localhost:8501).

---

## Contributions

Contributions are welcome! Open an issue or pull request for suggestions, improvements, or questions.

---

## License

This project is for educational and training purposes.