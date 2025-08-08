# Local Data Science App with ML & Optimization

A modular local Streamlit application for data upload, analysis, visualization, feature optimization, and no-code machine learning. Designed for ease of use without cloud dependencies or API keys.

---

## Features

- 📁 Upload structured data files (CSV, Excel) with automatic type detection and metadata generation.
- 📊 Interactive data analysis and visualization (pie charts, correlation heatmaps, grouped numeric summaries).
- 🤖 Machine learning interface supporting regression and classification with automatic encoding and model selection.
- ⚡ Feature optimization module to detect and remove redundant (highly correlated) features.
- 🔄 Utility functions to support type detection, encoding, and redundancy analysis.
- 💾 Model saving and loading utilities for seamless ML model management.

---

## Project Structure

your_project/
│
├── data_upload.py # Data upload and preprocessing module
├── data_visualize.py # Interactive data visualization module
├── machine_learning.py # No-code ML training and evaluation module
├── modelUtils.py # Model saving/loading utilities
├── optimization.py # Feature optimization module (redundancy detection)
├── utils.py # Utility functions (type detection, encoding, etc.)
└── data/ # Data storage directory (created automatically)
├── main_data.csv
├── metadata/
│ └── column_type_desc.csv
└── models/
└── (saved models)

text

---

## Setup Instructions

1. **Install Python 3.8+ and set up a virtual environment (recommended).**

2. **Install dependencies:**

pip install -r requirements.txt

text

3. **Run the app modules with Streamlit:**

For example, to run the data upload page:

streamlit run data_upload.py

text

You can similarly run other modules by replacing the filename.

4. **Data Flow:**

- Upload your CSV/XLSX dataset on the Upload page.
- Proceed to Data Analysis to explore and visualize the dataset.
- Use the Feature Optimization page to prune redundant features.
- Move to Machine Learning to train, evaluate, and save predictive models.

---

## Notes

- All data and models are stored locally in the `data/` folder.
- No authentication or API keys are required.
- Designed for simple, private, offline workflows.
- The app uses scikit-learn for modeling.
- Visualizations use matplotlib and seaborn.
- Utility functions encapsulate common data preprocessing tasks.

---

## Requirements

See `requirements.txt` for detailed Python package dependencies.

---


Additional Notes:
Your provided code handles files, visualizations, ML, and optimizations fully locally without external APIs.

The requirements here cover Streamlit UI, data handling, plotting, ML, model persistence.

You can extend the requirements if you add more libraries later.

Run pip install -r requirements.txt to install all dependencies before launching the app.

If you want, I can help you write a very simple main.py integrating these modules for a unified Streamlit app with a sidebar navigation — just let me know!