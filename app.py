# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
import os
import shutil
import json

# Import your original main.py (no changes needed)
from main import ProductivityAnalyzer

app = Flask(__name__)
app.secret_key = 'analysis_secret_key_2025'

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
OUTPUT_FOLDER = 'output'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Clear old static plots
for f in os.listdir(STATIC_FOLDER):
    if f.endswith('.png'):
        try:
            os.remove(os.path.join(STATIC_FOLDER, f))
        except:
            pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'data_file' not in request.files:
        flash('No file selected.')
        return redirect(url_for('index'))

    file = request.files['data_file']
    if file.filename == '':
        flash('No file selected.')
        return redirect(url_for('index'))

    if not file.filename.lower().endswith('.csv'):
        flash('Only CSV files are allowed.')
        return redirect(url_for('index'))

    # Save uploaded file with a generic name
    filepath = os.path.join(UPLOAD_FOLDER, 'analysis_data.csv')
    file.save(filepath)

    try:
        analyzer = ProductivityAnalyzer(
            data_file=filepath,
            output_dir=OUTPUT_FOLDER
        )
        results = analyzer.run_complete_analysis()

        if not results:
            flash("Analysis failed. Check logs.")
            return redirect(url_for('index'))

        # Copy generated plots to static/ for HTML
        plot_files = [
            'descriptive_analysis.png',
            'pca_analysis.png',
            'simple_regression_analysis.png',
            'multiple_regression_analysis.png'
        ]
        for plot in plot_files:
            src = os.path.join(OUTPUT_FOLDER, plot)
            dst = os.path.join(STATIC_FOLDER, plot)
            if os.path.exists(src):
                shutil.copy(src, dst)

        # Load JSON results for display
        results_json = {}
        json_path = os.path.join(OUTPUT_FOLDER, 'analysis_results.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                results_json = json.load(f)

        return render_template('results.html', results=results_json)

    except Exception as e:
        flash(f"Error: {str(e)}")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)