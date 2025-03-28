from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import pandas as pd
import os
import uuid
import logging
import stat
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Needed for flash messages

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash("No file uploaded", "danger")
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash("No selected file", "danger")
        return redirect(url_for('index'))

    column_names = request.form.get('column_names', '')  
    categorical_columns = [col.strip() for col in column_names.split(',') if col.strip()]  

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    df = pd.read_csv(filepath)
    df[categorical_columns] = df[categorical_columns].astype(str)

    # Metadata detection
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    # Train model
    try:
        logging.info("Training CTGAN model...")
        model = CTGANSynthesizer(metadata)
        model.fit(df)
        model_path = os.path.join(OUTPUT_FOLDER, "ctgan_model.pkl")
        model.save(model_path)
        logging.info("Model training completed and saved.")
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        flash(f"Error training model: {str(e)}", "danger")
        return redirect(url_for('index'))

    flash("File uploaded and model trained successfully!", "success")
    return redirect(url_for('generate_page'))

@app.route('/manual_entry', methods=['GET', 'POST'])
def manual_entry():
    if request.method == 'POST':
        try:
            num_columns = int(request.form.get('num_columns', 0))
            num_rows = int(request.form.get('num_rows', 0))
            column_names = request.form.getlist('column_names') or request.form.get('column_names', '').split(',')

            if num_columns != len(column_names) or not column_names:
                flash("Error: Column names are missing or do not match the column count.", "danger")
                return redirect(url_for('manual_entry'))

            return render_template('manual_entry.html', num_columns=num_columns, column_names=column_names, num_rows=num_rows)
        
        except Exception as e:
            logging.error(f"Error processing manual entry form: {str(e)}")
            flash("Invalid input format!", "danger")
            return redirect(url_for('manual_entry'))
    
    return render_template('manual_entry.html')

@app.route('/submit_manual_data', methods=['POST'])
def submit_manual_data():
    try:
        num_rows = int(request.form.get('num_rows', 0))
        column_names = request.form.getlist('column_names')

        if not column_names or all(name.strip() == "" for name in column_names):
            flash("Error: Column names are missing or empty!", "danger")
            return redirect(url_for('manual_entry'))

        data = []
        for i in range(num_rows):
            row = [request.form.get(f'row_{i}_{j}', '').strip() for j in range(len(column_names))]
            if any(cell != "" for cell in row):
                data.append(row)

        if not data:
            flash("Error: No valid data entered! Please fill at least one row.", "danger")
            return redirect(url_for('manual_entry'))

        df = pd.DataFrame(data, columns=column_names)
        csv_path = os.path.join(UPLOAD_FOLDER, "manual_data.csv")
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved manual data to: {csv_path}")

        # Train Model
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        logging.info("Training CTGAN model with manual data...")
        model = CTGANSynthesizer(metadata)
        model.fit(df)
        model_path = os.path.join(OUTPUT_FOLDER, "ctgan_model.pkl")
        model.save(model_path)
        logging.info("Model training completed with manual data.")

        flash("Data successfully saved and model trained!", "success")
        return redirect(url_for('generate_page'))

    except Exception as e:
        logging.error(f"Error processing manual data: {str(e)}")
        flash(f"Error processing manual data: {str(e)}", "danger")
        return redirect(url_for('manual_entry'))

@app.route('/generate_page')
def generate_page():
    return render_template('generate.html')

@app.route('/generate', methods=['POST'])
def generate_data():
    try:
        num_rows = int(request.form['num_rows'])
        if num_rows <= 0:
            flash("Error: Number of rows must be greater than zero.", "danger")
            return redirect(url_for('generate_page'))

        model_path = os.path.join(OUTPUT_FOLDER, "ctgan_model.pkl")
        if not os.path.exists(model_path):
            flash("Error: Model file not found. Please train the model first.", "danger")
            return redirect(url_for('generate_page'))

        logging.info("Loading trained CTGAN model...")
        model = CTGANSynthesizer.load(model_path)

        logging.info(f"Generating {num_rows} rows of synthetic data...")
        synthetic_data = model.sample(num_rows)

        if synthetic_data.shape[0] == 0:
            flash("Error: Generated data is empty. Try training again with a different dataset.", "danger")
            return redirect(url_for('generate_page'))

        output_file = os.path.join(OUTPUT_FOLDER, f"synthetic_data_{uuid.uuid4().hex}.csv")
        synthetic_data.to_csv(output_file, index=False)
        logging.info(f"Synthetic data generated successfully: {output_file}")

        return send_file(output_file, as_attachment=True)

    except Exception as e:
        logging.error(f"Error generating data: {str(e)}")
        flash(f"Error generating data: {str(e)}", "danger")
        return redirect(url_for('generate_page'))

if __name__ == '__main__':
    app.run(debug=True)
