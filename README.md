# AI-Drug-Interaction-System

An AI-powered Drug-Drug Interaction prediction system built with Python, Scikit-Learn (Random Forest), NumPy, Pandas, and Streamlit.

## Project Structure
- app.py : Main Streamlit frontend UI
- train_model.py : ML training pipeline (Random Forest)
- requirements.txt : Project dependencies
- README.md : Documentation
- drug_interactions.csv : Core dataset
- saved_models/ : Pre-trained .pkl model artifacts

## Setup & Installation
1. Clone the repository:
git clone [https://github.com/sandeep123-456/AI-Drug-Interaction-System.git](https://github.com/sandeep123-456/AI-Drug-Interaction-System.git)

2. Create virtual environment and install requirements:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3. Train the Model:
python3 train_model.py

4. Run the Dashboard:
streamlit run app.py

## Tech Stack
- Frontend: Streamlit + Custom CSS
- Model: Random Forest Classifier (Scikit-Learn)
- Data Processing: Pandas, NumPy