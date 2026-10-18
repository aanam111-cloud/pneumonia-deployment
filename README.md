# Pneumonia Detection - Streamlit App

Model: ResNet50 (Transfer Learning)
Task: No Pneumonia / Pneumonia from chest X-ray

## Run in Codespaces
pip install -r requirements.txt
streamlit run app.py --server.port 8501

Open the forwarded port 8501 URL.

## Docker
docker build -t pneumonia-app .
docker run -p 8501:8501 pneumonia-app
