# Disease Prediction AI System

This project is an end-to-end AI system for predicting disease risk from patient records using machine learning. It includes:
- A trained machine learning model (Random Forest)
- A FastAPI web API for serving predictions
- Instructions for integration with a mobile app

## Features
- Predicts disease risk based on patient data (age, gender, blood pressure, cholesterol, glucose, BMI, family history, smoking)
- REST API endpoint for real-time predictions
- Ready for integration with Android, iOS, or web/mobile apps

## Getting Started

### 1. Clone the Repository
```
git clone https://github.com/your-username/diseas_prediction_api.git
cd diseas_prediction_api
```

### 2. Install Requirements
```
pip install -r requirements.txt
```

### 3. Train the Model (Optional)
Run the script to train and save the model:
```
python "supervised learning.py"
```

### 4. Run the API Locally
```
uvicorn api:app --reload
```
The API will be available at http://127.0.0.1:8000

### 5. API Usage
- Visit http://127.0.0.1:8000/docs for interactive API documentation (Swagger UI).
- Use the `/predict` endpoint with a POST request and JSON body:
```json
{
  "age": 45,
  "gender": 1,
  "blood_pressure": 120,
  "cholesterol": 200,
  "glucose": 130,
  "bmi": 28.5,
  "family_history": 1,
  "smoking": 0
}
```

### 6. Deploying Online
- Deploy the API to a cloud platform (e.g., Render, Heroku, Azure) for public access.

### 7. Mobile App Integration
- Connect your mobile app to the API using HTTP POST requests.
- Example code for React Native (Expo Snack) and Android is provided in the documentation.

## Project Structure
```
├── api.py                  # FastAPI app
├── supervised learning.py  # Model training script
├── requirements.txt        # Python dependencies
├── disease_model.joblib    # Saved trained model
```

## License
This project is for educational purposes. Feel free to use and modify it for your own learning or projects.

---

**Author:** [Your Name]

For questions or contributions, please open an issue or pull request.
