# Healthcare-Fraud-Detection
Python pipeline for healthcare claims—data generation, ETL to processed features, Random Forest / logistic regression training, REST API for scores &amp; OpenAI explanations, and a Streamlit console for reviewers.
# 🏥 AI-Powered Healthcare Fraud Detection Platform

An end-to-end **production-grade data science system** designed to detect fraudulent healthcare claims using **machine learning, advanced analytics, and GenAI-powered explanations**.

---

## 🚀 Overview

Healthcare fraud costs billions annually. This project builds a **scalable fraud detection platform** that:

- Detects suspicious claims using ML models  
- Identifies abnormal provider & patient behavior  
- Generates **AI-powered explanations** for transparency  
- Provides real-time predictions via API & UI  

---

## 💡 Key Features

### 🔍 Fraud Detection Engine
- Machine learning models (Logistic Regression, Random Forest)
- Fraud probability scoring
- High-recall detection strategy

### 🧠 Feature Engineering
- Provider-level risk metrics  
- Patient behavioral patterns  
- Claim anomaly detection (deviation, frequency)  
- Time-based fraud signals  

### 🤖 GenAI Explainability (🔥 Unique)
- OpenAI-powered explanations  
- Converts model output → human-readable insights  
- Business-friendly fraud reasoning  

### 🔄 Data Pipeline (ETL)
- Synthetic healthcare data generation  
- Data cleaning, validation, transformation  
- Modular and scalable pipeline  

### 🌐 API Layer
- FastAPI-based REST service  
- Real-time fraud prediction  
- JSON-based request/response  

### 🖥️ Interactive App
- Streamlit UI for live fraud detection  
- Input claim → get score + explanation  
- Clean and simple interface  

---

## 🖼️ Application Preview

### 🌐 API Interface (FastAPI Docs)
Interactive API documentation for testing fraud prediction and explanation endpoints.

<img width="1792" height="978" alt="Screenshot 2026-04-01 at 9 34 53 PM" src="https://github.com/user-attachments/assets/206f8f4e-64cf-4dbf-a648-e59b080d190d" />


---

### 🧠 Fraud Risk Assessment Panel
Input claim details and advanced billing signals to evaluate fraud probability.

<img width="1496" height="976" alt="Screenshot 2026-04-01 at 9 40 35 PM" src="https://github.com/user-attachments/assets/8aec5bd0-3444-439f-9be8-04a8c6fa8b9b" />



---

### 📊 Fraud Risk Output
Real-time fraud score with clear risk classification and insights.

<img width="1799" height="979" alt="Screenshot 2026-04-01 at 9 41 54 PM" src="https://github.com/user-attachments/assets/d594211a-841d-40a9-9ac0-6af837e8dc1e" />


---

## 🏗️ Architecture

Raw Data → ETL Pipeline → Feature Engineering → ML Model → GenAI Explanation → API → Streamlit UI

---

## 🛠️ Tech Stack

**Languages & Libraries**
- Python (pandas, NumPy, scikit-learn)
**Data Engineering**
- ETL Pipelines  
- Data Validation  
- Feature Engineering  
**Machine Learning**
- Logistic Regression  
- Random Forest  
- Model Evaluation (Precision, Recall, F1, ROC-AUC)
**GenAI**
- OpenAI API  
- Prompt Engineering  
- Explainable AI  
**Backend**
- FastAPI  
- REST APIs  
**Frontend**
- Streamlit  
**Visualization**
- Power BI (optional dashboards)

---

## 📊 Sample Output
Fraud Score: 0.87 (High Risk)
Explanation:
This claim is flagged as high risk because the billing amount is significantly higher than the provider's average, and multiple claims were submitted within a short time window, indicating abnormal activity.

---


### 🖥️ Run Streamlit App

streamlit run src/healthcare_fraud_detection/app_streamlit/app.py

### 🔐 Environment Variables

Create .env file:

OPENAI_API_KEY=your_api_key_here

### 💼 Business Value

Detects fraudulent healthcare claims early
Reduces financial losses for insurers
Improves audit efficiency
Provides explainable AI for decision-making

### 📈 Future Improvements

Real-time streaming (Kafka)
Graph-based fraud detection
Cloud deployment (AWS/GCP)
Role-based dashboards
Integration with real healthcare datasets

### 👨‍💻 Author

Sathvik Putta
Data Analyst | Data Science Enthusiast

⭐ If you like this project

Give it a ⭐ on GitHub and feel free to connect!


---


