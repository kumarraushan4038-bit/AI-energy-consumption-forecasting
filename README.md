# AI-Powered Energy Consumption Forecasting System using Machine Learning

## 📌 Project Overview
This project predicts future energy consumption using machine learning based on historical usage patterns, weather conditions, and time-based features.

It is designed as an **industry-oriented Data Science project** to demonstrate how AI can automate real-world forecasting tasks used in utilities, smart buildings, factories, and energy management systems.

---

## 🎯 Objective
To build a predictive system that can forecast hourly energy consumption and help organizations:

- optimize electricity usage
- reduce operational costs
- improve load planning
- support energy efficiency initiatives

---

## 🧠 Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## ⚙️ Workflow
1. Data generation / import
2. Data cleaning
3. Feature engineering
4. Exploratory Data Analysis (EDA)
5. Model training using Random Forest Regressor
6. Performance evaluation
7. Visualization of predictions

---

## 📊 Features Used
- Hour
- Day
- Month
- Day of Week
- Weekend Flag
- Temperature
- Humidity
- Previous Hour Consumption (Lag Feature)
- Rolling Mean Consumption

---

## 📈 Evaluation Metrics
- RMSE
- MAE
- R² Score

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ai-energy-consumption-forecasting.git
cd ai-energy-consumption-forecasting
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the project
```bash
python src/main.py
```

---

## 📷 Outputs
After running the project, the following files are generated:

- predictions.csv
- feature_importance.csv
- Energy consumption graphs
- Actual vs Predicted comparison chart
- Feature importance chart

---

## 💼 Business Value
This project demonstrates how machine learning can support:

- smart energy forecasting
- utility load planning
- cost optimization
- demand-side management
- anomaly detection (future enhancement)

---

## 🔮 Future Improvements
- Add LSTM / Deep Learning model
- Use real smart meter dataset
- Deploy with Streamlit dashboard
- Add weather API integration
- Build anomaly detection module

---

