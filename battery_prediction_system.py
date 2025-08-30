import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🔋 Battery Health Checker",
    page_icon="🔋",
    layout="centered"
)

class BetterBatteryPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.trained = False
        self.data = None

    def generate_data(self, samples=2000):
        np.random.seed(42)
        age_months = np.random.randint(0, 60, samples)   
        daily_hours = np.random.uniform(2, 16, samples)
        charge_cycles = age_months * daily_hours / 8 + np.random.normal(0, 40, samples)
        charge_cycles = np.clip(charge_cycles, 0, 2500)
        temp_category = np.random.choice([0, 1, 2], samples, p=[0.2, 0.6, 0.2])

        
        health = 100.0
        
        age_loss = 15 * (1 - np.exp(-age_months / 20))
        
        usage_loss = (daily_hours / 16) * (age_months / 60) * 25
        
        cycle_loss = (charge_cycles / 500) * 10
        
        temp_loss = temp_category * (age_months / 12) * 2.5

        battery_health = health - age_loss - usage_loss - cycle_loss - temp_loss
        battery_health = np.clip(battery_health, 10, 100)
        battery_health += np.random.normal(0, 3, samples)  # small noise
        battery_health = np.clip(battery_health, 10, 100)

        data = pd.DataFrame({
            'age_months': age_months,
            'daily_hours': np.round(daily_hours, 1),
            'charge_cycles': np.round(charge_cycles).astype(int),
            'temp_category': temp_category,
            'battery_health': np.round(battery_health, 1),
            'created_date': datetime.now()
        })
        self.data = data
        return data

    def train_model(self, data):
        X = data[['age_months', 'daily_hours', 'charge_cycles', 'temp_category']]
        y = data['battery_health']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.trained = True
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        return r2, mae

    def predict(self, age_months, daily_hours, charge_cycles, temp_category):
        if not self.trained:
            return None
        prediction = self.model.predict([[age_months, daily_hours, charge_cycles, temp_category]])[0]
        return max(10, min(100, prediction))

    def get_recommendation(self, health):
        if health >= 85:
            return "✅ **EXCELLENT** - Your battery is in great shape!", "green"
        elif health >= 70:
            return "👍 **GOOD** - Battery is performing well. Monitor regularly.", "blue"
        elif health >= 50:
            return "⚠️ **FAIR** - Consider replacing in 6-12 months.", "orange"
        elif health >= 30:
            return "❗ **POOR** - Battery replacement recommended soon.", "red"
        else:
            return "🚨 **CRITICAL** - Replace battery immediately.", "red"


# Streamlit App
if 'predictor' not in st.session_state:
    predictor = BetterBatteryPredictor()
    data = predictor.generate_data(2000)
    r2, mae = predictor.train_model(data)
    st.session_state['predictor'] = predictor
    st.session_state['r2'] = r2
    st.session_state['mae'] = mae

predictor = st.session_state['predictor']

st.title("🔋 Laptop Battery Health Checker")
st.markdown("**Improved model with more accurate predictions**")

tab1, tab2 = st.tabs(["🔮 Check Battery", "📈 Statistics"])


with tab1:
    st.header("Check Your Battery Health")
    col1, col2 = st.columns(2)
    with col1:
        age_months = st.number_input("Laptop Age (months)", 0, 72, 18)
        daily_hours = st.slider("Daily usage (hours)", 1.0, 16.0, 8.0, 0.5)
    with col2:
        charge_cycles = st.number_input("Estimated charge cycles", 0, 2500, int(age_months * daily_hours / 8))
        temp_options = {"Cool (AC, well-ventilated)": 0, "Normal (room temp)": 1, "Hot (gaming/heavy use)": 2}
        temp_category = st.selectbox("Operating temperature", options=list(temp_options.keys()))

    if st.button("🔮 Check Battery Health", type="primary", use_container_width=True):
        temp_val = temp_options[temp_category]
        prediction = predictor.predict(age_months, daily_hours, charge_cycles, temp_val)
        if prediction:
            st.subheader("🎯 Your Battery Health")
            st.metric("Battery Health", f"{prediction:.1f}%")
            recommendation, color = predictor.get_recommendation(prediction)
            if color == "green":
                st.success(recommendation)
            elif color == "blue":
                st.info(recommendation)
            elif color == "orange":
                st.warning(recommendation)
            else:
                st.error(recommendation)


with tab2:
    st.header("Battery Health Statistics")
    data = predictor.data
    st.metric("Model Accuracy (R²)", f"{st.session_state['r2']:.3f}")
    st.metric("Average Error", f"{st.session_state['mae']:.1f}%")
    st.subheader("📊 Dataset Overview")
    st.dataframe(data.head())
    fig = px.histogram(data, x='battery_health', nbins=25, title="Battery Health Distribution")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.scatter(data, x="age_months", y="battery_health", color="temp_category",
                      title="Battery Health vs Laptop Age")
    st.plotly_chart(fig2, use_container_width=True)
