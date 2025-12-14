import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved model
try:
    data = joblib.load('rf_model.pkl')
    pipeline = data['pipeline']
except FileNotFoundError:
    st.error("Error: 'rf_model.pkl' not found. Please run 'train_model.py' first.")
    st.stop()

# ---------------------------------------------------------
# 2. DEFINE WORKOUT CATEGORIES
# ---------------------------------------------------------
workout_map = {
    "Cardio": ["Running", "Walking", "Cycling", "Swimming", "Elliptical", "Stair Machine", "Jump Rope"],
    "Strength": ["Weightlifting", "Bodyweight Exercises", "Circuit Training", "Crossfit", "Resistance Band"],
    "Flexibility": ["Yoga", "Pilates", "Stretching"],
    "HIIT": ["Burpees", "Sprinting", "Tabata", "High Intensity Interval"]
}

# ---------------------------------------------------------
# 3. APP INTERFACE
# ---------------------------------------------------------
st.set_page_config(page_title="Calorie Predictor", page_icon="💪", layout="centered")
st.title("💪 Calorie Burn & Fat Loss Calculator")

with st.form("main_form"):
    
    # --- ROW 1: PERSONAL STATS ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gender = st.selectbox("Gender", ["male", "female"])
    with c2:
        age = st.number_input("Age (years)", 10, 100, 25)
    with c3:
        height = st.number_input("Height (cm)", 100, 250, 170)
    with c4:
        weight = st.number_input("Weight (kg)", 30, 200, 70)

    # --- ROW 2: WORKOUT SELECTION ---
    c5, c6 = st.columns(2)
    
    with c5:
        category = st.selectbox("Workout Category", list(workout_map.keys()) + ["Other / Type My Own"])
        
    with c6:
        final_workout_name = ""
        
        if category == "Other / Type My Own":
            final_workout_name = st.text_input("Specific Workout Name", placeholder="e.g. Rock Climbing")
        else:
            specific_options = workout_map[category]
            selection = st.selectbox("Specific Exercise", specific_options + ["Not Listed"])
            
            if selection == "Not Listed":
                final_workout_name = st.text_input("Type Exercise Name", placeholder="e.g. Trail Running")
            else:
                final_workout_name = selection

    # --- ROW 3: INTENSITY METRICS ---
    c7, c8, c9 = st.columns(3)
    with c7:
        # CHANGED: Duration is now a number input box instead of a slider
        duration = st.number_input("Duration (min)", min_value=1, max_value=300, value=45)
    with c8:
        heart_rate = st.slider("Heart Rate (bpm)", 60, 220, 110)
    with c9:
        body_temp = st.slider("Body Temp (°C)", 36.0, 42.0, 39.5)

    # --- SUBMIT BUTTON ---
    submitted = st.form_submit_button("Calculate Burn", type="primary")

# ---------------------------------------------------------
# 4. PREDICTION & OUTPUT
# ---------------------------------------------------------
if submitted:
    if not final_workout_name:
        st.error("Please enter a workout name.")
    else:
        # Calculate BMI
        bmi = weight / ((height / 100) ** 2)
        
        # Prepare Input
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Workout_Type': [final_workout_name],
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'Duration': [duration],
            'Heart_Rate': [heart_rate],
            'Body_Temp': [body_temp],
            'BMI': [bmi]
        })
        
        # Predict
        calories = pipeline.predict(input_data)[0]
        fat_loss_g = (calories / 7700) * 1000
        
        # Display Results
        st.markdown("---")
        res1, res2 = st.columns(2)
        
        res1.metric("🔥 Calories Burned", f"{calories:.0f} kcal")
        res2.metric("⚖️ Fat Loss", f"{fat_loss_g:.1f} g")
        
        if fat_loss_g > 0:
            st.success(f"Great job! That's equivalent to burning **{fat_loss_g:.1f} grams** of pure body fat.")