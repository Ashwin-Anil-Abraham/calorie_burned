import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import joblib

# Load Data

cal = pd.read_csv('calories.csv')
exer = pd.read_csv('exercise.csv')
df = exer.merge(cal, on='User_ID')

# Add BMI

df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)

#Split (X) and(y)

X = df.drop(['User_ID', 'Calories'], axis=1)
y = df['Calories']

#Define columns which are Categorical and Numerical

categorical_cols = ['Gender', 'Workout_Type'] 
numerical_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']

# now Preprocessing
preprocessor = ColumnTransformer(transformers=[
  ('num', StandardScaler(), numerical_cols),
  ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])

# 6. Create the Pipeline (Preprocessor + Model)
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
     ])

# 7. Train and Predict
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now model fitting

my_pipeline.fit(X_train, y_train)
y_pred = my_pipeline.predict(X_test)


print("R2 Score:", r2_score(y_test, y_pred))


# We save the unique workout types so the app knows what to put in the dropdown
workout_options = sorted(X['Workout_Type'].unique().tolist())

model_data = {
    'pipeline': my_pipeline,
    'workout_options': workout_options
}

joblib.dump(model_data, 'rf_model.pkl')
print("Model saved as 'rf_model.pkl'")