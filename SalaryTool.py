# %%
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the cleaned dataset
file_path = "cleaned_kaggle_data.csv"
data = pd.read_csv(file_path)

# Clean the Salary Column
def clean_salary(salary):
    if isinstance(salary, str):
        salary = salary.replace("$", "").replace(",", "").strip()  # Remove "$" and ","
        if "-" in salary:  # If it's a range like '10,000-14,999'
            lower, upper = salary.split("-")
            return (int(lower) + int(upper)) / 2  # Return the midpoint
        elif salary.isdigit():  # If it's a single value like '10000'
            return int(salary)
    return np.nan  # If it's invalid

# Apply cleaning and drop invalid rows
data["Salary"] = data["Salary"].apply(clean_salary)
data = data.dropna(subset=["Salary"])

# Initialize a random generator for reproducibility
rng = np.random.default_rng(seed=42)

# Mock Primary Programming Language column if it doesn't exist
if "Primary_Programming_Language" not in data.columns:
    data["Primary_Programming_Language"] = rng.choice(
        ["Python", "R", "SQL", "Java", "C++", "JavaScript", "Other"], size=len(data)
    )

# Mock "Title" column if it doesn't exist
if "Select_the_Title_Most_Similar_to_Your_Current_Role" not in data.columns:
    data["Select_the_Title_Most_Similar_to_Your_Current_Role"] = rng.choice(
        ["Data Scientist", "Machine Learning Engineer", "Data Analyst", "Software Engineer", "Other"], 
        size=len(data)
    )


# Encode categorical features
data = pd.get_dummies(data, columns=["Primary_Programming_Language", "Select_the_Title_Most_Similar_to_Your_Current_Role"], drop_first=True)

# Define relevant features
X = data[
    ["Experience", "Education"]
    + [col for col in data.columns if col.startswith("Primary_Programming_Language_")]
    + [col for col in data.columns if col.startswith("Select_the_Title_Most_Similar_to_Your_Current_Role_")]
]
y = data["Salary"]

# Train-test split with consistent random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model with consistent random state
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit app
st.title("Data Science Salary Predictor")

# Sliders and dropdowns for user input
experience = st.slider("Years of Experience", 0, 40, 5)  # Default 5 years
education = st.selectbox(
    "Highest Level of Education",
    ["No formal education past high school", "Some college/university study without earning a bachelor's degree",
     "Bachelor’s degree", "Master’s degree", "Doctoral degree", "Professional doctorate"]
)
primary_language = st.selectbox(
    "Primary Programming Language",
    ["Python", "R", "SQL", "Java", "C++", "JavaScript", "Other"]
)
current_role = st.selectbox(
    "Select the Title Most Similar to Your Current Role",
    ["Data Scientist", "Machine Learning Engineer", "Data Analyst", "Software Engineer", "Other"]
)

# Encode inputs for prediction
education_map = {
    "No formal education past high school": 1,
    "Some college/university study without earning a bachelor's degree": 2,
    "Bachelor’s degree": 3,
    "Master’s degree": 4,
    "Doctoral degree": 5,
    "Professional doctorate": 6
}
mapped_education = education_map[education]

# Create a feature array for prediction
input_data = pd.DataFrame({
    "Experience": [experience],
    "Education": [mapped_education],
    **{col: [1] if f"Primary_Programming_Language_{primary_language}" == col else [0] for col in X.columns if col.startswith("Primary_Programming_Language_")},
    **{col: [1] if f"Select_the_Title_Most_Similar_to_Your_Current_Role_{current_role}" == col else [0] for col in X.columns if col.startswith("Select_the_Title_Most_Similar_to_Your_Current_Role_")}
})

# Align input data with training features
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# Predict salary
if st.button("Predict"):
    predicted_salary = model.predict(input_data)[0]
    st.subheader(f"Predicted Salary: ${predicted_salary:,.2f}")


# %%
