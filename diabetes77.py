import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pickle
import streamlit as st

# Load data
data = pd.read_csv("diabetes (1).csv")

# Data exploration
data.head()
data.describe()
data.info()
data.isna().sum()
data.duplicated().sum()

# Data visualization
plt.figure(figsize=(12, 6))
sns.countplot(x='Outcome', data=data)
plt.show()

plt.figure(figsize=(12, 12))
for i, col in enumerate(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x=col, data=data)
plt.show()

sns.pairplot(hue='Outcome', data=data)
plt.show()

plt.figure(figsize=(12, 12))
for i, col in enumerate(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']):
    plt.subplot(3, 3, i + 1)
    sns.histplot(x=col, data=data, kde=True)
plt.show()

plt.figure(figsize=(12, 12))
sns.heatmap(data.corr(), vmin=-1.0, center=0, cmap='RdBu_r', annot=True)
plt.show()

# Data preprocessing
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(data.drop(["Outcome"], axis=1)))
y = data['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Model training
knn = KNeighborsClassifier(13)
knn.fit(X_train, y_train)

# Model evaluation
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model
with open('classifier.pkl', 'wb') as f:
    pickle.dump(knn, f)

# Load the trained model
with open('classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app
def preprocess_input(user_input):
    df = pd.DataFrame(user_input, index=[0])
    df = scaler.transform(df)
    return df

def main():
    st.title("Diabetes Prediction App")
    st.subheader("Enter your health information:")

    pregnancies = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0)
    insulin = st.number_input("Insulin (μU/mL)", min_value=0)
    bmi = st.number_input("BMI (kg/m²)", min_value=0.0)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function")
    age = st.number_input("Age (years)", min_value=0)

    if st.button("Predict"):
        user_data = {'Pregnancies': pregnancies,
                     'Glucose': glucose,
                     'BloodPressure': blood_pressure,
                     'SkinThickness': skin_thickness,
                     'Insulin': insulin,
                     'BMI': bmi,
                     'DiabetesPedigreeFunction': diabetes_pedigree_function,
                     'Age': age}

        user_input_df = preprocess_input(user_data)

        try:
            prediction = model.predict(user_input_df)
            if prediction[0] == 1:
                st.write("**You are at risk for diabetes.**")
                st.warning("Please consult a healthcare professional for further evaluation.")
            else:
                st.success("**You are likely not at high risk for diabetes based on this prediction.**")
                st.info("However, it's important to maintain healthy habits and schedule regular checkups with your doctor.")
        except Exception as e:
            st.error(f"An error occurred while making prediction: {str(e)}")

if __name__ == '__main__':
    main()
