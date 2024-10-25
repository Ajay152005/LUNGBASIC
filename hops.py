import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import random
import os

# Load the dataset
data = pd.read_excel(r'data/ML(PRO).xlsx')

# Preprocessing: Drop unused columns and target
X = data.drop(['target', 'max_hr'], axis=1)  # Removing max_hr
y = data['target']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


# Prediction function
def predict_heart_disease(input_data):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[:, 1][0]
    return prediction, prediction_proba


# Function to generate random responses with a name
def random_response(name, options):
    return random.choice(options).format(name=name)


# Prepare output Excel file with the Name column
output_file = "predicted_patients.xlsx"
if not os.path.exists(output_file):
    pd.DataFrame(columns=['Name', 'Age', 'Sex', 'Chest Pain Type', 'Resting BP',
                          'Rest ECG', 'Slope', 'Prediction', 'Probability']).to_excel(output_file, index=False)

# Main loop to get user input
while True:
    print("\n--- Enter Patient Details ---")
    print()

    # Get the patient's name
    name = input("Please enter your name (or (q) to quit): ").strip()
    if name.lower() == 'q':
        print("Exiting the program. Goodbye!")
        break

    # Get other inputs with validation
    try:
        age = int(input(f"Hi {name}, please enter your age: "))

        if age < 18:
            print(random_response(name, [
                "You're too young for this test, {name}. Stay healthy and happy!",
                "Enjoy your youth, {name}. You're in good shape!"
            ]))
        elif 18 <= age <= 30:
            print(random_response(name, [
                "Good to see you taking care of your health, {name}!",
                "You're young and proactive, {name}. Keep it up!"
            ]))
        elif 31 <= age <= 49:
            print(random_response(name, [
                "It's a great time to focus on your health, {name}. Keep going!",
                "Never too late to keep an eye on your well-being, {name}!"
            ]))
        else:
            print(random_response(name, [
                "You're doing the right thing by taking care of your health, {name}.",
                "Stay strong, {name}! A little care goes a long way."
            ]))

        print()
        sex = int(input(f"Enter your sex, {name} (1 for male, 0 for female): "))
        if sex not in [0, 1]:
            raise ValueError("Sex must be 0 or 1.")

        sex_response = {
            1: [
                "Good to see you taking charge of your health, Mr. {name}!",
                "Stay active and fit, Mr. {name}!"
            ],
            0: [
                "You're doing great, Ms. {name}!",
                "Your health is your superpower, Ms. {name}!"
            ]
        }
        print(random_response(name, sex_response[sex]))

        print()
        chest_pain_type = int(input("Enter your chest pain type (0-3): "))
        if not (0 <= chest_pain_type <= 3):
            raise ValueError("Chest pain type must be between 0 and 3.")
        print(random_response(name, [
            "Thanks for sharing the chest pain details, {name}. Monitoring it is key!",
            "Let's keep an eye on your chest health, {name}."
        ]))

        print()
        resting_bp = float(input("Enter your resting blood pressure: "))
        print(random_response(name, [
            f"Resting BP of {resting_bp} noted, {name}. Keep monitoring it!",
            f"BP logged, {name}. Managing it is essential to staying healthy!"
        ]))

        print()
        restecg = int(input("Enter your resting ECG result (0-2): "))
        if not (0 <= restecg <= 2):
            raise ValueError("Resting ECG result must be between 0 and 2.")
        print(random_response(name, [
            "Thanks for the ECG input, {name}. Every detail matters!",
            "ECG logged successfully, {name}."
        ]))

        slope = 1  # Default slope value

    except ValueError as ve:
        print(f"Invalid input: {ve}")
        continue

    # Collect input data into a dictionary
    user_input = {
        'age': age,
        'sex': sex,
        'chest_pain_type': chest_pain_type,
        'resting_bp': resting_bp,
        'restecg': restecg,
        'slope': slope
    }

    # Make prediction
    result, probability = predict_heart_disease(user_input)
    print()
    print(f"Heart Disease Prediction for {name}: {'YES' if result == 1 else 'NO'}")

    prob_message = random_response(name, [
        f"The probability of heart disease is {probability * 100:.2f}%, {name}.",
        f"Estimated risk is {probability * 100:.2f}%, {name}."
    ])
    print(prob_message)

    # Save data to Excel
    new_data = pd.DataFrame([{
        'Name': name,
        'Age': age,
        'Sex': sex,
        'Chest Pain Type': chest_pain_type,
        'Resting BP': resting_bp,
        'Rest ECG': restecg,
        'Slope': slope,
        'Prediction': 'YES' if result == 1 else 'NO',
        'Probability': f"{probability * 100:.2f}%"
    }])

    # Append to the existing Excel file
    existing_data = pd.read_excel(output_file)
    updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    updated_data.to_excel(output_file, index=False)

    print(f"Patient data for {name} saved successfully.")