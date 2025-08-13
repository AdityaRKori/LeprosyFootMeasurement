import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load saved objects for Clustering
try:
    kmeans = joblib.load("kmeans_model.joblib")
    scaler = joblib.load("input_scaler.joblib")
    label_encoders_clustering = joblib.load("label_encoders.joblib") # Renamed for clarity
    cluster_means = joblib.load("cluster_means.joblib")
    st.sidebar.success("Clustering models loaded successfully!")
except FileNotFoundError:
    st.sidebar.error("Clustering model files not found. Please check and try again.")
    # In a real Streamlit app, you might handle this more gracefully
    # st.stop() # Don't stop the app if clustering models are missing, allow classification to run if available

# Load saved objects for Classification
try:
    classifier_model = joblib.load("leprosy_classifier_model.joblib")
    label_encoders_classification = joblib.load("leprosy_label_encoders.joblib") # Load classification specific encoders
    classification_features_names = joblib.load("leprosy_classification_features.joblib") # Load feature names
    st.sidebar.success("Classification model loaded successfully!")
except FileNotFoundError:
     st.sidebar.error("Classification model files not found. Please check and try again.")
     # st.stop() # Don't stop the app if classification models are missing, allow clustering to run if available

# Define mappings for display labels and their underlying numerical values
# Assuming the numerical encoding (0, 1) corresponds to specific categories
display_mappings = {
    'Patient Gender': {0: 'Male', 1: 'Female'},
    'Diabeties': {0: 'No', 1: 'Yes'},
    'Grade': {0: 'Unknown', 1: 'Grade 1', 2: 'Grade 2'} # Updated Grade mapping for display
}

# Mapping from display label back to the numerical value for model input
# This needs to handle the new Grade options mapping back to the model's expected 0/1
display_to_numerical_mapping = {
    'Patient Gender': {'Male': 0, 'Female': 1},
    'Diabeties': {'No': 0, 'Yes': 1},
    'Grade': {'Unknown': 0, 'Grade 1': 1, 'Grade 2': 1} # Map Grade 1 and Grade 2 to the same numerical value (1) for the binary model
}


# Streamlit UI enhancements
st.set_page_config(layout="wide") # Use wide layout

st.title("Foot Measurement Estimator and Leprosy Risk Predictor")

st.markdown("""
This application has two main sections:
1.  **Foot Measurement Estimation:** Estimates foot measurements based on age, gender, diabetes, and leprosy grade using a clustering model.
2.  **Leprosy Risk Prediction:** Predicts the likelihood of having leprosy based on patient information and foot measurements using a classification model.
""")

# Add navigation or sections
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Foot Measurement Estimation", "Leprosy Risk Prediction"])


if page == "Foot Measurement Estimation":
    st.header("Foot Measurement Estimation")
    st.markdown("""
    Enter the patient's information below to get an estimated average foot measurement based on similar patient profiles.
    """)

    if 'kmeans' in locals(): # Check if clustering models were loaded
        # Create columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Patient Information for Estimation")
            # Collect user input for clustering features
            age_est = st.number_input("Enter Patient Age for Estimation", min_value=0, max_value=120, value=50, key='age_est')

            # Use display_mappings for selectbox options, but get the underlying numerical value
            # Ensure keys exist in display_mappings
            gender_display_options = list(display_mappings['Patient Gender'].values()) if 'Patient Gender' in display_mappings else ['Unknown']
            diabeties_display_options = list(display_mappings['Diabeties'].values()) if 'Diabeties' in display_mappings else ['Unknown']
            grade_display_options = list(display_mappings['Grade'].values()) if 'Grade' in display_mappings else ['Unknown']

            gender_est_label = st.selectbox("Select Gender for Estimation", gender_display_options, key='gender_est_label')
            diabeties_est_label = st.selectbox("Diabeties Status for Estimation", diabeties_display_options, key='diabeties_est_label')
            grade_est_label = st.selectbox("Grade of Leprosy for Estimation", grade_display_options, key='grade_est_label')

            # Get the numerical value corresponding to the selected label using display_to_numerical_mapping
            gender_est = display_to_numerical_mapping['Patient Gender'].get(gender_est_label, 0) # Default to 0 if label not found
            diabeties_est = display_to_numerical_mapping['Diabeties'].get(diabeties_est_label, 0) # Default to 0 if label not found
            grade_est = display_to_numerical_mapping['Grade'].get(grade_est_label, 0) # Default to 0 if label not found


        with col2:
            st.subheader("Estimation Results")
            if st.button("Estimate Foot Measurements"):
                try:
                    # Create input DataFrame using the numerical values
                    input_df_est = pd.DataFrame({
                        'Patient Age': [age_est],
                        'Patient Gender': [gender_est],
                        'Diabeties': [diabeties_est],
                        'Grade': [grade_est]
                    })

                    # Encode categorical features using clustering encoders (will encode 0/1 strings to 0/1 integers if needed)
                    for col in ['Patient Gender', 'Diabeties', 'Grade']:
                         if col in input_df_est.columns and col in label_encoders_clustering:
                            le = label_encoders_clustering[col]
                            # Transform the numerical value (as string) using the loaded encoder
                            # Check if the numerical value as string is in the encoder's classes
                            if str(input_df_est[col].iloc[0]) in le.classes_:
                                input_df_est[col] = le.transform(input_df_est[col].astype(str))
                            else:
                                st.warning(f"Numerical value '{input_df_est[col].iloc[0]}' for '{col}' is not recognized by clustering encoder. Using a default encoding (0).")
                                # Handle unseen data - use a default. Using 0 as a simple default.
                                input_df_est[col] = 0
                         elif col in input_df_est.columns:
                             # If encoder not found for this column, assume it should be numerical and cast
                             input_df_est[col] = input_df_est[col].astype(int)


                    # Scale inputs using clustering scaler
                    input_scaled_est = scaler.transform(input_df_est)

                    # Predict cluster
                    cluster = kmeans.predict(input_scaled_est)[0]

                    st.info(f"Predicted Cluster: {cluster}")
                    st.subheader("Estimated Average Foot Measurements:")

                    cluster_output = cluster_means.loc[cluster][[
                        'Length of Foot Right',
                        'Length of Foot Left',
                        'Ball Girth Right',
                        'Ball Girth Left',
                        'In Step Right',
                        'In Step Left'
                    ]]

                    st.dataframe(cluster_output.to_frame(name="Average Value"))

                except Exception as e:
                    st.error(f"Estimation failed: {e}")
    else:
        st.warning("Foot measurement estimation models are not available. Please ensure all model files are uploaded.")


elif page == "Leprosy Risk Prediction":
    st.header("Leprosy Risk Prediction")
    st.markdown("""
    Enter the patient's information and foot measurements below to get a prediction on the likelihood of having leprosy.
    """)

    if 'classifier_model' in locals() and 'classification_features_names' in locals(): # Check if classification models and feature names were loaded
        # Collect user input for classification features
        st.subheader("Patient Information and Foot Measurements for Prediction")
        age_pred = st.number_input("Enter Patient Age for Prediction", min_value=0, max_value=120, value=50, key='age_pred')

        # Use display_mappings for selectbox options in prediction section as well
        gender_pred_display_options = list(display_mappings['Patient Gender'].values()) if 'Patient Gender' in display_mappings else ['Unknown']
        diabeties_pred_display_options = list(display_mappings['Diabeties'].values()) if 'Diabeties' in display_mappings else ['Unknown']
        grade_pred_display_options = list(display_mappings['Grade'].values()) if 'Grade' in display_mappings else ['Unknown']


        gender_pred_label = st.selectbox("Select Gender for Prediction", gender_pred_display_options, key='gender_pred_label')
        diabeties_pred_label = st.selectbox("Diabeties Status for Prediction", diabeties_pred_display_options, key='diabeties_pred_label')
        grade_pred_label = st.selectbox("Grade of Leprosy for Prediction", grade_pred_display_options, key='grade_pred_label')

        # Get the numerical value corresponding to the selected label using display_to_numerical_mapping
        gender_pred = display_to_numerical_mapping['Patient Gender'].get(gender_pred_label, 0) # Default to 0 if label not found
        diabeties_pred = display_to_numerical_mapping['Diabeties'].get(diabeties_pred_label, 0) # Default to 0 if label not found
        grade_pred = display_to_numerical_mapping['Grade'].get(grade_pred_label, 0) # Default to 0 if label not found


        st.markdown("---")
        st.subheader("Foot Measurements")
        length_right = st.number_input("Length of Foot Right (cm)", min_value=0.0, value=25.0, key='length_right')
        length_left = st.number_input("Length of Foot Left (cm)", min_value=0.0, value=25.0, key='length_left')
        ball_girth_right = st.number_input("Ball Girth Right (cm)", min_value=0.0, value=22.0, key='ball_girth_right')
        ball_girth_left = st.number_input("Ball Girth Left (cm)", min_value=0.0, value=22.0, key='ball_girth_left')
        in_step_right = st.number_input("In Step Right (cm)", min_value=0.0, value=24.0, key='in_step_right')
        in_step_left = st.number_input("In Step Left (cm)", min_value=0.0, value=24.0, key='in_step_left')


        if st.button("Predict Leprosy Risk"):
            try:
                # Create input DataFrame for classification using numerical values
                input_df_pred = pd.DataFrame({
                    'Patient Age': [age_pred],
                    'Patient Gender': [gender_pred],
                    'Diabeties': [diabeties_pred],
                    'Grade': [grade_pred],
                    'Length of Foot Right': [length_right],
                    'Length of Foot Left': [length_left],
                    'Ball Girth Right': [ball_girth_right],
                    'Ball Girth Left': [ball_girth_left],
                    'In Step Right': [in_step_right],
                    'In Step Left': [in_step_left]
                })

                # Encode categorical features using classification encoders (will encode 0/1 strings to 0/1 integers if needed)
                for col in ['Patient Gender', 'Diabeties', 'Grade']:
                     if col in input_df_pred.columns and col in label_encoders_classification:
                        le = label_encoders_classification[col]
                         # Transform the numerical value (as string) using the loaded encoder
                        # Check if the numerical value as string is in the encoder's classes
                        if str(input_df_pred[col].iloc[0]) in le.classes_:
                             input_df_pred[col] = le.transform(input_df_pred[col].astype(str))
                        else:
                             st.warning(f"Numerical value '{input_df_pred[col].iloc[0]}' for '{col}' is not recognized by classification encoder. Using a default encoding (0).")
                             # Handle unseen data - use a default. Using 0 as a simple default.
                             input_df_pred[col] = 0

                     elif col in input_df_pred.columns:
                         # If encoder not found for this column, assume it should be numerical and cast
                         input_df_pred[col] = input_df_pred[col].astype(int)


                # Ensure column order matches training data using loaded feature names
                if classification_features_names:
                     # Reindex to match training columns order
                     input_df_pred = input_df_pred[classification_features_names]
                else:
                     st.warning("Classification training feature names not found. Proceeding with input order. Ensure consistency.")


                prediction_proba = classifier_model.predict_proba(input_df_pred)[0]
                # Assuming the target variable 'Footwear/Leprosy' is binary (0 or 1)
                # We need to know which class index corresponds to 'Leprosy' (1)
                # Based on the classification report, class 1 is Leprosy
                leprosy_probability = prediction_proba[1] # Probability of class 1 (Leprosy)
                no_leprosy_probability = prediction_proba[0] # Probability of class 0 (No Leprosy)

                st.subheader("Prediction Results:")
                # Display probability as percentage
                st.info(f"Predicted Probability of having Leprosy: **{leprosy_probability:.2%}**")

                # Create Pie Chart
                labels = ['Probability of Leprosy', 'Probability of No Leprosy']
                sizes = [leprosy_probability, no_leprosy_probability]
                colors = ['#ff9999','#66b3ff'] # Example colors
                explode = (0.1, 0)  # explode 1st slice (i.e. 'Probability of Leprosy')

                fig1, ax1 = plt.subplots()
                ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                        shadow=True, startangle=90)
                ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

                st.pyplot(fig1)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.warning("Leprosy risk prediction models or feature names are not available. Please ensure all model files are uploaded.")

st.markdown("---") # Add a horizontal rule
st.markdown("© 2024 Foot Measurement Estimator and Leprosy Risk Predictor")
