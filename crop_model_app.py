import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load the trained model
model = joblib.load('crop_health_model.pkl')

# Load the CSV data
df = pd.read_csv('combined_scores_with_health_status.csv')

# Extract year from Year_Month column
df['Year'] = df['Year_Month'].apply(lambda x: x.split('_')[0])

# Calculate correlation matrix between the index scores
index_columns = ["Mean_NDVI_Score", "Mean_MI_Score", "Mean_SAVI_Score", "Mean_NDBI_Score", "Mean_EVI_Score", "Mean_NDWI_Score"]
correlation_matrix = df[index_columns].corr()

# Sidebar for selecting which index to adjust
st.sidebar.title("Crop Health Prediction")
selected_index = st.sidebar.selectbox("Select Index to Adjust", index_columns)

# Sidebar slider for adjusting the selected index
selected_value = st.sidebar.slider(f"Adjust {selected_index}", min_value=0.0, max_value=1.0, step=0.01, value=0.3)

# Create a dictionary to store the adjusted index scores
adjusted_scores = {}

# Adjust all indexes based on the selected index and its correlations
for index in index_columns:
    if index == selected_index:
        adjusted_scores[index] = selected_value
    else:
        # Adjust the value based on correlation with the selected index
        correlation = correlation_matrix[selected_index][index]
        # Calculate the new value based on the correlation
        adjusted_scores[index] = selected_value * correlation + (1 - correlation) * np.mean(df[index])

# Store the adjusted scores in a DataFrame
adjusted_data = pd.DataFrame([adjusted_scores])

# Predict crop health status based on the adjusted index scores
# prediction = model.predict(adjusted_data)

# Display the prediction result
# if prediction == 1:
#    st.write("The crop health status is: **Good**")
# else:
#     st.write("The crop health status is: **Bad**")

# Visualize the adjusted index scores
st.subheader("Adjusted Index Scores Visualization")
fig, ax = plt.subplots(figsize=(8, 5))

# Plot bar chart for the adjusted index scores
index_labels = list(adjusted_scores.keys())
index_values = list(adjusted_scores.values())

ax.bar(index_labels, index_values, color=['green', 'blue', 'brown', 'orange', 'red', 'cyan'])

# Set labels and title
ax.set_ylabel('Adjusted Index Scores')
ax.set_title('Adjusted Index Scores Based on Correlation')

# Rotate the x-axis labels by 45 degrees
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
st.pyplot(fig)
