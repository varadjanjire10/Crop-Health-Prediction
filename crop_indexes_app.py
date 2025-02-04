import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV into a DataFrame
df = pd.read_csv("combined_scores_with_health_status.csv")

# Extract the year from 'Year_Month' column
df['Year'] = df['Year_Month'].apply(lambda x: x.split('_')[0])

# Sidebar for Year Selection
year_selected = st.sidebar.selectbox("Select Year", df['Year'].unique())

# Filter the data for the selected year
df_filtered = df[df['Year'] == year_selected]

# Visualize the 6 index scores for the selected year
st.title(f"Index Scores for Year {year_selected}")

# Plot the six index scores
fig, ax = plt.subplots(figsize=(12, 8))  # Adjust figure size
index_columns = [f'Mean_{index}_Score' for index in ['NDVI', 'MI', 'SAVI', 'NDBI', 'EVI', 'NDWI']]
df_filtered[index_columns].plot(kind='bar', ax=ax, width=0.6, legend=True,
                                 color=['green', 'blue', 'brown', 'orange', 'red', 'cyan'])  # Add colors

# Set labels and title
ax.set_xlabel('Month')
ax.set_ylabel('Score')
ax.set_title(f"Index Scores for Year {year_selected}")

# Show the plot
plt.xticks(rotation=45)
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)