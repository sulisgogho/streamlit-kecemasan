from navigation import make_sidebar
import streamlit as st
import pandas as pd
from scipy.spatial import distance

make_sidebar()

# Load the dataset
df = pd.read_csv('kecemasanatlet.csv')

# Display the entire dataset
st.write('## Full Dataset')
st.dataframe(df)

# Assuming the first row as the reference point for demonstration
st.write('## Reference (Testing) Data')
reference_data = df.iloc[0]
st.dataframe(reference_data.to_frame().T)

# Selecting the features columns (P1 to P17)
features = df.columns[1:-1]

# Calculating Euclidean distance from the reference point to all other points
df['Euclidean Distance'] = df[features].apply(lambda row: distance.euclidean(row, reference_data[features]), axis=1)

# Adding ranking based on Euclidean Distance
df['Ranking'] = df['Euclidean Distance'].rank(method='first').astype(int)

# Displaying the Euclidean distance calculation with ranking
st.write('## Euclidean Distance Calculation with Ranking')
distance_df = df[['Nama', 'Diagnosa', 'Euclidean Distance', 'Ranking']].reset_index(drop=True)
st.dataframe(distance_df)

# Displaying the values based on ranking
st.write('## Values Based on Ranking')
ranked_df = distance_df.sort_values(by='Ranking')  # Sort by Ranking
st.dataframe(ranked_df)
