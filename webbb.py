import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open(r"Adelaide.pkl", 'rb'))

# Add custom CSS for background image
background_image = "https:///Users/divyanshuyadav/Desktop/ml/seoul.jpg"
background_color = "rgba(0, 0, 0, 0.0)"  # Adjust the opacity as needed

st.markdown(
    f"""
    <style>
    body {{
        background-image: url("{background_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center center;
        background-attachment: fixed;
        background-color: {background_color};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar - About section
st.sidebar.title("About")
st.sidebar.image("/Users/divyanshuyadav/Desktop/ml/dypic.jpeg", use_column_width=True)
st.sidebar.write("Name: Divyanshu Yadav")
st.sidebar.write("Enrollment No: 10919011921")
st.sidebar.write("Branch: AI&DS")
st.sidebar.write("Batch: B2")

# Main content
st.title('Temperature Forecast Bias Correction')

# Input fields for temperature forecast data
cols = []
for i in range(6):
    col = st.number_input(f"Enter NWP forecast data {i+1}", value=0.0)
    cols.append(col)

if st.button('Perform Bias Correction'):
    # Convert input data to a numpy array
    data = np.array(cols).reshape(1, -1)

    # Perform bias correction prediction
    corrected_data = model.predict(data)

    # Display the result
    st.write("Bias-corrected forecast data:")
    st.write(corrected_data)
