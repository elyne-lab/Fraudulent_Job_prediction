import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly


# Load the saved model and vectorizer
model = pickle.load(open('fraudulentjob_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

st.markdown(
    """
    <h1 style='text-align: center; color: #567ED0; font-family: Verdana, Geneva, sans-serif;'>
    Job Posting: Fraud Detection Model
    </h1>
    """,
    unsafe_allow_html=True,
)


def main():
    # Set page title and description
    
    st.markdown("""
    This application predicts whether a job posting might be fraudulent based on its details.
    Please fill in the information on the job listing that you have identified and submit on the below tab:
    """)
    
    # Create input fields
    st.sidebar.write("Option 1: Enter Details")
    title = st.sidebar.text_input('Job Title')
    
    description = st.sidebar.text_area('Job Description')
    
    requirements = st.sidebar.text_area('Job Requirements')
    
    company_profile = st.sidebar.text_area('Company Profile')
    
    location = st.sidebar.text_input('Location')

    # Addin validation metrics {Error Handing & input validation} to the app features
    def validate_inputs(title, description, requirements, company_profile, location):
        if not title or not description:
            return False, "Job title and description are required fields."
        if len(description) < 50:
            return False, "Job description should be at least 50 characters long."
        return True, ""

    # Modify your main() function to include validation
    if st.button('Predict'):
        # Validate inputs
        is_valid, error_message = validate_inputs(title, description, requirements, 
                                                company_profile, location)
        
        if not is_valid:
            st.error(error_message)
        else:
            # Combine all text fields
            combined_text = f"{title} {description} {requirements} {company_profile} {location}"
            
            # Transform the text using saved TF-IDF vectorizer
            text_tfidf = tfidf.transform([combined_text])
            
            # Make prediction
            prediction = model.predict(text_tfidf)
            probability = model.predict_proba(text_tfidf)
            
            # Display results
            st.subheader('Prediction Results:')
            
            if prediction[0] == 1:
                st.error('⚠️ This job posting might be fraudulent!')
                st.write(f'Probability of being fraudulent: {probability[0][1]:.2%}')
            else:
                st.success('✅ This job posting appears to be legitimate.')
                st.write(f'Probability of being legitimate: {probability[0][0]:.2%}')
            
            # Display confidence metrics
            st.subheader('Prediction Confidence:')
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Legitimate Probability", 
                        value=f"{probability[0][0]:.2%}")
            with col2:
                st.metric(label="Fraudulent Probability", 
                        value=f"{probability[0][1]:.2%}")
            
            # Add visualization
            st.write("View the Visualised Probabilities")
            import plotly.graph_objects as go
            
            fig = go.Figure(go.Bar(
                x=['Legitimate', 'Fraudulent'],
                y=[probability[0][0], probability[0][1]],
                marker_color=['green', 'red']
            ))
            fig.update_layout(title='Prediction Probabilities',
                            yaxis_title='Probability',
                            yaxis_range=[0,1])
            st.plotly_chart(fig)

if __name__ == '__main__':
    main()



import streamlit as st
import pandas as pd

# Main content
st.title("Fraudulent Job Analytics")
st.write("Explore the fraudulent job dataset using the sidebar!")

#load the dataset
#df= pd.read_csv('Fake Postings.csv')

## Create a siderbar for user inputs
st.header('The Reporting Dashboard')
st.write('''To make an analysis of the fraudulent jobs in a list of jobs identified on the job site,
         proceed and upload the data file onto the model''')
st.sidebar.subheader('Option 2: File Upload')
st.sidebar.write('Upload a csv file with your job features')

##
df = st.sidebar.file_uploader('Upload your file', type=['csv'])

st.subheader("Sample Fake Job Postings Data")
st.write("Data Preview:")

if df is not None:
    # read the file
    df = pd.read_csv(df)
    st.dataframe(df.head())
else:
    # Display a message if no file is uploaded
    st.warning('''Please upload a file to Option 2 to proceed.
               
            Refer to the error below!''')

#Creating our Data Header
#st.subheader("Sample Fake Job Postings Data")
#st.write("Data Preview:")
#st.dataframe(df.head())

#Creating an input where the person interracting with the app can show more data
st.sidebar.subheader("Show More Fake Job Data")
if st.sidebar.checkbox('Show More Job Data'):
    #Specified number of rows to show
    num_rows = st.sidebar.number_input("Number of rows to show", 2, 100)
    st.sidebar.table(df.head(num_rows))


 # Sidebar for bar chart for fraudulent jobs by industry
    st.sidebar.subheader("Fake Jobs by Industry")
    fake_jobs_by_industry = (
        df[df["fraudulent"] == 1]
        .groupby("industry")
        .size()
        .sort_values(ascending=False)
    )   

    # Display bar chart
    st.sidebar.bar_chart(fake_jobs_by_industry)


# Analyze Fraudulent Jobs by Employment Type
fraud_by_emp_type = df[df["fraudulent"] == 1].groupby("employment_type").size().sort_values(ascending=False)

# Sidebar Option
st.sidebar.header("By Employment Type")
show_emp_type = st.sidebar.checkbox("Show Fraudulent Jobs by Employment Type", value=True)

# Display the bar chart
if show_emp_type:
    st.subheader("Fraudulent Jobs Distribution by Employment Type")

    # Matplotlib Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    fraud_by_emp_type.plot(kind="bar", color="skyblue", ax=ax)
    ax.set_title("Fraudulent Jobs Distribution by Employment Type", fontsize=14, color = 'blue')
    ax.set_xlabel("Employment Type", fontsize=12, color = 'blue')
    ax.set_ylabel("No of Fraudulent Jobs", fontsize=12, color ='red')
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    st.pyplot(fig)  # Display the plot in Streamlit

    # Data Display
    st.write("Fraudulent Job Counts by Employment Type:")
    st.table(fraud_by_emp_type.reset_index(name="Number of Fraudulent Jobs"))


# Sidebar Option
st.sidebar.header("By Location")
show_top_10 = st.sidebar.checkbox("Show Top 10 Locations with Fraudulent Jobs", value=True)

# Analyze Fraudulent Jobs
fraud_by_location = df[df["fraudulent"] == 1].groupby("location").size().sort_values(ascending=False)
top_10_locations = fraud_by_location.head(10)

# Plot if the checkbox is selected
if show_top_10:
    st.subheader("Top 10 Locations with the Highest Number of Fraudulent Jobs")

    # Matplotlib Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    top_10_locations.plot(kind="barh", color="pink", ax=ax)
    ax.set_title("Top 10 Locations with Fraudulent Jobs", fontsize=14, color = 'blue')
    ax.set_xlabel("Location", fontsize=12)
    ax.set_ylabel("No. of Fraudulent Jobs", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    st.pyplot(fig)  # Display the plot in Streamlit

    # Data Display
    st.write("Fraudulent Job Counts by Location:")
    st.table(top_10_locations.reset_index(name="Number of Fraudulent Jobs"))


 # Analyze Fraudulent Jobs by the type of benefits in the job offer
fraud_by_benefits = df[df["fraudulent"] == 1].groupby("employment_type").size().sort_values(ascending=False)

# Sidebar Option
st.sidebar.header("By Benefits Offered")
show_benefits = st.sidebar.checkbox("Show Fraudulent Jobs by Benefits", value=True)

# Display the bar chart
if show_benefits:
    st.subheader("Fraudulent Jobs Distribution by Benefits")

    # Matplotlib Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    fraud_by_benefits.plot(kind="bar", color="skyblue", ax=ax)
    ax.set_title("Fraudulent Jobs Distribution by Benefits", fontsize=14, color = 'blue')
    ax.set_xlabel("Benefits", fontsize=12, color = 'blue')
    ax.set_ylabel("No of Fraudulent Jobs", fontsize=12, color ='red')
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    st.pyplot(fig)  # Display the plot in Streamlit

    # Data Display
    st.write("Fraudulent Job Counts by Benefits:")
    st.table(fraud_by_benefits.reset_index(name="Number of Fraudulent Jobs"))

# Fraudulent jobs distribution by industry
fraud_by_industry = df[df["fraudulent"] == 1].groupby("industry").size()

# Streamlit App Title
st.title("Fraudulent Jobs Distribution by Industry")

# Create a pie chart
fig, ax = plt.subplots(figsize=(4, 4))
fraud_by_industry.plot(kind="pie", autopct="%1.1f%%", startangle=140, ax=ax, 
                       colors=plt.cm.tab10.colors, labels=fraud_by_industry.index)
ax.set_ylabel("")  # Remove y-axis label for cleaner look
ax.set_title("Fraudulent Job Distribution by Industry", fontsize=14)

# Display the pie chart in Streamlit
st.pyplot(fig)

# Data Display
st.write("Fraudulent Job Counts by Industry:")
st.table(fraud_by_industry.reset_index(name="Number of Fraudulent Jobs"))


##Calculating the average salaries, and displaying it on the line chart:

# Streamlit application title
st.subheader("Average Salary by Employment Type")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())

    # Check if required columns exist
    required_columns = ["lower_sal", "upper_sal", "employment_type"]
    if all(col in df.columns for col in required_columns):
        # Calculate the average salary for each row
        df["avg_salary"] = (df["lower_sal"] + df["upper_sal"]) / 2

        # Group by employment_type and calculate the average salary
        average_salary_by_employment_type = df.groupby("employment_type")["avg_salary"].mean()

        # Plot the line chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            average_salary_by_employment_type.index, 
            average_salary_by_employment_type.values, 
            marker="o", linestyle="-", color="green", label="Average Salary"
        )

        # Customize the plot
        ax.set_title("Average Salary by Employment Type", fontsize=14)
        ax.set_xlabel("Employment Type", fontsize=12)
        ax.set_ylabel("Average Salary", fontsize=12)
        ax.set_xticks(range(len(average_salary_by_employment_type.index)))
        ax.set_xticklabels(average_salary_by_employment_type.index, rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend()

        # Display the plot
        st.pyplot(fig)
    else:
        st.warning(f"Please ensure your dataset contains the required columns: {required_columns}")
else:
    st.info("Please upload a CSV file to proceed.")

st.subheader('End of the app')
