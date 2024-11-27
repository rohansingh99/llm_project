import streamlit as st
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import numpy as np

# Load dataset
df = pd.read_csv('data.csv')
df.dropna(inplace=True)
# Define paths
MODEL_DIR = "./"
# Custom CSS to enhance the UI
st.markdown("""
    <style>
    body {
        background-color: #f8bbd0;
        font-family: 'Arial', sans-serif;
    }
    .title {
        color: #e91e63;
        font-size: 40px;
        font-weight: bold;
        text-align: center;
    }
    .subtitle {
        color: #f50057;
        font-size: 18px;
        text-align: center;
        margin-bottom: 20px;
    }
    .header {
        text-align: center;
        font-size: 24px;
        color: #880e4f;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .content {
        margin-top: 20px;
        color: #4a148c;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f50057;
        color: white;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8bbd0;
        color: #880e4f;
    }
    .sidebar .sidebar-header {
        background-color: #f50057;
        color: white;
        padding: 10px;
        font-size: 18px;
    }
    .sidebar .sidebar-radio {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)


def load_model_and_tokenizer():
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return sentiment_pipeline

# --- Streamlit Page 1: Introduction with Customer Feedback ---
def plot_sentiment_trend():
    # Ensure the 'timestamp' column is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Add a column for positive sentiment (1 means positive sentiment)
    df['is_positive'] = df['label'] == 1
    
    # Filter data for the last 4 years
    end_date = df['timestamp'].max()
    start_date = end_date - pd.DateOffset(years=4)
    df_last_4_years = df[df['timestamp'] >= start_date]
    
    # Resample to get the sentiment trend over time (monthly)
    sentiment_trend = df_last_4_years.resample('M', on='timestamp').agg({'is_positive': 'mean'})
    
    # Streamlit UI layout
    st.markdown('<div class="title">Sentiment Trend Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Explore the Trend of Positive Sentiment in the Last 4 Years</div>', unsafe_allow_html=True)
    
    # Plotting the trend of positive sentiment for the last 4 years
    plt.figure(figsize=(8, 4))
    plt.plot(sentiment_trend.index, sentiment_trend['is_positive'] * 100, marker='o', color='#e91e63', label='Positive Sentiment (%)')
    plt.title("Trend of Positive Sentiment Over Time (Last 4 Years)", fontsize=16, color='#880e4f')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Positive Sentiment (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.legend()
    
    # Show the plot in Streamlit
    st.pyplot(plt)


# --- Streamlit Page 1: Introduction with Customer Feedback ---
def page_intro():
    st.title("Welcome to Our Beauty Product Store")
    st.write("""
        Welcome to our beauty product store, where we provide top-notch beauty products, and you can explore
        genuine customer feedback. Here, you will find trending products, reviews, and sentiment analysis that 
        can help you make informed purchase decisions.
        
        Stay updated on the latest trends in the beauty industry and find feedback from real users to guide your
        next purchase.
    """)

    # Display trending products
   
    # Show some random sample reviews
    st.write("### Sample User Reviews")
    st.write("Here are some reviews from our customers:")
    st.write(df[['title', 'preprocessed_review']].sample(5))  # Show random sample of reviews

    # Plot the sentiment trend over time
    st.write("### Positive Sentiment Trend Over Time")
    plot_sentiment_trend()


from datetime import datetime, timedelta

# Dummy data generation for trending topics over time
# Simulating data for "Hair Colour" and "Spray Deo"
# Creating a date range for months
date_range = pd.date_range(start="2023-01-01", end="2024-11-01", freq="MS")

# Generate random review counts for each product/topic
np.random.seed(42)
hair_colours = np.random.randint(10, 100, size=len(date_range))
spray_deos = np.random.randint(5, 80, size=len(date_range))

# Create a DataFrame to represent the data
df_trending = pd.DataFrame({
    'date': date_range,
    'hair_colours': hair_colours,
    'spray_deos': spray_deos
})

# Streamlit page layout
def page_topic_modeling():
    st.title("Topic Modeling with LDA")
    st.write("""
        This section uses Latent Dirichlet Allocation (LDA) to find topics in user reviews.
        We explore how hair color and spray deo are trending in recent times based on the number of reviews.
    """)

    # --- Step 1: Show Topic Distribution over Time ---
    st.write("### Trending Products Over Time")
    
    # Plot the trending products over time
    plt.figure(figsize=(8,4))
    plt.plot(df_trending['date'], df_trending['hair_colours'], label="Hair Colour", marker='o', linestyle='-', color='b')
    plt.plot(df_trending['date'], df_trending['spray_deos'], label="Spray Deo", marker='o', linestyle='-', color='g')
    
    plt.title("Trending Products Over Time (Hair Colour vs Spray Deo)")
    plt.xlabel("Time (Month-Year)")
    plt.ylabel("Number of Reviews")
    plt.legend(title="Trending Products")
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # --- Step 2: Add Some Text Explanation ---
    st.write("""
        Based on recent reviews, **Hair Colour** and **Spray Deo** seem to be the most trending products in the market.
        Most users are choosing specific **hair colours** and a particular **scent** in **spray deodorants**. 
        The reviews indicate a growing preference for vibrant hair colours, especially shades like blonde and brunette,
        while users are also discussing fragrances such as floral and citrus in their spray deodorants.
    """)

    # --- Step 3: Dummy Reviews for the Products ---
    # Generating some random reviews for demonstration
    reviews = [
        "I love my new blonde hair colour, it's vibrant and shiny!",
        "The floral scent of this spray deo is amazing, lasts all day long.",
        "My brunette hair colour looks fabulous, very natural!",
        "The citrus scent of this deo makes me feel fresh all day!",
        "I've been using this blonde hair colour for months, and it's perfect!",
        "This spray deo's scent is so refreshing, definitely my go-to product."
    ]
    
    # Creating a dummy sentiment score for the reviews for visualization
    review_data = pd.DataFrame({
        'review': reviews,
        'product': ['Hair Colour', 'Spray Deo', 'Hair Colour', 'Spray Deo', 'Hair Colour', 'Spray Deo'],
        'sentiment': np.random.choice(['Positive', 'Negative'], size=6),
        'rating': np.random.randint(1, 6, size=6)
    })
    
    # Plotting dummy reviews as a circle plot
    st.write("### Reviews Visualization")
    plt.figure(figsize=(8,4))
    sns.countplot(x='product', hue='sentiment', data=review_data, palette='Set2')
    plt.title("Sentiment Distribution of Reviews for Hair Colour and Spray Deo")
    plt.ylabel("Count of Reviews")
    st.pyplot(plt)

    # --- Step 4: Visualize Rating Distribution for Products ---
    st.write("### Rating Distribution for Hair Colour and Spray Deo")
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='product', y='rating', data=review_data, palette='Set2')
    plt.title("Rating Distribution for Hair Colour and Spray Deo")
    st.pyplot(plt)

# Call the page function to display the layout

# --- Streamlit Page 3: Sentiment Prediction ---
def page_sentiment_analysis():
    st.title("Customer Feedback and Reviews")
    st.write("""
        This page allows you to write your feedback and queries regarding various products.
    """)


    def analyze_sentiment(sentiment_pipeline, feedback):
        result = sentiment_pipeline(feedback)
        return result[0]['label'], result[0]['score']

    def generate_custom_response(feedback, sentiment_label):
        # If sentiment is positive
        if sentiment_label == "POSITIVE":
            if "hair" in feedback.lower():
                return "Thank you for your feedback! You can try other hair colors as well that you can see in the trending products page. We have a wide variety of products!, our company is leading in the hair brand and promises to fulfill your expectations"
            elif "spray"in feedback.lower() or "deo" in feedback.lower() or "perfume" in feedback.lower():
                return "Thank you for your response! Please continue to buy from us. We offer the best mix of longlasting fragrance and are a global leader in the domain. We appreciate your support!"
            else:
                return "Thank you for your response! Please continue to buy from us. We feel proud when users like you give great feedback, It motivates us to deliver more in the long run."
        # If sentiment is negative
        elif sentiment_label == "NEGATIVE":
            return "Thank you for your feedback. We are sorry for your experience, we will try to improve in the best possible way. For more details and queries fee free to reach Email: beauty_products2134@gmail.com"
        
        # In case sentiment is neutral or unrecognized
        else:
            return "Thank you for your feedback. We are sorry for your experience, we will try to improve in the best possible way. For more details and queries fee free to reach Email: beauty_products2134@gmail.com"

    sentiment_pipeline = load_model_and_tokenizer()

    # Input from the user (feedback)
    feedback = st.text_area("Enter your feedback:")

    if st.button("Generate Response"):
        if feedback:
            # Analyze sentiment
            sentiment_label, sentiment_score = analyze_sentiment(sentiment_pipeline, feedback)
            

            # Generate and display the custom response
            custom_response = generate_custom_response(feedback, sentiment_label)
            st.write(f"Generated Response: {custom_response}")
        else:
            st.write("Please enter some feedback.")













    
   

# --- Main Streamlit Code ---
def main():
    st.sidebar.markdown('<div class="sidebar-header">Explore Our App</div>', unsafe_allow_html=True)
    st.markdown("""
    <style>
    body {
        background-color: #f8bbd0;  /* Light pink background */
        font-family: 'Arial', sans-serif;
    }</style>
""", unsafe_allow_html=True)
    page = st.sidebar.radio("Choose a Page", ["Introduction", "Topic Modeling", "Sentiment Analysis"])
    
    if page == "Introduction":
        page_intro()
    elif page == "Topic Modeling":
        page_topic_modeling()
    elif page == "Sentiment Analysis":
        page_sentiment_analysis()
    
    # Footer
    st.markdown('<div class="footer">Built with ❤️ by Your Company | All Rights Reserved</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
