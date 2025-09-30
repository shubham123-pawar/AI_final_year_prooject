import streamlit as st
from googleapiclient.discovery import build
from textblob import TextBlob
import matplotlib.pyplot as plt
import re
import pandas as pd
from datetime import datetime

# 🔑 IMPORTANT: Replace with your actual Google API key
# NOTE: The provided key is a placeholder. You must use a valid Google API key 
# with the YouTube Data API enabled for the first tab to work.
API_KEY = "AIzaSyChm6Pl7qiGqobBRFg8Hp7bNrRfwzbxLiE" 

# --- CORE FUNCTIONS ---

def extract_video_id(url):
    """Extracts the 11-character YouTube video ID from various URL formats."""
    pattern = re.compile(
        r'(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})',
        re.IGNORECASE
    )
    match = pattern.search(url)
    if match:
        return match.group(1)
    if len(url) == 11 and not any(char in url for char in ['/', '=']):
        return url
    raise ValueError("Invalid YouTube URL or ID format.")

def get_comments(video_id, max_results=50):
    """Fetches a list of top-level comments from a given video ID."""
    try:
        youtube = build("youtube", "v3", developerKey=API_KEY)
        comments = []
        request = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=max_results, textFormat="plainText"
        )
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        return comments
    except Exception as e:
        if "404" in str(e) and "videoNotFound" in str(e):
            st.error("Error 404: The video was not found or has comments disabled.")
        elif "403" in str(e) and "commentsDisabled" in str(e):
            st.error("Error 403: Comments are disabled for this video.")
        else:
            st.error(f"An API error occurred. Please check your API key and permissions. Details: {e}")
        return None

def analyze_sentiment(comments):
    """Analyzes sentiment for each comment in a list."""
    results = {"Positive": 0, "Negative": 0, "Neutral": 0}
    polarities = []
    
    for comment in comments:
        blob = TextBlob(comment)
        polarity = blob.sentiment.polarity
        polarities.append(polarity)
        
        if polarity > 0:
            results["Positive"] += 1
        elif polarity < 0:
            results["Negative"] += 1
        else:
            results["Neutral"] += 1
            
    return results, polarities, comments

@st.cache_data
def analyze_single_text_sentiment(text):
    """Analyzes sentiment for a single text string (used for student feedback)."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.05:
        sentiment = "Positive"
    elif polarity < -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, polarity

# --- INITIALIZATION ---

# Initialize a list in session state to hold all submitted feedback records
if 'feedback_records' not in st.session_state:
    st.session_state.feedback_records = []

# --- STREAMLIT APP LAYOUT ---

st.set_page_config(page_title="Multi-Source Sentiment Analyzer", layout="wide")
st.title("Sentiment Analyzer Dashboard 📊")

# Define the tabs structure
tab1, tab2, tab3 = st.tabs(["📹 YouTube Comments", "💬 Student Feedback Form", "📈 Data Dashboard"])

# =============================================================
# TAB 1: YOUTUBE COMMENT ANALYZER
# =============================================================
with tab1:
    st.header("YouTube Comment Sentiment Analyzer 🎥")
    st.markdown("Enter a YouTube video link to analyze the sentiment of the latest comments.")

    video_link = st.text_input("Paste YouTube Video Link Here:", 
                               "https://www.youtube.com/watch?v=dQw4w9WgXcQ", 
                               key="tab1_link_input")

    max_comments = st.slider("Max Comments to Analyze:", 10, 100, 50, step=10, key="tab1_slider")

    if st.button("Analyze YouTube Comments", type="primary"):
        if not video_link:
            st.warning("Please enter a YouTube video link to proceed.")
            st.stop()
            
        try:
            video_id = extract_video_id(video_link)
            st.info(f"Successfully extracted Video ID: `{video_id}`. Fetching {max_comments} comments...")

            with st.spinner("Fetching and analyzing comments..."):
                comments = get_comments(video_id, max_comments)

            if comments:
                st.success(f"Fetched {len(comments)} comments.")
                sentiment_results, polarities, raw_comments = analyze_sentiment(comments)
                total_comments = sum(sentiment_results.values())
                
                col1, col2 = st.columns(2)
                
                # Column 1: Pie Chart
                with col1:
                    st.subheader("Sentiment Distribution")
                    labels = sentiment_results.keys()
                    sizes = sentiment_results.values()
                    colors = ['#8bc34a', '#f44336', '#ffc107']
                    
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
                    ax.axis('equal')
                    st.pyplot(fig)
                    
                # Column 2: Key Metrics
                with col2:
                    st.subheader("Key Metrics")
                    metrics_data = {
                        "Sentiment": list(sentiment_results.keys()),
                        "Count": list(sentiment_results.values()),
                        "Percentage": [f"{v/total_comments * 100:.1f}%" for v in sentiment_results.values()]
                    }
                    st.dataframe(pd.DataFrame(metrics_data).set_index("Sentiment"))
                    
                    avg_polarity = sum(polarities) / len(polarities) if polarities else 0
                    st.metric(label="Average Polarity (Overall Tone)", value=f"{avg_polarity:.3f}")
                
                with st.expander("View Raw Comments and Polarity"):
                    df_comments = pd.DataFrame({
                        "Comment": raw_comments,
                        "Polarity": polarities,
                        "Sentiment": ['Positive' if p > 0 else 'Negative' if p < 0 else 'Neutral' for p in polarities]
                    })
                    st.dataframe(df_comments)
            
        except ValueError as e:
            st.error(f"Input Error: {e}")
        except Exception:
            # General exception handled within get_comments function
            pass

# =============================================================
# TAB 2: STUDENT FEEDBACK FORM
# =============================================================
with tab2:
    st.header("Student Feedback Submission & Analysis 📝")
    st.markdown("Enter student details and feedback. Data is analyzed live and collected for the dashboard.")

    with st.form(key='student_feedback_form'):
        
        col_id, col_course = st.columns(2)
        
        student_id = col_id.text_input("Student ID (e.g., S101)", key="student_id")
        course_options = ["Machine Learning", "Data Structures", "Web Development", "Database Systems"]
        course_name = col_course.selectbox("Course Name", options=course_options, key="course_name")

        feedback_text = st.text_area(
            "Detailed Feedback/Response:", 
            placeholder="I found the practical sessions highly engaging, but the lecture pace was sometimes too fast.",
            height=150,
            key="feedback_text"
        )
        
        submit_button = st.form_submit_button(label="Analyze & Submit Feedback")

    if submit_button:
        if not student_id or not feedback_text:
            st.error("Please fill in the Student ID and the Feedback text before submitting.")
        else:
            with st.spinner(f"Analyzing feedback from {student_id}..."):
                
                raw_sentiment, polarity = analyze_single_text_sentiment(feedback_text)
                
                new_record = {
                    "Date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "ID": student_id,
                    "Course": course_name,
                    "Feedback": feedback_text,
                    "Sentiment": raw_sentiment,
                    "Polarity": f"{polarity:.3f}"
                }
                
                st.session_state.feedback_records.append(new_record)
                
                st.success("Feedback submitted and analyzed successfully!")
                
                st.subheader("Analysis Results")
                colA, colB, colC = st.columns(3)
                colA.metric("Student ID", student_id)
                colB.metric("Course", course_name)
                
                if raw_sentiment == "Positive":
                    colC.success(f"Sentiment: Positive 😊")
                elif raw_sentiment == "Negative":
                    colC.error(f"Sentiment: Negative 😠")
                else:
                    colC.info(f"Sentiment: Neutral 😐")
                
                st.metric(
                    label="Polarity Score (Range: -1.0 to 1.0)",
                    value=f"{polarity:.3f}",
                    delta_color="off"
                )
                
    if st.session_state.feedback_records:
        st.markdown("---")
        df_feedback = pd.DataFrame(st.session_state.feedback_records)
        csv_data = df_feedback.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download All Feedback as CSV",
            data=csv_data,
            file_name='student_sentiment_feedback.csv',
            mime='text/csv',
            key='download_csv_tab2'
        )

# =============================================================
# TAB 3: DASHBOARD (Aggregate Analysis)
# =============================================================
with tab3:
    st.header("Feedback Analysis Dashboard 📈")
    
    if not st.session_state.feedback_records:
        st.info("No feedback data has been submitted yet in this session. Go to the 'Student Feedback Form' tab to enter some records.")
    else:
        df = pd.DataFrame(st.session_state.feedback_records)
        st.success(f"Dashboard analyzing {len(df)} feedback records.")

        # 2. DISPLAY KEY METRICS
        st.subheader("Overall Statistics")
        col1, col2, col3 = st.columns(3)
        
        sentiment_counts = df['Sentiment'].value_counts()
        avg_polarity = df['Polarity'].astype(float).mean()
        
        col1.metric("Total Records", len(df))
        col2.metric("Avg. Polarity Score", f"{avg_polarity:.3f}")
        col3.metric("Most Frequent Sentiment", sentiment_counts.index[0] if not sentiment_counts.empty else "N/A")

        st.markdown("---")
        
        # 3. VISUALIZATION: Sentiment Distribution by Course
        st.subheader("Sentiment Distribution Across Courses")
        
        sentiment_by_course = df.groupby('Course')['Sentiment'].value_counts().unstack(fill_value=0)
        st.bar_chart(sentiment_by_course, height=400)
        
        st.markdown("---")
        
        # 4. VISUALIZATION: Detailed Breakdown
        st.subheader("Detailed Breakdown by Course")
        
        selected_course = st.selectbox(
            "Select a Course for Detailed Review:", 
            options=df['Course'].unique(),
            key='dashboard_course_select'
        )
        
        df_filtered = df[df['Course'] == selected_course]
        
        st.dataframe(df_filtered[['ID', 'Feedback', 'Sentiment', 'Polarity']], use_container_width=True)
        
        course_sentiment_counts = df_filtered['Sentiment'].value_counts()
        
        st.markdown("#### Sentiment Counts for " + selected_course)
        st.bar_chart(course_sentiment_counts)