import re
import pandas as pd
import plotly.graph_objects as go
from PyPDF2 import PdfReader
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(layout="wide")
st.markdown("""<div class="centered-title">
    <h1 class="gradient-text">Lab Report Analyzer</h1>
</div>
""", unsafe_allow_html=True)

def load_css(file_path):
    with open(file_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Construct path relative to root or use absolute
css_path = os.path.join("assets", "report_analyzer.css")
print(css_path)
try:
    load_css(css_path)
except FileNotFoundError:
    st.warning("report_analyzer.css file not found.")

lab_file = st.file_uploader("Upload a lab report PDF", type=["pdf"], key="lab_report")

def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def parse_lab_report(text):
    pattern = re.compile(
        r'([a-zA-Z%/µ.^0-9]+)\s+'                # Unit
        r'([\d.]+|Nil|%Nil)\s+'                  # Value
        r'(<|>|≤|≥)?([\d.]+)\s*-\s*([\d.]+)?'    # Min - Max
        r'([A-Z0-9 \-()/%]+)'                    # Test Name
    )
    matches = pattern.findall(text)
    results = []

    for unit, value, comparator, ref_min, ref_max, test_name in matches:
        try:
            numeric_value = 0.0 if value in ['Nil', '%Nil'] else float(value)
            ref_min_f = float(ref_min)
            ref_max_f = float(ref_max) if ref_max else None
        except:
            numeric_value, ref_min_f, ref_max_f = 0.0, None, None

        status = "Normal"
        if ref_min_f is not None and numeric_value < ref_min_f:
            status = "Deficiency"
        elif ref_max_f is not None and numeric_value > ref_max_f:
            status = "Excess"

        results.append({
            "Test Name": test_name.strip(),
            "Value": numeric_value,
            "Unit": unit.strip(),
            "Reference Range": f"{ref_min} - {ref_max}" if ref_max else f">{ref_min}",
            "Status": status
        })

    return pd.DataFrame(results)

def get_lab_summary_text(excess_df, deficiency_df):
    summary = ""

    if not excess_df.empty:
        summary += "Excess Parameters:\n"
        for _, row in excess_df.iterrows():
            summary += f"- {row['Test Name']} = {row['Value']} {row['Unit']} (Ref: {row['Reference Range']})\n"

    if not deficiency_df.empty:
        summary += "\nDeficiency Parameters:\n"
        for _, row in deficiency_df.iterrows():
            summary += f"- {row['Test Name']} = {row['Value']} {row['Unit']} (Ref: {row['Reference Range']})\n"

    return summary.strip()

# -----------------------------------
# 🧾 Main Execution
# -----------------------------------
if lab_file:
    extracted_text = extract_text_from_pdf(lab_file)
    df = parse_lab_report(extracted_text)
    
    if not df.empty:
        st.success("✅ Extracted Lab Results")      
        st.markdown("### 📋 Lab Test Results")
        st.dataframe(df)

        st.markdown("### 🔴 Elevated Markers")
        excess_df = df[df["Status"] == "Excess"]
        if not excess_df.empty:
            st.dataframe(excess_df[["Test Name", "Value", "Unit", "Reference Range"]])
        else:
            st.info("✅ No parameters in excess.")

        st.markdown("### 🔵 Insufficient Levels")
        deficiency_df = df[df["Status"] == "Deficiency"]
        if not deficiency_df.empty:
            st.dataframe(deficiency_df[["Test Name", "Value", "Unit", "Reference Range"]])
        else:
            st.info("✅ No parameters in deficiency.")

        st.markdown("### 📊 Visual Summary ")
        color_map = {
            "Excess": "red",
            "Deficiency": "lightblue",
            "Normal": "green"
        }

        # Create a new column for the color based on 'Status'
        df['Color'] = df['Status'].map(color_map)

        # Create the bar chart with all the results
        fig = go.Figure([ 
            go.Bar(
                x=df["Test Name"],
                y=df["Value"],
                marker_color=df["Color"],  # Use the Color column to assign colors
                text=df["Unit"],
                textposition="auto"
            )
        ])

        # Update the layout
        fig.update_layout(
            xaxis_tickangle=-45, 
            yaxis_title="Value", 
            height=600,  # Increased height to accommodate all results
            title="Lab Test Results"
        )

        # Display the Plotly chart
        st.plotly_chart(fig)

        # ------------------------------
        # 💬 Gemini Flash Chatbot
        # ------------------------------
        st.markdown("---")
        st.markdown("## 🤖 Ask Questions Based on Lab Report")

        summary_text = get_lab_summary_text(excess_df, deficiency_df)

        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.4,
            google_api_key=api_key
        )

        # Initialize conversation history
        if "lab_chat_history" not in st.session_state:
            st.session_state.lab_chat_history = []

        user_input = st.chat_input("Ask anything about the lab results...")

        # System context prompt
        system_msg = f"You are a medical assistant. Based on this patient's abnormal lab results, answer the user's question or provide insights.\n\n{summary_text}"

        for message in st.session_state.lab_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_input:
            st.session_state.lab_chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            response = model.invoke([
                HumanMessage(content=f"{system_msg}\n\nUser Question: {user_input}")
            ])

            reply = response.content
            st.session_state.lab_chat_history.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

    else:
        st.warning("⚠ No lab test data could be extracted. Try a different report format.")
