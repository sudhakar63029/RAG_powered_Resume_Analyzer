import streamlit as st
from api_client import send_resume_for_analysis

st.set_page_config(page_title="Dataset-Powered RAG Resume Analyzer", layout="centered")

st.title("📄 Full-Stack Dataset-Powered RAG Resume Analyzer")
st.write("Upload your resume and paste a target job description to match it against historical dataset insights and expert ATS guidelines.")

job_description = st.text_area("Paste Target Job Description:")
uploaded_file = st.file_uploader("Upload Resume (PDF format)", type=["pdf"])

if st.button("Run AI Analysis"):
    if uploaded_file and job_description:
        with st.spinner("Retrieving historical dataset benchmarks and running RAG evaluation..."):
            result = send_resume_for_analysis(
                job_description=job_description,
                file_name=uploaded_file.name,
                file_bytes=uploaded_file.getvalue()
            )
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Analysis Complete!")
                
                
                st.subheader("💡 RAG Evaluation & Recommendations")
                st.markdown(result["feedback"])
    else:
        st.warning("Please provide both a job description and upload a resume file.")