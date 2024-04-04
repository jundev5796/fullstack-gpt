import streamlit as st

st.set_page_config(
  page_title="QuizGPT",
  page_icon="❓",
)

st.title("QuizGPT")

with st.sidebar:
  choice = st.selectbox(
    "Choose what you want to use.", 
    (
      "File",
      "Wikipedia Article", 
    ),
  )
  if choice == "File":
    file = st.file_uploader(
      "Upload a .docx, .txt or .pdf file",
      type=["pdf", "txt", "docx"],
    )
  else:
    topic = st.text_input("Name of the article")