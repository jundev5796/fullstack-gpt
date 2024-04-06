import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.retrievers import WikipediaRetriever
from langchain.callbacks import StreamingStdOutCallbackHandler

st.set_page_config(
  page_title="QuizGPT",
  page_icon="❓",
)

st.title("QuizGPT")

llm = ChatOpenAI(
  temperature=0.1,
  model="gpt-3.5-turbo-0125",
  streaming=True,
  callbacks=[StreamingStdOutCallbackHandler()],
)


def save_message(message, role):
  st.session_state["messages"].append({"message": message, "role": role})

class ChatCallbackHandler(BaseCallbackHandler):
  message = ""

  def on_llm_start(self, *args, **kwargs):
    self.message_box = st.empty()

  def on_llm_end(self, *args, **kwargs):
    save_message(self.message, "ai")

  def on_llm_new_token(self, token, *args, **kwargs):
    self.message += token
    self.message_box.markdown(self.message)


# File splitter
@st.cache_data(show_spinner="Loading file...")
def split_file(file):
  file_content = file.read()
  file_path = f"./.cache/quiz_gpt_files/{file.name}"
  with open(file_path, "wb") as f:
    f.write(file_content)
  splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
  )
  loader = UnstructuredFileLoader(file_path)
  docs = loader.load_and_split(text_splitter=splitter)
  return docs


# Sidebar
with st.sidebar:
  api_key = st.text_input("Input your OpenAI Key")
  button = st.button("Save Key")
  docs = None
  choice = st.selectbox(
    "Choose what you want to use.",
      (
        "File",
        "Wikipedia Article",
      ),
  )
  if choice == "File":
    file = st.file_uploader(
      "Upload a .docx , .txt or .pdf file",
      type=["pdf", "txt", "docx"],
    )
    if file:
      docs = split_file(file)
  else:
    topic = st.text_input("Search Wikipedia...")
    if topic:
      retriever = WikipediaRetriever(top_k_results=5)
      with st.status("Searching Wikipedia..."):
        docs = retriever.get_relevant_documents(topic)
  github_link = st.sidebar.markdown("https://github.com/jundev5796/fullstack-gpt/blob/master/quiz_gpt.py")

  # if button:
  #   save_api_key(api_key)
  #   st.write(f"API_KEY = {"api_key}")
  #   if openai_key == "":
  #     st.warning("Please correctly input OpenAI Key")

if api_key:
  llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
      ChatCallbackHandler(),
    ],
    openai_api_key=st.session_state["api_key"]
  )
else:
  st.warning("OpenAI Key not found.")


if not docs:
  st.markdown(
  """
  Welcome to QuizGPT.
                
  I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
  Get started by uploading a file or searching on Wikipedia in the sidebar.
  """
  )
else:
  dificulty = st.selectbox(
    "Select Difficulty", 
      (
        "Easy",
        "Hard",
      )
  )