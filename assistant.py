from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
import openai as client
import yfinance
import json
import streamlit as st

st.title("Assistant")

with st.sidebar:
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""

    api_key_input = st.empty()

    def reset_api_key():
        st.session_state["api_key"] = ""
        print(st.session_state["api_key"])

    if st.button(":red[Reset API_KEY]"):
        reset_api_key()

    api_key = api_key_input.text_input(
        "**:blue[OpenAI API_KEY]**",
        value=st.session_state["api_key"],
        key="api_key_input",
    )

    if api_key != st.session_state["api_key"]:
        st.session_state["api_key"] = api_key
        st.rerun()

    # print(api_key)

    url = st.text_input(
        "**:blue[Write down a URL]**",
        placeholder="https://example.com",
        value="https://developers.cloudflare.com/sitemap.xml",
    )
    url_name = url.split("://")[1].split("/")[0] if url else None

    st.markdown(
        """
        GitHub Repo: https://github.com/jundev5796/fullstack-gpt/blob/master/SiteGPT.py
        """
    )