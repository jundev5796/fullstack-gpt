from langchain.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
# from openai import OpenAI
import openai as client
import streamlit as st
import yfinance
import json
import time
import os


class ThreadClient:
    def __init__(self, client):
        self.client = client

    def get_run(self, run_id, thread_id):
        return self.client.beta.threads.runs.retrieve(
            run_id=run_id,
            thread_id=thread_id,
        )

    def send_message(self, thread_id, content):
        return self.client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=content
        )

    def get_messages(self, thread_id):
        messages = self.client.beta.threads.messages.list(thread_id=thread_id)
        messages = list(messages)
        messages.reverse()
        return messages

    def get_tool_outputs(self, run_id, thread_id):
        run = self.get_run(run_id, thread_id)
        outputs = []
        for action in run.required_action.submit_tool_outputs.tool_calls:
            action_id = action.id
            function = action.function
            print(f"Calling function: {function.name} with arg {function.arguments}")
            outputs.append(
                {
                    "output": functions_map[function.name](
                        json.loads(function.arguments)
                    ),
                    "tool_call_id": action_id,
                }
            )
        return outputs

    def submit_tool_outputs(self, run_id, thread_id):
        outputs = self.get_tool_outputs(run_id, thread_id)
        return self.client.beta.threads.runs.submit_tool_outputs(
            run_id=run_id, thread_id=thread_id, tool_outputs=outputs
        )

    def wait_on_run(self, run, thread):
        while run.status == "queued" or run.status == "in_progress":
            run = self.get_run(run.id, thread.id)
            time.sleep(0.5)
        return run


class IssueSearchClient:
    def __init__(self):
        self.wiki = WikipediaAPIWrapper()
        self.ddg = DuckDuckGoSearchAPIWrapper()

    def get_issue(self, issue):
        return self.wiki.run(issue)

    def get_issue_description(self, category):
        return self.ddg.run(category)


functions_map = {
    "get_issue": IssueSearchClient.get_issue,
    "get_issue_description": IssueSearchClient.get_issue_description,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "get_issue",
            "description": "When receiving a category, find recent issues related to that category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "When receiving a category, find recent issues related to that category.",
                    }
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_issue_description",
            "description": "When receiving an issue, give an explanation about the issue",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue": {
                        "type": "string",
                        "description": "Receive an issue.",
                    }
                },
                "required": ["issue"],
            },
        },
    },
]


st.set_page_config(
    page_title="AssistantGPT",
    page_icon="🤖",
)


st.title("Assistant")


api_key = st.sidebar.text_input(
    "**:blue[Enter your OpenAI Key:]**",
)


with st.sidebar:
    st.text("")
    st.text("")
    st.markdown(
        """
        GitHub Repo: https://github.com/jundev5796/fullstack-gpt/blob/master/assistant.py
        """
    )


if api_key and api_key.startswith("sk-"):
    st.session_state["api_key"] = api_key
    client.api_key = api_key
    # client = OpenAI(api_key=api_key)

    assistant_id = "asst_KNw8bVa9WzsXaMUD5BWRm0dv"

    category = st.text_input("**:blue[Enter a Keyword or a Topic:]**")

    if category:
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": f"I would like to know about {category}.",
                }
            ]
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
        )

        assistant = ThreadClient(client)
        run = assistant.wait_on_run(run, thread)

        if run.status == "completed":
            messages = assistant.get_messages(thread.id)

            # Display messages in Streamlit
            conversation = ""
            for message in messages:
                if message.role == "user":
                    conversation += f"Human: {message.content[0].text.value}\n"
                    st.markdown(f"**Human**: {message.content[0].text.value}")
                else:
                    conversation += f"AI: {message.content[0].text.value}\n"
                    st.markdown(f"**AI**: {message.content[0].text.value}")

            # Write conversation to a text file
            with open(f"{category}_conversation.txt", "w", encoding="utf-8") as file:
                file.write(conversation)

            # Download conversation as txt file
            with open(f"{category}_conversation.txt", "r", encoding="utf-8") as file:
                st.download_button(
                    label="Download Conversation",
                    data=file,
                    file_name=f"{category}_conversation.txt",
                    mime="text/plain",
                )

            # Remove the temporary file
            os.remove(f"{category}_conversation.txt")