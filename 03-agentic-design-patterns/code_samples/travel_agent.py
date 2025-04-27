import json
import os

from typing import Annotated

from dotenv import load_dotenv

from IPython.display import display, HTML

from openai import AsyncOpenAI

from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import FunctionCallContent, FunctionResultContent, StreamingTextContent
from semantic_kernel.functions import kernel_function

import random   

# Define a sample plugin for the sample

class DestinationsPlugin:
    """A List of Random Destinations for a vacation."""

    def __init__(self):
        # List of vacation destinations
        self.destinations = [
            "Barcelona, Spain",
            "Paris, France",
            "Berlin, Germany",
            "Tokyo, Japan",
            "Sydney, Australia",
            "New York, USA",
            "Cairo, Egypt",
            "Cape Town, South Africa",
            "Rio de Janeiro, Brazil",
            "Bali, Indonesia"
        ]
        # Track last destination to avoid repeats
        self.last_destination = None

    @kernel_function(description="Provides a random vacation destination.")
    def get_random_destination(self) -> Annotated[str, "Returns a random vacation destination."]:
        # Get available destinations (excluding last one if possible)
        available_destinations = self.destinations.copy()
        if self.last_destination and len(available_destinations) > 1:
            available_destinations.remove(self.last_destination)

        # Select a random destination
        destination = random.choice(available_destinations)

        # Update the last destination
        self.last_destination = destination

        return destination

        load_dotenv()
client = AsyncOpenAI(
    api_key=os.environ.get("GITHUB_TOKEN"), 
    base_url="https://models.inference.ai.azure.com/",
)

# Create an AI Service that will be used by the `ChatCompletionAgent`
chat_completion_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o-mini",
    async_client=client,
)

AGENT_INSTRUCTIONS = """You are a helpful AI Agent that can help plan vacations for customers.

Important: When users specify a destination, always plan for that location. Only suggest random destinations when the user hasn't specified a preference.

When the conversation begins, introduce yourself with this message:
"Hello! I'm your TravelAgent assistant. I can help plan vacations and suggest interesting destinations for you. Here are some things you can ask me:
1. Plan a day trip to a specific location
2. Suggest a random vacation destination
3. Find destinations with specific features (beaches, mountains, historical sites, etc.)
4. Plan an alternative trip if you don't like my first suggestion

What kind of trip would you like me to help you plan today?"

Always prioritize user preferences. If they mention a specific destination like "Bali" or "Paris," focus your planning on that location rather than suggesting alternatives.
"""

agent = ChatCompletionAgent(
    service=chat_completion_service, 
    plugins=[DestinationsPlugin()],
    name="TravelAgent",
    instructions=AGENT_INSTRUCTIONS,
)

user_inputs = [
    "Plan me a day trip.",
    "I don't like that destination. Plan me another vacation.",
]

async def main():
    thread: ChatHistoryAgentThread | None = None

    for user_input in user_inputs:
        html_output = (
            f"<div style='margin-bottom:10px'>"
            f"<div style='font-weight:bold'>User:</div>"
            f"<div style='margin-left:20px'>{user_input}</div></div>"
        )

        agent_name = None
        full_response: list[str] = []
        function_calls: list[str] = []

        # Buffer pour les appels de fonction
        current_function_name = None
        argument_buffer = ""

        async for response in agent.invoke_stream(
            messages=user_input,
            thread=thread,
        ):
            thread = response.thread
            agent_name = response.name
            content_items = list(response.items)

            for item in content_items:
                if isinstance(item, FunctionCallContent):
                    if item.function_name:
                        current_function_name = item.function_name
                    if isinstance(item.arguments, str):
                        argument_buffer += item.arguments
                elif isinstance(item, FunctionResultContent):
                    if current_function_name:
                        formatted_args = argument_buffer.strip()
                        try:
                            parsed_args = json.loads(formatted_args)
                            formatted_args = json.dumps(parsed_args)
                        except Exception:
                            pass  # laisser tel quel

                        function_calls.append(f"Calling function: {current_function_name}({formatted_args})")
                        current_function_name = None
                        argument_buffer = ""

                    function_calls.append(f"\nFunction Result:\n\n{item.result}")
                elif isinstance(item, StreamingTextContent) and item.text:
                    full_response.append(item.text)

        if function_calls:
            html_output += (
                "<div style='margin-bottom:10px'>"
                "<details>"
                "<summary style='cursor:pointer; font-weight:bold; color:#0066cc;'>Function Calls (click to expand)</summary>"
                "<div style='margin:10px; padding:10px; background-color:#f8f8f8; "
                "border:1px solid #ddd; border-radius:4px; white-space:pre-wrap; font-size:14px; color:#333;'>"
                f"{chr(10).join(function_calls)}"
                "</div></details></div>"
            )

        html_output += (
            "<div style='margin-bottom:20px'>"
            f"<div style='font-weight:bold'>{agent_name or 'Assistant'}:</div>"
            f"<div style='margin-left:20px; white-space:pre-wrap'>{''.join(full_response)}</div></div><hr>"
        )

        display(HTML(html_output))

await main()
