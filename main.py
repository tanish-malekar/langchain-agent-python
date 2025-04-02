import os
os.environ['GRPC_DNS_RESOLVER'] = 'native'

from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables
load_dotenv()


print(os.environ)


from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()


# Define conversation state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize graph
graph_builder = StateGraph(State)

# Initialize tools
tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Define chatbot function
def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

# Add nodes
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile the graph
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

# Function to handle user input
def stream_graph_updates(user_input: str):
    print("User:", user_input)
    try:
        for event in graph.stream({"messages": [{"role": "user", "content": user_input}]},     config):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)
    except Exception as e:
        print("Error during graph.stream execution:", e)

# Main loop
if __name__ == "__main__":
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
