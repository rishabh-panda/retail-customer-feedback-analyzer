import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

# Test Zhipu connection
api_key = os.getenv("ZHIPU_API_KEY")
print(f"API Key found: {api_key[:10]}...{api_key[-10:] if api_key else 'MISSING'}")

if not api_key:
    print("No API key found in .env file")
    exit(1)

try:
    llm = ChatOpenAI(
        model="glm-4.7-flash",
        temperature=0,
        openai_api_key=api_key,
        openai_api_base="https://open.bigmodel.cn/api/paas/v4"
    )
    
    response = llm.invoke([HumanMessage(content="Say 'Hello from Zhipu API!' in JSON format: {'message': 'your response'}")])
    print(f"Connection successful! Response: {response.content}")
    
except Exception as e:
    print(f"Connection failed: {e}")