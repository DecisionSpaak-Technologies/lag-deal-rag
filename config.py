import os
import dotenv

# Set up environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = dotenv.get_key('.env', 'LANGCHAIN_API_KEY')
os.environ["OPENAI_API_KEY"] = dotenv.get_key('.env', 'OPENAI_API_KEY')
os.environ["LANGSMITH_ENDPOINT"] = 'https://api.smith.langchain.com'
os.environ["LANGSMITH_PROJECT"] = dotenv.get_key('.env', 'LANGSMITH_PROJECT')