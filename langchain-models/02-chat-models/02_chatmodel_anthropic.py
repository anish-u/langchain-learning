from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model_name="claude-3-5-haiku-20241022")
result = model.invoke("Write is the capital of India")

print(result.content)