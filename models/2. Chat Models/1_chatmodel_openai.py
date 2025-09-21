from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model_name="gpt-4", temperature=2, max_completion_tokens=500)

result = model.invoke("Suggest three names for an AI startup.")

print(result.content)