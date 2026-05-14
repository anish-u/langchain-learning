from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-5-nano")

class Person(BaseModel):
    name: str = Field(description="Name of person")
    age: int = Field(gt=18, description="Age of person")
    city: str = Field(description="Name of the city the person is in")

parser = PydanticOutputParser(pydantic_object=Person)

# Detailed report Prompt
template = PromptTemplate(
    template="Give me the name, age and city of a fictional {text} character. \n {format_instruction}",
    input_variables=['text'],
    partial_variables={'format_instruction': parser.get_format_instructions()},
)

# prompt = template.invoke({'text': 'Indian'})
# result = model.invoke(prompt)
#
# final_result = parser.parse(result.content)
# print(final_result)

chain = template | model | parser
result = chain.invoke({'text': 'British'})
print(result)