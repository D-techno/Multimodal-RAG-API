from app.generation import LLMGenerator

llm = LLMGenerator()

response = llm.generate(
    context="Cisco builds networking hardware like routers and switches.",
    question="What does Cisco manufacture?"
)

print(response)