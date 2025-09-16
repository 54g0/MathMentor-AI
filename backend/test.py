import dspy
import os
lm = dspy.LM(
    model = "openai/gpt-oss-120b",
    temperature = 0.7,
    api_key = os.getenv("GROQ_API_KEY")
)
dspy.configure(lm=lm)
print(dspy.ChainOfThought(question="12 + 15 = ?"))