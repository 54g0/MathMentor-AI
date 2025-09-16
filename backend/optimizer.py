import dspy
import asyncio
from dspy.teleprompt import BootstrapFewShot
from agent import MathTutorAgent
from typing import Optional

def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        return asyncio.ensure_future(coro)
    return asyncio.run(coro)

class MathTutorModule(dspy.Module):
    """Wraps MathTutorAgent for DSPy. Provides both async and sync forward."""
    def __init__(self, math_agent: MathTutorAgent):
        super().__init__()
        self.math_agent = math_agent

    async def a_forward(self, question: str):
        try:
            response = await self.math_agent.get_response(question)
            return dspy.Prediction(answer=response)
        except Exception as e:
            return dspy.Prediction(answer=f"Error: {e}")
    def forward(self, question: str):
        result = run_async(self.a_forward(question))
        if asyncio.isfuture(result):
            try:
                if result.done():
                    return result.result()
            except Exception as e:
                return dspy.Prediction(answer=f"Error: {e}")
            return dspy.Prediction(answer="(Pending async execution – please await)")
        return result

class FeedbackAgent:
    def __init__(self, tutor_module: MathTutorModule):
        self.data_collection_module = tutor_module
        self.examples: list[dspy.Example] = []

    async def ask(self, question: str):
        prediction = await self.data_collection_module.a_forward(question)
        ans = getattr(prediction, 'answer', '') or ''
        print(f"\n🤖 Initial Answer: {ans}")

        feedback_text = input("🧑‍🎓 Your Feedback: ")
        ex = dspy.Example(question=question, answer=ans, feedback=feedback_text).with_inputs("question")
        self.examples.append(ex)
        print("✅ Feedback recorded.")
        return ans

    def optimize_with_bootstrap(self):
        if not self.examples:
            print("🛑 No examples to optimize.")
            return None
        print("\n🚀 Optimizing with BootstrapFewShot...")
        optimizer = BootstrapFewShot(
            metric=self.simple_metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=10,
            max_rounds=2,
        )
        optimized_module = optimizer.compile(self.data_collection_module, trainset=self.examples)
        print("✅ BootstrapFewShot optimization complete.")
        return optimized_module

    @staticmethod
    def simple_metric(example, pred, trace=None): 
        if pred is None:
            return 0.0
        ans = getattr(pred, 'answer', '')
        return 1.0 if isinstance(ans, str) and ans.strip() and not ans.startswith("Error:") else 0.0


async def main():
    print("--- Phase 1: Data Collection ---")
    math_agent = MathTutorAgent(
        model_provider="groq",
        model_name="openai/gpt-oss-120b",  
        debug=True
    )
    feedback_agent = FeedbackAgent(MathTutorModule(math_agent))

    while True:
        question = input("\nEnter a math question (or 'optimize'): ")
        if question.lower() == "optimize":
            break
        if not question.strip():
            continue
        await feedback_agent.ask(question)

    optimized_tutor = feedback_agent.optimize_with_bootstrap()
    if optimized_tutor:
        print("\n--- Phase 3: Testing Optimized Tutor ---")
        while True:
            question = input("\n[Optimized Agent] Ask a question (or 'exit'): ")
            if question.lower() == "exit":
                break
            response = optimized_tutor.forward(question=question)
            ans = getattr(response, 'answer', '(no answer)')
            print(f"\n🤖 Optimized Answer: {ans}")

if __name__ == "__main__":
    asyncio.run(main())
