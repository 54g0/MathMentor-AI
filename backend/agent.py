from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from model import Model
from pylatexenc.latex2text import LatexNodes2Text
from typing import Optional
import asyncio
import json
from langchain.tools import tool
import re  # <-- add

load_dotenv()

# --------- Add these helpers (HTML and Unicode superscripts) ---------
_SUP_MAP = {
    '0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹',
    '+':'⁺','-':'⁻','(':'⁽',')':'⁾','n':'ⁿ','i':'ⁱ'
}

def caret_to_html_sup(s: str) -> str:
    # Matches base^exp where exp is {…} or a word/number/sign sequence
    # Examples: x^3, x^{n+1}, f(x)^2, y^-3
    pattern = re.compile(r'(?P<base>(?:\w|\)|\]))\^(?P<exp>\{[^}]+\}|-?[\w()+-]+)')
    def _repl(m):
        base = m.group('base')
        exp = m.group('exp')
        clean = exp[1:-1] if exp.startswith('{') and exp.endswith('}') else exp
        return f'{base}<sup>{clean}</sup>'
    return pattern.sub(_repl, s)

def caret_to_unicode_sup(s: str) -> str:
    # Conservative pass: convert only if exponent consists entirely of mappable chars
    pattern = re.compile(r'(?P<base>(?:\w|\)|\]))\^(?P<exp>\{[^}]+\}|-?[\w()+-]+)')
    def _map_exp(exp: str) -> Optional[str]:
        clean = exp[1:-1] if exp.startswith('{') and exp.endswith('}') else exp
        # Allow only characters we can map; otherwise, skip conversion
        if all(ch in _SUP_MAP for ch in clean):
            return ''.join(_SUP_MAP[ch] for ch in clean)
        # Fast path: pure digits (possibly signed) — very common case
        if re.fullmatch(r'-?\d+', clean):
            return ''.join(_SUP_MAP[ch] for ch in clean)
        return None
    def _repl(m):
        base = m.group('base')
        exp = m.group('exp')
        mapped = _map_exp(exp)
        return f'{base}{mapped}' if mapped is not None else m.group(0)
    return pattern.sub(_repl, s)
# ---------------------------------------------------------------------


class MathTutorAgent:
    def __init__(self, model_provider: str, model_name: Optional[str] = None, exponent_render: str = "unicode"):
        self.model_provider = model_provider
        self.model_name = model_name
        self.exponent_render = exponent_render  # "unicode" or "html"
        self.llm = Model(model_provider=model_provider, model_name=model_name).create_model()
        self.system_prompt = """You are MathMentor AI. For EVERY math question, you MUST call tools in this EXACT order BEFORE ANY solving. DO NOT SKIP or solve directly—ALWAYS start with retrieve_data.


MANDATORY SEQUENCE (NO EXCEPTIONS, EVEN FOR SIMPLE PROBLEMS):
1. Check if the input question is related to the mathematics field or a problem in maths if yes then continue the process if not strictly stop the process and respond with "Please ask only mathematical questions."
2. Call retrieve_data with the query.
3. If insufficient, call web_search.
4. Solve step-by-step USING TOOL RESULTS.
5. Give final answer with step by step solution and reasoning.


EXAMPLE 1 (FOLLOW EXACTLY):
User: Solve x + 1 = 2
Thought: First, call retrieve_data.
Tool Call: retrieve_data(query="solve x + 1 = 2")
[Use result if required]
If needed: web_search(query="solve linear equation x + 1 = 2")
Solve: x = 1
Reasoning: answer the question step by step
Final: x=1
NEVER give answers without tools. If non-math, respond: "Please ask only mathematical questions." """

    def _render_exponents(self, text: str) -> str:
        if self.exponent_render == "html":
            return caret_to_html_sup(text)
        return caret_to_unicode_sup(text)

    async def get_response(self, question: str):
        self.current_question = question
        client = MultiServerMCPClient({
            "mcp_server": {
                "url": "http://127.0.0.1:8001/mcp",
                "transport": "streamable_http"
            }
        })
        
        try:
            mcp_tools = await client.get_tools()
            llm_with_tools = self.llm.bind_tools(mcp_tools)
            agent = create_react_agent(
                model=llm_with_tools,
                tools=mcp_tools,
            )
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question)
            ]

            print("\nPROCESSING:\n")
            async for event in agent.astream({"messages": messages}):
                for key, value in event.items():
                    if key == "agent" and "messages" in value:
                        for msg in value["messages"]:
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    print(f"\n🔨 CALL: {tc['name']} args: {json.dumps(tc['args'], indent=2)}")
                            elif hasattr(msg, 'content') and msg.content:
                                print(f"THINKING: {msg.content[:200]}...")
                    elif key == "tools" and "messages" in value:
                        for msg in value["messages"]:
                            if hasattr(msg, 'content') and hasattr(msg, 'tool_call_id'):
                                print(f" RESPONSE (ID: {msg.tool_call_id}): {msg.content[:500]}...")

            result = await agent.ainvoke({"messages": messages})
            final_message = result["messages"][-1]
            final_content = final_message.content if hasattr(final_message, 'content') else str(final_message)
            # Existing LaTeX -> text pass
            text = LatexNodes2Text().latex_to_text(final_content)
            # New exponent rendering
            return self._render_exponents(text)

        except Exception as e:
            return f"Error: {str(e)}"


class feedbackAgent:
    def __init__(self, model_provider: str, model_name: Optional[str] = None, exponent_render: str = "unicode"):
        self.model_provider = model_provider
        self.model_name = model_name
        self.exponent_render = exponent_render  # "unicode" or "html"
        self.llm = Model(model_provider=model_provider, model_name=model_name).create_model()
        
        self.system_prompt = """You are MathMentor. 
You take the feedback from the user and improve the answer based on the feedback.
No need to call tools or remember past history. 
Just refine the given answer based on the feedback provided."""

    def _render_exponents(self, text: str) -> str:
        if self.exponent_render == "html":
            return caret_to_html_sup(text)
        return caret_to_unicode_sup(text)
    
    def get_feedback_answer(self, question: str, answer: str, feedback: str):
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Question: {question}\nAnswer: {answer}\nFeedback: {feedback}\nImprove the answer based on the feedback.")
        ]
        try:
            response = self.llm.invoke(messages)
            content = getattr(response, 'content', None)
            if isinstance(content, list):
                content = ' '.join([c.get('text','') if isinstance(c, dict) else str(c) for c in content])
            # Existing LaTeX -> text pass
            text = LatexNodes2Text().latex_to_text(content) or LatexNodes2Text().latex_to_text(str(response))
            # New exponent rendering
            return self._render_exponents(text)
        except Exception as e:
            return f"Error: {str(e)}"


async def main():
    math_agent = MathTutorAgent(
        model_provider="groq",
        model_name="openai/gpt-oss-120b",
        exponent_render="unicode",  # or "html"
    )
    response = await math_agent.get_response("who is elon musk?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
