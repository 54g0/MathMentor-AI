import os
import re
import asyncio
import argparse
from datasets import load_dataset
from agent import MathTutorAgent
def _normalize_choice_text(s: str) -> str:
    """Return a normalized representation for a choice's text to match against model output.
    Prefer numeric extraction (e.g. '784' from '$784$', '784\\,' etc.),
    otherwise return cleaned lowercased text."""
    if s is None:
        return ""

    s = re.sub(r'\$|\\\(|\\\)|\\\[|\\\]', '', s)
    s = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', s)  
    s = re.sub(r'\\[a-zA-Z]+', '', s)          
    s = s.replace(',', '')                      
    s = s.strip()

    m = re.search(r'-?\d+(?:\.\d+)?', s)
    if m:
        return m.group(0)
    return s.lower()


def extract_option(text: str, question: str) -> str:
    """Robustly extract MCQ option 1..4 from model output `text` using `question` options as fallback.
    Returns '1'|'2'|'3'|'4' or 'None'."""
    if not text:
        return "None"
    m = re.search(r'\b(?:option|choice|ans|answer)\s*[:\-]?\s*\(?([1-4])\)?\b', text, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    parentheses_matches = list(re.finditer(r'\(\s*([1-4])\s*\)', text))
    if parentheses_matches:
        for pm in parentheses_matches:
            start_idx = pm.start()
            before = text[max(0, start_idx - 100):start_idx]
            if re.search(r'(final|answer|hence|therefore|so|thus)', before, flags=re.I):
                return pm.group(1)
        last_pm = parentheses_matches[-1]
        after = text[last_pm.end():last_pm.end()+20]
        if not re.search(r'\b(step|step:)\b', after, flags=re.I):
            return last_pm.group(1)
    choices = {}
    for m in re.finditer(r'\(\s*([1-4])\s*\)\s*([^\n\r]*)', question):
        idx, val = m.group(1), m.group(2).strip()
        choices[idx] = val
    if not choices:
        for m in re.finditer(r'([1-4])\)\s*([^\n\r]*)', question):
            idx, val = m.group(1), m.group(2).strip()
            choices[idx] = val
    norm_choices = {opt: _normalize_choice_text(text) for opt, text in choices.items()}

    for opt, norm_val in norm_choices.items():
        if not norm_val:
            continue
        if re.fullmatch(r'-?\d+(?:\.\d+)?', norm_val):
            if re.search(rf'\b{re.escape(norm_val)}\b', text):
                return opt
        else:
            if norm_val.lower() in text.lower():
                return opt
    m = re.search(r'(?:final answer|final|final:|answer:|ans:|hence|therefore|so|thus)[^\d\n\r\-]{0,40}(-?\d+(?:\.\d+)?)', text, flags=re.I)
    if m:
        final_num = m.group(1)
        for opt, norm_val in norm_choices.items():
            if norm_val == final_num:
                return opt
        if final_num in {"1", "2", "3", "4"}:
            return final_num
    last_pos = -1
    last_opt = None
    for opt, norm_val in norm_choices.items():
        if not norm_val:
            continue
        if re.fullmatch(r'-?\d+(?:\.\d+)?', norm_val):
            m = list(re.finditer(rf'\b{re.escape(norm_val)}\b', text))
            if m:
                if m[-1].end() > last_pos:
                    last_pos = m[-1].end()
                    last_opt = opt
        else:
            m = list(re.finditer(re.escape(norm_val), text, flags=re.I))
            if m:
                if m[-1].end() > last_pos:
                    last_pos = m[-1].end()
                    last_opt = opt
    if last_opt:
        return last_opt

    return "None"

async def ask(agent: MathTutorAgent, q: str) -> str:
    return await agent.get_response(q)


def run_async(coro):
    """Safe wrapper to run async coroutines from sync code."""
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        raise

def main():
    parser = argparse.ArgumentParser(description="Simple JEE Mains MCQ benchmark")
    parser.add_argument('--max', type=int, default=20, help='Max questions to evaluate')
    args = parser.parse_args()
    ds = load_dataset("CK0607/2025-Jee-Mains-Question", split='train')

    questions, gold = [], []
    for i, row in enumerate(ds):
        if i >= args.max:
            break
        q = row.get("Question Text") or row.get("Question") or row.get("question")
        a = row.get("Correct Option") or row.get("Answer") or row.get("answer")
        if not q or not a:
            continue
        questions.append(q)
        gold.append(str(a).strip().upper())

    if not questions:
        print("No questions loaded.")
        return

    agent = MathTutorAgent(
        os.getenv("MODEL_PROVIDER", "groq"),
        os.getenv("MODEL_NAME", "groq/deepseek-r1-distill-llama-70b")
    )

    preds, correct = [], 0
    for idx, q in enumerate(questions, start=1):
        print(f"\nQ{idx}: {q}")
        raw = run_async(ask(agent, q))
        print(f"\nRaw Answer: {raw}\n")

        opt = extract_option(raw, q)
        preds.append(opt)
        print(f"Extracted Option: {opt} | Gold: {gold[idx-1]}")

        if opt == gold[idx-1]:
            correct += 1

    total = len(gold)
    acc = correct / total if total else 0.0
    print(f"\n📊 Accuracy: {correct}/{total} = {acc:.2%}")


if __name__ == '__main__':
    main()
