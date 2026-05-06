import json

from openai import OpenAI

from agent.llm_config import DEEPSEEK_CHAT_MODEL
from agent.state import AgentRuntimeState


def _extract_json_content(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return raw


def extract_memory_update(client: OpenAI, state: AgentRuntimeState, last_result: str) -> dict:
    facts_payload = json.dumps(state.facts, ensure_ascii=False)
    working_payload = json.dumps(state.working_memory, ensure_ascii=False)
    messages = [
        {
            "role": "system",
            "content": (
                "你是记忆抽取器。"
                "请从当前执行结果中提取可复用结论 facts，并更新 working_memory。"
                "只输出 JSON，格式: "
                "{\"facts\": {...}, \"working_memory_update\": {...}}。"
                "不要编造，不要重复已有事实。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"目标:\n{state.goal}\n\n"
                f"已有 facts:\n{facts_payload}\n\n"
                f"已有 working_memory:\n{working_payload}\n\n"
                f"当前结果:\n{str(last_result or '')}\n\n"
                "请输出 JSON。"
            ),
        },
    ]
    try:
        response = client.chat.completions.create(model=DEEPSEEK_CHAT_MODEL, messages=messages)
        content = str(response.choices[0].message.content or "")
        parsed = json.loads(_extract_json_content(content))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {"facts": {}, "working_memory_update": {}}


def apply_memory_update(state: AgentRuntimeState, update: dict) -> None:
    if not isinstance(update, dict):
        return
    state.merge_facts(update.get("facts") or {})
    state.update_working_memory(update.get("working_memory_update") or {})
