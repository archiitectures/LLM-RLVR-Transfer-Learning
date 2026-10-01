"""Explicit Qwen XML tool protocol shared by training and evaluation.

TRL 0.29 uses the legacy tokenizer.parse_response(ids) interface. Install a
model-protocol adapter instead of depending on Transformers' changing schema API.
"""

import json
import re
from types import MethodType


def parse_tool_response(text: str) -> dict:
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    calls = []
    for block in blocks:
        value = json.loads(block)
        if (
            not isinstance(value, dict)
            or value.get("name") not in {"execute_python", "submit_answer"}
            or not isinstance(value.get("arguments"), dict)
        ):
            raise ValueError("Invalid tool call")
        calls.append({"type": "function", "function": value})
    content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S)
    result = {"role": "assistant", "content": content}
    if calls:
        result["tool_calls"] = calls
    return result


def _parse_response(tokenizer, ids, **kwargs):
    text = ids if isinstance(ids, str) else tokenizer.decode(ids, skip_special_tokens=True)
    return parse_tool_response(text)


def configure_tool_tokenizer(tokenizer):
    if "<tool_call>" not in (tokenizer.chat_template or ""):
        raise ValueError("Tool arms require a Qwen XML tool-call chat template")
    # TRL checks this attribute before calling the explicit bound parser below.
    tokenizer.response_schema = {"transferlab_protocol": "qwen_xml_v1"}
    tokenizer.parse_response = MethodType(_parse_response, tokenizer)
    return tokenizer
