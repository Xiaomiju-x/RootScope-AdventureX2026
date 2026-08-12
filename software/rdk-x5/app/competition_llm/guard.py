"""Prompt-injection and command-shaped text guard for untrusted inputs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Tuple


_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "INSTRUCTION_OVERRIDE",
        re.compile(
            r"(?:ignore|disregard|forget)\s+(?:all\s+|the\s+)?"
            r"(?:previous|prior|above|system).{0,32}(?:instruction|rule|prompt)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "INSTRUCTION_OVERRIDE",
        re.compile(
            r"忽略.{0,20}(?:以上|之前|先前|系统).{0,20}(?:指令|规则|提示词)",
            re.DOTALL,
        ),
    ),
    (
        "ROLE_OVERRIDE",
        re.compile(
            r"(?:you\s+are\s+now|act\s+as\s+(?:the\s+)?system|"
            r"你现在(?:是|作为)|扮演.{0,12}(?:系统|管理员))",
            re.IGNORECASE,
        ),
    ),
    (
        "SYSTEM_PROMPT_EXFILTRATION",
        re.compile(
            r"(?:(?:show|print|reveal|leak).{0,30}(?:system|developer).{0,12}prompt|"
            r"(?:显示|打印|泄露|输出).{0,20}(?:系统|开发者).{0,12}(?:提示词|指令))",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "DELIMITER_SMUGGLING",
        re.compile(
            r"(?:<\s*/?\s*tool(?:_call)?\s*>|\[(?:system|developer|assistant)\]|"
            r"<\|(?:system|assistant|tool)\|>)",
            re.IGNORECASE,
        ),
    ),
    (
        "COMMAND_REQUEST",
        re.compile(
            r"(?:\bsudo\b|\brm\s+-rf\b|\bcurl\s+https?://|\bwget\s+https?://|"
            r"\bpowershell(?:\.exe)?\b|\bcmd\.exe\b|"
            r"(?:执行|运行).{0,12}(?:shell|命令|工具调用)|"
            r"(?:启动|打开|写入|触发).{0,12}(?:水泵|串口|GPIO))",
            re.IGNORECASE,
        ),
    ),
    (
        "ENCODED_INSTRUCTION",
        re.compile(
            r"(?:base64\s+(?:decode|解码)|"
            r"(?:decode|解码).{0,16}base64|"
            r"\b[A-Za-z0-9+/]{160,}={0,2}\b)",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class GuardAssessment:
    blocked: bool
    reasons: Tuple[str, ...]


def assess_untrusted_text(text: str) -> GuardAssessment:
    if not isinstance(text, str):
        return GuardAssessment(True, ("NON_TEXT_INPUT",))
    reasons: list[str] = []
    for name, pattern in _PATTERNS:
        if pattern.search(text) is not None and name not in reasons:
            reasons.append(name)
    return GuardAssessment(bool(reasons), tuple(reasons))
