from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from openai import OpenAI



@dataclass
class Segment:
    text: str
    label: str 
    reasons: List[str]
    math_check: Optional[str] = None


@dataclass
class PromptResult:
    prompt: str
    response: str
    deterministic: List[Segment]
    interpretive: List[Segment]
    unknown: List[Segment]
    counts: Dict[str, int]


INTERPRETIVE_WORDS = {
    "likely", "possibly", "maybe", "might", "could", "suggests", "appears",
    "probably", "seems", "recommend", "should", "advice", "best", "consider",
    "in my opinion", "it may", "it could", "it seems", "practical guidance",
    "contributing factor", "is important", "generally", "typically"
}

FORMAL_LOGIC_MARKERS = {
    "if", "then", "therefore", "thus", "hence", "premise", "conclusion",
    "cannot conclude", "does not follow", "invalid", "valid",
    "insufficient evidence", "correlation", "causation"
}

SYSTEM_NOISE_PATTERNS = [
    r"^#{1,6}\s+",
    r"^[-*_]{3,}$",
    r"^```.*$",
    r"^\*\s+",
]


def preprocess_text(text: str) -> str:
    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        skip = False
        for pat in SYSTEM_NOISE_PATTERNS:
            if re.match(pat, line):
                skip = True
                break
        if skip:
            continue

        line = re.sub(r"^\*\s+", "", line)
        lines.append(line)

    cleaned = " ".join(lines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def split_sentences(text: str) -> List[str]:
    """
    Conservative sentence split to reduce over-fragmentation.
    """
    cleaned = preprocess_text(text)
    if not cleaned:
        return []

    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z(\[])', cleaned)
    segments = [p.strip() for p in parts if len(p.strip()) >= 4]
    return segments


def check_binary_equation(sentence: str) -> Optional[bool]:
    """
    Check equations like:
      a + b = c
      a - b = c
      a * b = c
      a / b = c
    """
    s = sentence.replace(",", "")
    m = re.search(
        r'(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)',
        s
    )
    if not m:
        return None

    a = float(m.group(1))
    op = m.group(2)
    b = float(m.group(3))
    c = float(m.group(4))

    if op == "+":
        v = a + b
    elif op == "-":
        v = a - b
    elif op == "*":
        v = a * b
    else:
        if b == 0:
            return False
        v = a / b

    return abs(v - c) < 1e-9


def check_percent_of(sentence: str) -> Optional[bool]:
    """
    Check patterns like:
      8% of 2500 = 200
    """
    s = sentence.replace(",", "").lower()
    m = re.search(r'(-?\d+(?:\.\d+)?)\s*%\s*of\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)', s)
    if not m:
        return None

    pct = float(m.group(1))
    base = float(m.group(2))
    claimed = float(m.group(3))
    actual = (pct / 100.0) * base
    return abs(actual - claimed) < 1e-9


def has_formal_logic_structure(sentence: str) -> bool:
    s = sentence.lower()
    marker_hits = sum(1 for m in FORMAL_LOGIC_MARKERS if m in s)

    if marker_hits >= 2:
        return True

    if re.search(r"\bif\b.+\bthen\b", s):
        return True

    return False


def has_interpretive_language(sentence: str) -> bool:
    s = sentence.lower()
    return any(w in s for w in INTERPRETIVE_WORDS)


def classify_segment(sentence: str) -> Segment:
    reasons: List[str] = []
    math_check: Optional[str] = None

    eq1 = check_binary_equation(sentence)
    eq2 = check_percent_of(sentence)

    has_math_verified = (eq1 is not None) or (eq2 is not None)
    if eq1 is True or eq2 is True:
        math_check = "true"
        reasons.append("explicit arithmetic equation verified")
    elif eq1 is False or eq2 is False:
        math_check = "false"
        reasons.append("explicit arithmetic equation failed verification")

    logic_ok = has_formal_logic_structure(sentence)
    interpretive = has_interpretive_language(sentence)

    if has_math_verified:
        return Segment(
            text=sentence,
            label="deterministic",
            reasons=reasons if reasons else ["explicit mathematical structure"],
            math_check=math_check
        )

    if logic_ok and not interpretive:
        reasons.append("explicit formal logic structure")
        return Segment(
            text=sentence,
            label="deterministic",
            reasons=reasons,
            math_check=math_check
        )

    if interpretive:
        reasons.append("hedging/advice/interpretive language")
        return Segment(
            text=sentence,
            label="interpretive",
            reasons=reasons,
            math_check=math_check
        )

    return Segment(
        text=sentence,
        label="unknown",
        reasons=["no explicit verifiable structure detected"],
        math_check=math_check
    )


def classify_response(response: str) -> Tuple[List[Segment], List[Segment], List[Segment]]:
    segments = split_sentences(response)
    deterministic: List[Segment] = []
    interpretive: List[Segment] = []
    unknown: List[Segment] = []

    for s in segments:
        seg = classify_segment(s)
        if seg.label == "deterministic":
            deterministic.append(seg)
        elif seg.label == "interpretive":
            interpretive.append(seg)
        else:
            unknown.append(seg)

    return deterministic, interpretive, unknown


def call_ambient(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    retries: int = 2,
) -> str:
    last_error: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Be concise. Separate reasoning steps clearly. "
                            "When evidence is insufficient, say so explicitly."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            return content.strip()

        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(1.2 * (attempt + 1))
            else:
                break

    raise RuntimeError(f"Ambient API call failed after retries: {last_error}") from last_error


def to_prompt_result(prompt: str, response: str) -> PromptResult:
    d, i, u = classify_response(response)
    return PromptResult(
        prompt=prompt,
        response=response,
        deterministic=d,
        interpretive=i,
        unknown=u,
        counts={
            "deterministic": len(d),
            "interpretive": len(i),
            "unknown": len(u),
            "total_segments": len(d) + len(i) + len(u),
        },
    )


def save_json_report(path: str, model: str, results: List[PromptResult]) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "runs": [
            {
                "prompt": r.prompt,
                "response": r.response,
                "verifiable_layer": [asdict(x) for x in r.deterministic],
                "non_verifiable_layer": [asdict(x) for x in (r.interpretive + r.unknown)],
                "counts": r.counts,
            }
            for r in results
        ],
        "aggregate": aggregate_counts(results),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def aggregate_counts(results: List[PromptResult]) -> Dict[str, int]:
    agg = {"deterministic": 0, "interpretive": 0, "unknown": 0, "total_segments": 0}
    for r in results:
        for k in agg:
            agg[k] += r.counts[k]
    return agg


def save_markdown_summary(path: str, model: str, results: List[PromptResult]) -> None:
    agg = aggregate_counts(results)
    lines: List[str] = []
    lines.append("# Week 5 Web2 Boundary Split Report")
    lines.append("")
    lines.append(f"- Model: `{model}`")
    lines.append(f"- Generated (UTC): `{datetime.now(timezone.utc).isoformat()}`")
    lines.append("")
    lines.append("## Aggregate Counts")
    lines.append("")
    lines.append("| deterministic | interpretive | unknown | total |")
    lines.append("|---:|---:|---:|---:|")
    lines.append(f"| {agg['deterministic']} | {agg['interpretive']} | {agg['unknown']} | {agg['total_segments']} |")
    lines.append("")

    for idx, r in enumerate(results, start=1):
        lines.append(f"## Prompt {idx}")
        lines.append("")
        lines.append("**Prompt**")
        lines.append("")
        lines.append(f"> {r.prompt}")
        lines.append("")
        lines.append("**Counts**")
        lines.append("")
        lines.append(f"- deterministic: {r.counts['deterministic']}")
        lines.append(f"- interpretive: {r.counts['interpretive']}")
        lines.append(f"- unknown: {r.counts['unknown']}")
        lines.append(f"- total_segments: {r.counts['total_segments']}")
        lines.append("")
        lines.append("**Top deterministic segments (up to 3)**")
        lines.append("")
        if r.deterministic:
            for s in r.deterministic[:3]:
                lines.append(f"- {s.text}")
        else:
            lines.append("- (none)")
        lines.append("")
        lines.append("**Top non-verifiable segments (up to 3)**")
        lines.append("")
        nonver = r.interpretive + r.unknown
        if nonver:
            for s in nonver[:3]:
                lines.append(f"- [{s.label}] {s.text}")
        else:
            lines.append("- (none)")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


DEFAULT_PROMPTS = [
    "In a ledger: treasury is 2500, tax is 8%, then split equally among 3 guilds. "
    "A report claims each gets 850 and therefore Guild A is corrupt. Evaluate.",
    "All Moon Mages wear silver rings. Doran wears a silver ring. Therefore Doran is a Moon Mage. "
    "Is this logically valid?",
    "Summarize why city reform succeeded based only on: citizens seemed happier, markets felt calmer, "
    "and complaints were fewer.",
    "If revenue consistently increases shortly after marketing spend increases, does that prove causation?"
]


def load_prompts(args: argparse.Namespace) -> List[str]:
    prompts: List[str] = []

    if args.prompt:
        prompts.extend(args.prompt)

    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)

    if not prompts:
        prompts = DEFAULT_PROMPTS

    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ambient Week 5 boundary splitter")
    parser.add_argument("--model", default="zai-org/GLM-4.6", help="Ambient model name")
    parser.add_argument("--base-url", default="https://api.ambient.xyz/v1", help="Ambient OpenAI-compatible base URL")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--prompt", action="append", help="Single prompt (can be repeated)")
    parser.add_argument("--prompts-file", help="Path to file with one prompt per line")
    parser.add_argument("--out-json", default="week5_web2_report.json")
    parser.add_argument("--out-md", default="week5_web2_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.getenv("AMBIENT_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "API key not found. Set AMBIENT_API_KEY (or OPENAI_API_KEY) in environment."
        )

    prompts = load_prompts(args)
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    results: List[PromptResult] = []
    for idx, p in enumerate(prompts, start=1):
        print(f"[{idx}/{len(prompts)}] Calling model...")
        response = call_ambient(
            client=client,
            model=args.model,
            prompt=p,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retries=2,
        )
        result = to_prompt_result(prompt=p, response=response)
        results.append(result)

        c = result.counts
        print(
            f"  -> segments: total={c['total_segments']} | "
            f"det={c['deterministic']} | int={c['interpretive']} | unk={c['unknown']}"
        )

    save_json_report(args.out_json, args.model, results)
    save_markdown_summary(args.out_md, args.model, results)

    print("\nDone.")
    print(f"- JSON report: {args.out_json}")
    print(f"- Markdown summary: {args.out_md}")


if __name__ == "__main__":
    main()

