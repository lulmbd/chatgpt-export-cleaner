#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, argparse, re, unicodedata
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def clean_text(s: str) -> str:
    """Clean and normalize raw text content."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\t•", "•").replace("\t", "    ")
    s = s.replace("•\t", "• ").replace("•  ", "• ")
    s = s.replace("\u00A0", " ")
    s = re.sub(r'""+$', '"', s.strip())
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def extract_messages_from_mapping(conv: dict) -> list:
    """
    Reconstruct message order using mapping + current_node.
    Only keep user and assistant messages (ignore system/tool).
    Flatten parts into readable text.
    """
    mapping = conv.get("mapping", {})
    current = conv.get("current_node")
    ordered_nodes = []
    seen = set()

    while current and current in mapping and current not in seen:
        seen.add(current)
        node = mapping[current]
        ordered_nodes.append(node)
        current = node.get("parent")

    ordered_nodes.reverse()
    msgs = []

    for node in ordered_nodes:
        m = node.get("message")
        if not m:
            continue
        author = (m.get("author") or {}).get("role", "")
        if author in ("tool", "ChatGPT"):
            author = "assistant"
        if author == "system" and not (m.get("metadata") or {}).get("is_user_system_message"):
            continue

        content = m.get("content") or {}
        ctype = content.get("content_type")
        if ctype not in ("text", "multimodal_text"):
            continue

        parts = content.get("parts") or []
        chunks = []

        for p in parts:
            if isinstance(p, str) and p.strip():
                chunks.append(p)
            elif isinstance(p, dict):
                if p.get("text"):
                    chunks.append(p["text"])
                elif p.get("content_type") == "audio_transcription" and p.get("text"):
                    chunks.append(p["text"])

        text = clean_text("\n".join(chunks))
        if not text:
            continue

        role = "assistant" if author == "assistant" else "user"
        msgs.append({"role": role, "text": text})
    return msgs

def messages_to_pairs(messages: list) -> list:
    """
    Group consecutive user messages as a single prompt,
    paired with the next assistant response as completion.
    """
    pairs = []
    buffer_user = []

    for m in messages:
        if m["role"] == "user":
            if m["text"]:
                buffer_user.append(m["text"])
        else:
            if buffer_user and m["text"]:
                prompt = clean_text("\n\n".join(buffer_user))
                completion = clean_text(m["text"])
                if prompt and completion:
                    pairs.append({"prompt": prompt, "completion": completion})
            buffer_user = []
    return pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True, help="Path to OpenAI's conversations.json file")
    parser.add_argument("--out", dest="outdir", required=True, help="Output folder for cleaned exports")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    md_dir = outdir / "markdown_by_conversation"
    md_dir.mkdir(exist_ok=True)

    data = json.loads(Path(args.inp).read_text(encoding="utf-8", errors="ignore"))

    # Handle OpenAI export format: {"conversations": [...]} or just a list
    conversations = data["conversations"] if isinstance(data, dict) and "conversations" in data else data

    all_convos = []
    all_pairs = []

    for conv in tqdm(conversations, desc="Parsing conversations"):
        title = conv.get("title") or "Conversation"
        messages = extract_messages_from_mapping(conv)
        if not messages:
            continue

        # Save Markdown version of each conversation
        md_lines = [f"# {title}\n"]
        for m in messages:
            who = "👤 You" if m["role"] == "user" else "🤖 Assistant"
            md_lines.append(f"**{who}**:\n\n{m['text']}\n")
        safe_title = re.sub(r'[^\w\-.]+', '_', title)[:120] or 'conversation'
        (md_dir / f"{safe_title}.md").write_text("\n".join(md_lines), encoding="utf-8")

        all_convos.append({"title": title, "messages": messages})

        # Optional: generate prompt-completion pairs
        pairs = messages_to_pairs(messages)
        for p in pairs:
            p["_title"] = title
        all_pairs.extend(pairs)

    # Output all structured data
    (outdir / "all_conversations.json").write_text(
        json.dumps(all_convos, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (outdir / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n✅ Export completed!\n→ {outdir/'all_conversations.json'}\n→ {outdir/'pairs.jsonl'}\n→ {md_dir}/\n")

if __name__ == "__main__":
    main()