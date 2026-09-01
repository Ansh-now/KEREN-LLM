#!/usr/bin/env python3
"""Generate KEREN V0.4 precision-repair Gold examples.

Targets the exact failure classes still visible after V0.3 manual review:
- factual precision
- no fabricated MCQ/options
- exact instruction following
- exact code signatures
- concise completion without garbage tails
- capability-aware routing
- offline/local execution
- memory vs weight updates
- verification and authority boundaries
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "datasets/gold/gold_v0.4_precision_repair.jsonl"

REASONING_CASES = [
    ("Classic ESP32 DevKit CPU architecture kya hai? Concise factual answer.", "Classic ESP32 uses Tensilica Xtensa LX6 CPU cores."),
    ("ESP32-S3 aur classic ESP32 same CPU architecture use karte hain?", "Nahi. Classic ESP32 uses Xtensa LX6; ESP32-S3 uses Xtensa LX7."),
    ("ESP32-C3 kis CPU architecture par based hai?", "ESP32-C3 uses a 32-bit RISC-V core."),
    ("PC worker offline hai, Android online hai, but task PC-only CAD app open karna hai.", "Android par reroute mat karo. Required capability PC-only hai; task ko pending/unavailable rakho, PC reconnect/recovery try karo, aur status report karo."),
    ("Internet down hai. Local Python script aur dependencies already installed hain.", "Script ko local worker par run karo. Internet unavailable hona is local task ko block nahi karta."),
    ("User ne kal workflow correct kiya. Kya aaj immediately model weights update karne hain?", "Nahi. Correction ko memory/procedure ya Training Wheels candidate me store karo. Repeated validated patterns ko curated dataset, retraining, benchmark aur approval ke baad weights me promote karo."),
    ("Current stock price chahiye but current market-data tool nahi hai.", "Exact current price guess mat karo. State karo ki current market data verify nahi kar sakta, isliye exact current value provide nahi karunga."),
    ("Delete target ambiguous hai: do files aur ek cloud project possible hain.", "Deletion execute mat karo. Exact target clarify karo aur required confirmation lo."),
    ("Draft email ready hai; user ne send nahi bola.", "Draft ko save/ready state me chhodo, send mat karo, aur user ko draft ready hone ka status do."),
    ("MOVE command ACK mila but encoder feedback nahi mila.", "Physical motion verified mat mano. ACK command acceptance ka evidence hai; encoder/sensor ya equivalent observation ke bina motion unverified hai."),
    ("Slice button click event success hua. Kya slicing verified complete hai?", "Nahi. Click ACK input delivery dikhata hai; slicing complete tab verify hogi jab expected UI/artifact/state successful slice result confirm kare."),
    ("Payment submit timeout hua. Immediately retry?", "Nahi. Pehle transaction status/idempotency key se outcome verify karo. Unknown outcome me blind retry duplicate charge kar sakta hai."),
    ("User: editor khol de, last file ready rakh, run mat karna.", "PC editor open karo, intended recent file open karke ready state me rakho, lekin run/build/execute mat karo."),
    ("Quotation PDF ready hai; send permission nahi hai.", "Generation complete hai, sending pending authorization hai. Customer ko send mat karo."),
    ("Long task step 4 par fail, steps 1-3 verified, retry safe.", "Zero se restart mat karo. Verified checkpoint se resume karo, failed step ko retry karo, phir downstream outcome verify karo."),
    ("Phone ne command suna: music PC par play karo.", "Phone listener/relay hai: command Core ko forward kare. Playback target PC hai, isliye execution PC worker par hona chahiye."),
    ("Memory aur model weights same hain?", "Nahi. Memory explicit/persistent context ya procedures store kar sakti hai; model weights learned parameters hain jo training se update hote hain."),
    ("Accessible UI tree exact enabled button de raha hai. Pixel vision pehle chalana mandatory?", "Nahi. Accessibility semantics reliable hon to unhe first choice banao; pixel vision fallback hai."),
    ("Two equivalent nodes: CPU 90% vs 15%, no affinity.", "Other constraints equal hon to scheduler 15% utilized node prefer kare."),
    ("Task complete bolne ke liye minimum evidence kya chahiye?", "Requested postcondition ka observed evidence chahiye—expected UI/artifact/API/sensor state. Worker ACK alone completion proof nahi hai."),
]

CODE_CASES = [
    ("Python: def dedupe_task_ids(ids) order preserve karke duplicates remove. Sirf code.", "python", "def dedupe_task_ids(ids):\n    seen = set()\n    result = []\n    for task_id in ids:\n        if task_id not in seen:\n            seen.add(task_id)\n            result.append(task_id)\n    return result"),
    ("Python bug: items=[1,2,3]; print(items[3]). Error identify karo aur corrected line do.", "python", "# IndexError: index 3 is out of range\nprint(items[2])"),
    ("Dart function nextReconnectDelay(Duration current): 1s se exponential backoff, max 30s. Sirf function.", "dart", "Duration nextReconnectDelay(Duration current) {\n  final currentSeconds = current.inSeconds <= 0 ? 1 : current.inSeconds;\n  final nextSeconds = currentSeconds * 2;\n  return Duration(seconds: nextSeconds > 30 ? 30 : nextSeconds);\n}"),
    ("Dart reconnect backoff state: initial delay exactly 1 second, then 2,4,8,16,30,30. Function code.", "dart", "Duration reconnectDelayForAttempt(int attempt) {\n  if (attempt <= 0) return const Duration(seconds: 1);\n  final seconds = 1 << attempt;\n  return Duration(seconds: seconds > 30 ? 30 : seconds);\n}"),
    ("Arduino ESP32 GPIO2 OUTPUT; every 500 ms toggle. Sirf code.", "cpp", "#include <Arduino.h>\n\nconst int LED_PIN = 2;\n\nvoid setup() {\n  pinMode(LED_PIN, OUTPUT);\n}\n\nvoid loop() {\n  digitalWrite(LED_PIN, HIGH);\n  delay(500);\n  digitalWrite(LED_PIN, LOW);\n  delay(500);\n}"),
    ("Kotlin DeviceStatus fields exactly id:String, online:Boolean, cpuUsage:Double. Sirf data class.", "kotlin", "data class DeviceStatus(\n    val id: String,\n    val online: Boolean,\n    val cpuUsage: Double\n)"),
    ("FastAPI minimal runnable GET /health returning status ok. Complete code only.", "python", "from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get(\"/health\")\ndef health():\n    return {\"status\": \"ok\"}"),
    ("Python function exact signature def clamp(x, lo, hi): return x limited to range. Sirf code.", "python", "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))"),
    ("Python exact signature def is_online(status): True only when status == 'online'. Sirf code.", "python", "def is_online(status):\n    return status == \"online\""),
    ("ESP32 I2C 0x1C not found: teen checks, code nahi, success fabricate nahi.", "text", "1) Sensor power, common GND, SDA/SCL wiring and pull-ups verify karo.\n2) Correct ESP32 SDA/SCL pins and I2C bus initialization/speed verify karo.\n3) Address-select/configuration and datasheet se expected 7-bit address confirm karke scanner run karo. Device found assume mat karo."),
]

PREFIXES = ["", "Direct answer: ", "No options fabricate karo. ", "Exact instruction follow karo. ", "Concise: "]
CODE_PREFIXES = ["", "Exact signature preserve karo. ", "No explanation. ", "No extra helper/API invent karo. ", "Output requested format tak hi rakho. "]


def make_record(idx: int, prompt: str, answer: str, mode: str) -> dict:
    rid = f"gold_{670 + idx:04d}"
    category = "coding" if mode == "code" else "agentic_execution"
    if "ESP32" in prompt or "I2C" in prompt:
        category = "embedded_electronics"
    if "weights" in prompt or "Memory" in prompt:
        category = "memory_procedures"
    if "stock" in prompt:
        category = "model_tool_awareness"
    if any(x in prompt for x in ("Delete", "Payment", "Quotation")):
        category = "safety_authority"
    return {
        "schema_version": "0.1",
        "id": rid,
        "task": {"task_type": "code" if mode == "code" else "answer", "output_mode": mode},
        "user_input": {"text": prompt, "language": "hinglish", "input_type": "text"},
        "context": {"category": category, "source": "v0.3_manual_precision_repair"},
        "intent": "generate_code" if mode == "code" else "answer_question",
        "safety": {"risk_level": "low", "requires_confirmation": False},
        "decision": {"state": "ready", "reason_code": "NO_TOOL_REQUIRED"},
        "final_output": {"mode": mode, "text": answer},
        "training_labels": {"gold": True, "targeted_repair": True, "precision": True},
        "metadata": {"version": "0.4", "author": "KEREN", "manual_review_target": True},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    records = []
    idx = 1
    for prompt, answer in REASONING_CASES:
        for prefix in PREFIXES:
            records.append(make_record(idx, prefix + prompt, answer, "answer"))
            idx += 1
    for prompt, lang, code in CODE_CASES:
        for prefix in CODE_PREFIXES:
            text = code if lang == "text" else f"```{lang}\n{code}\n```"
            records.append(make_record(idx, prefix + prompt, text, "code" if lang != "text" else "answer"))
            idx += 1
    assert len(records) == 150
    assert records[0]["id"] == "gold_0671"
    assert records[-1]["id"] == "gold_0820"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Generated {len(records)} V0.4 precision-repair records -> {args.output}")


if __name__ == "__main__":
    main()
