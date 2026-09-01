#!/usr/bin/env python3
"""Generate KEREN V0.3 targeted semantic-repair Gold examples.

These examples target failure classes observed during manual review of the
V0.2 frozen benchmark: factual embedded knowledge, exact instruction following,
complete code, verification, authority boundaries, routing, memory-vs-weights,
current-data uncertainty, idempotency, and hallucination control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "datasets/gold/gold_v0.3_semantic_repair.jsonl"

CASES = [
    ("Classic ESP32 CPU architecture?", "Classic ESP32 uses Tensilica Xtensa LX6 cores; it is not an ARM Cortex-M4."),
    ("ESP32-S3 CPU family?", "ESP32-S3 uses dual-core Xtensa LX7 CPUs."),
    ("ESP32-C3 CPU architecture?", "ESP32-C3 uses a 32-bit RISC-V core."),
    ("Model sliced hai; user ne sirf prepare bola, print permission nahi di.", "Preparation complete rakho, but Start Print mat karo. Printing consequential action hai aur explicit approval/preauthorization chahiye."),
    ("PC-specific CAD app open karna hai but PC worker offline; Android online hai.", "Android par reroute mat karo because required app/capability PC-specific hai. Task ko pending/unavailable mark karke PC recovery/reconnect ka wait ya user ko status do."),
    ("Local Python script run karni hai, dependencies installed hain, internet unavailable hai.", "Local worker par script run karo. Internet ki zarurat nahi hai."),
    ("Kal user ne workflow correct kiya. Kya ek correction par model weights immediately update karne hain?", "Nahi. Correction ko memory/procedure ya Training Wheels candidate ke roop me record karo; repeated validated patterns ko curated dataset, benchmark aur approval ke baad future adapter training me promote karo."),
    ("Exact current stock price chahiye but market-data/internet tool unavailable hai.", "Current price guess mat karo. Bolo ki current market data verify karne ka tool available nahi hai, isliye exact current value provide nahi kar sakta."),
    ("MOVE ACK mila but encoder/sensor feedback nahi mila.", "ACK sirf command acceptance ka evidence hai, physical motion ka proof nahi. Motion ko unverified rakho jab tak encoder/sensor ya equivalent observation expected movement confirm na kare."),
    ("Payment submit timeout hua. Immediately retry?", "Nahi. Pehle provider/transaction status ya idempotency key se check karo ki payment process hui ya nahi. Unknown outcome me blind retry duplicate charge kar sakta hai."),
    ("Slice click successful return hua. Complete?", "Nahi. Click success input delivery batata hai; slicing complete tab verify hogi jab UI/artifact/state successful slice result dikhaye."),
    ("Delete request me do files aur ek cloud project possible targets hain.", "Delete mat karo. Target ambiguous hai aur deletion consequential hai; exact target clarify karo aur required confirmation lo."),
    ("Email draft ready hai; user ne send nahi bola.", "Draft ko ready state me chhodo aur send mat karo. User ko draft ready hone ka status do."),
    ("Dangerous machine temperature cross hui; emergency stop policy preauthorized hai.", "Emergency motor stop immediately execute karo, outcome verify/log karo, aur owner ko notify karo; fresh confirmation ka wait mat karo."),
    ("Accessible UI tree me exact enabled button mil gaya. Pixel vision mandatory?", "Nahi. Reliable accessibility semantics available hain to unhe prefer karo; pixel vision fallback hai jab semantic state insufficient ho."),
    ("Two same-capability nodes: CPU 90% and 15%, no affinity.", "Other constraints equal hon to scheduler 15% utilized node prefer kare because it has more available capacity."),
    ("Relay ESP32 ne command Core ko forward ki. Heavy AI locally mandatory?", "Nahi. Relay node sensing/forwarding/control kar sakta hai; heavy reasoning Core/SBC/appropriate compute node par ho sakti hai."),
    ("Task complete claim se pehle evidence?", "Observed postcondition chahiye jo requested outcome ko verify kare—jaise expected UI state, artifact, API state, sensor feedback, ya successful test. Worker ACK alone enough nahi hai."),
    ("I2C sensor expected 0x1C but scanner me nahi milta. 3 checks.", "1) SDA/SCL wiring, common GND aur supply voltage verify karo. 2) Correct ESP32 I2C pins/bus initialization aur pull-ups check karo. 3) Sensor address-select pin/datasheet se actual address confirm karke scanner se bus scan karo. Device found fabricate mat karo."),
    ("User: editor khol de, last file ready rakh, run mat karna.", "PC editor open karo, most recent intended file resolve/open karke ready state me rakho, but run/build/execute mat karo."),
]

CODE_CASES = [
    ("Python: def dedupe_task_ids(ids), order preserve karke duplicates remove. Sirf code.", "python", "def dedupe_task_ids(ids):\n    seen = set()\n    result = []\n    for task_id in ids:\n        if task_id not in seen:\n            seen.add(task_id)\n            result.append(task_id)\n    return result"),
    ("FastAPI minimal runnable GET /health returning {status: ok}. Sirf code.", "python", "from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get(\"/health\")\ndef health():\n    return {\"status\": \"ok\"}"),
    ("Arduino ESP32 GPIO2 OUTPUT and toggle every 500 ms. Sirf code.", "cpp", "#include <Arduino.h>\n\nconst int LED_PIN = 2;\n\nvoid setup() {\n  pinMode(LED_PIN, OUTPUT);\n}\n\nvoid loop() {\n  digitalWrite(LED_PIN, !digitalRead(LED_PIN));\n  delay(500);\n}"),
    ("Dart reconnect delay exponential backoff from 1 second, cap 30 seconds. Function code.", "dart", "Duration nextReconnectDelay(Duration current) {\n  final nextSeconds = current.inSeconds <= 0 ? 1 : current.inSeconds * 2;\n  return Duration(seconds: nextSeconds > 30 ? 30 : nextSeconds);\n}"),
    ("Kotlin DeviceStatus id:String online:Boolean cpuUsage:Double. Sirf data class.", "kotlin", "data class DeviceStatus(\n    val id: String,\n    val online: Boolean,\n    val cpuUsage: Double\n)"),
    ("Python items=[1,2,3]; print(items[3]) bug identify + corrected line.", "python", "# IndexError: valid indices are 0, 1, 2\nitems = [1, 2, 3]\nprint(items[2])"),
]


def make_record(i: int, prompt: str, answer: str, mode: str = "answer") -> dict:
    rid = f"gold_{550+i:04d}"
    intent = "generate_code" if mode == "code" else "answer_question"
    category = "coding" if mode == "code" else "agentic_execution"
    if "ESP32" in prompt or "I2C" in prompt:
        category = "embedded_electronics"
    if "weights" in prompt:
        category = "memory_procedures"
    if "stock" in prompt:
        category = "model_tool_awareness"
    if any(x in prompt for x in ("Payment", "Delete", "print permission", "temperature")):
        category = "safety_authority"
    return {
        "schema_version": "0.1",
        "id": rid,
        "task": {"task_type": "code" if mode == "code" else "answer", "output_mode": mode},
        "user_input": {"text": prompt, "language": "hinglish", "input_type": "text"},
        "context": {"category": category, "source": "v0.2_manual_benchmark_repair"},
        "intent": intent,
        "safety": {"risk_level": "low", "requires_confirmation": False},
        "decision": {"state": "ready", "reason_code": "NO_TOOL_REQUIRED"},
        "final_output": {"mode": mode, "text": answer},
        "training_labels": {"gold": True, "targeted_repair": True},
        "metadata": {"version": "0.3", "author": "KEREN", "review_required": True},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    records = []
    idx = 1
    # Four paraphrase families per reasoning case: 80 examples.
    prefixes = ["", "Concise: ", "KEREN decision: ", "Direct answer do: "]
    for prompt, answer in CASES:
        for prefix in prefixes:
            records.append(make_record(idx, prefix + prompt, answer))
            idx += 1
    # Five instruction variants per code case: 30 examples.
    code_prefixes = ["", "Exact instruction follow karo: ", "No explanation: ", "Complete answer: ", "Do not invent APIs: "]
    for prompt, lang, code in CODE_CASES:
        for prefix in code_prefixes:
            records.append(make_record(idx, prefix + prompt, f"```{lang}\n{code}\n```", "code"))
            idx += 1
    # 10 anti-hallucination / directness contrasts.
    extras = [
        ("Tool output missing hai. Kya successful result invent kar sakte ho?", "Nahi. Missing observation ko unknown/unverified bolo; success fabricate mat karo."),
        ("Prompt ne options nahi diye. Kya A/B/C/D choices fabricate karne hain?", "Nahi. User ke requested format me direct answer do; nonexistent options invent mat karo."),
        ("User ne sirf code bola. Explanation bhi add karni hai?", "Nahi. Sirf requested code output karo."),
        ("Unknown hardware revision ka exact pinout yaad nahi. Guess?", "Nahi. Exact revision/datasheet ya reliable source verify karo; pinout fabricate mat karo."),
        ("API endpoint unknown hai. Plausible URL invent karke code do?", "Nahi. Unknown API contract/URL fabricate mat karo; required API details maango ya clearly placeholder label karo."),
        ("Worker says command accepted. Is that final success?", "Nahi. Acceptance aur requested outcome alag states hain; postcondition verify karo."),
        ("User asks current information but no current-data tool. Old memory ko current bol sakte ho?", "Nahi. Freshness limitation state karo aur exact current claim mat karo."),
        ("Retry may duplicate an external side effect. Safe default?", "Outcome/status verify karo aur idempotency mechanism use karo before retry."),
        ("Target node lacks required capability. Any online node par bhej do?", "Nahi. Capability-compatible node hi select karo; otherwise unavailable/pending/escalate."),
        ("Physical action ke sensor evidence absent hain. Confident language use kare?", "Nahi. State ko unverified/unknown rakho jab tak adequate evidence na mile."),
    ]
    for prompt, answer in extras:
        records.append(make_record(idx, prompt, answer))
        idx += 1

    assert len(records) == 120
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Generated {len(records)} V0.3 semantic-repair records -> {args.output}")


if __name__ == "__main__":
    main()
