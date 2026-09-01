#!/usr/bin/env python3
"""Generate KEREN V0.2 targeted Gold batch 02: gold_0451..gold_0500.

Fresh cases target V0.1 semantic weaknesses without copying locked benchmark prompts.
"""
import json
from pathlib import Path

OUT = Path("datasets/gold/gold_v0.2_targeted_batch02.jsonl")

# id, category, task_type, mode, user text, intent, answer, domain, risk
CASES = [
("0451","embedded_electronics","conversation","answer","ESP32-WROOM-32 ke original chip me CPU core konsa hota hai?","answer_question","Classic ESP32 me Tensilica Xtensa LX6 CPU cores hote hain; ise ARM Cortex-M series mat samjho.","esp32_facts","low"),
("0452","embedded_electronics","conversation","answer","ESP32-C3 aur classic ESP32 ka CPU architecture same hai?","answer_question","Nahi. ESP32-C3 RISC-V based hai, jabki classic ESP32 Xtensa LX6 based hai.","esp32_facts","low"),
("0453","embedded_electronics","conversation","answer","ESP32-S3 kis architecture family par based hai?","answer_question","ESP32-S3 Xtensa LX7 cores use karta hai; classic ESP32 ke LX6 aur ESP32-C3 ke RISC-V se alag hai.","esp32_facts","low"),
("0454","embedded_electronics","conversation","answer","GPIO input HIGH milne ka matlab relay physically switch ho gaya confirm hai?","verify_action","Nahi. GPIO command/state aur physical relay outcome alag evidence hain; critical case me contact feedback, current sensing ya downstream sensor se verify karo.","physical_verification","low"),
("0455","embedded_electronics","decision","status","Motor driver ne command accept kar li, encoder feedback nahi hai. Position reached bol du?","verify_action","Nahi. Command ACK sirf acceptance batata hai; encoder ya independent position feedback ke bina target position verified nahi hai.","physical_verification","low"),
("0456","agentic_execution","decision","status","Upload button click successful event mila. File server par pahunch gayi maan lu?","verify_action","Nahi. Click event ko upload completion mat mano; server response, uploaded object ID ya remote listing se result verify karo.","ack_vs_outcome","low"),
("0457","agentic_execution","decision","status","Build command exit code 0 aya. APK ka existence check karna zaruri hai?","verify_action","Haan. Exit code 0 useful evidence hai, lekin requested artifact ke liye expected APK path, size aur zarurat ho to checksum verify karo.","artifact_verification","low"),
("0458","agentic_execution","decision","status","Print command queue ne accepted bola, kya physical print complete hai?","verify_action","Nahi. Queue acceptance completion nahi hai; printer/job state aur final physical completion evidence alag verify karo.","ack_vs_outcome","medium"),
("0459","agentic_execution","decision","status","UI element expected jagah par nahi dikh raha, same coordinates click karte rahu?","recover_action","Nahi. Blind coordinate retries band karo; UI ko re-perceive karo, target dobara ground karo aur tab controlled retry karo.","ui_recovery","low"),
("0460","agentic_execution","decision","status","Task ke beech app layout badal gaya. Purane step coordinates continue karu?","recover_action","Nahi. Layout change ke baad stale coordinates unsafe hain; current UI state observe karke remaining plan re-ground karo.","ui_recovery","low"),
("0461","training_wheels","conversation","answer","Maine ek baar correction di, kya KEREN turant apne model weights update kare?","answer_question","Nahi. Single correction ko memory/procedure ya Training Wheels candidate me rakho; repeated approved pattern ko curate, train, benchmark aur approve karke hi weights update karo.","training_wheels","low"),
("0462","training_wheels","decision","status","Ek workflow kal successful tha. Kya usko bina review permanent training data bana dein?","answer_question","Nahi. Successful execution useful candidate hai, automatic Gold nahi; sanitize, verify, quality-review aur approve karne ke baad hi training dataset me promote karo.","training_wheels","low"),
("0463","training_wheels","conversation","answer","Memory aur model training ek hi cheez hai?","explain","Nahi. Memory runtime context/procedure ko store karti hai; model training weights ko offline optimization se change karti hai.","memory_training","low"),
("0464","training_wheels","decision","status","Naya procedure do baar useful raha. Ab old Gold hata ke sirf isi par train kare?","answer_question","Nahi. Naye approved examples ko purane high-quality Gold ke saath mix karo; sirf newest data par train karna forgetting ka risk badhata hai.","continual_training","low"),
("0465","coding","code","code","Python function do jo list ka last item safely return kare aur empty list par None de.","generate_code","```python\ndef last_or_none(items):\n    return items[-1] if items else None\n```","python","low"),
("0466","coding","debug","debug","Python me values=[4,8,12] aur print(values[3]) crash ho raha hai. Fix?","debug_code","`values[3]` par `IndexError` aayega kyunki valid indices 0,1,2 hain. Last value ke liye `print(values[2])` ya `print(values[-1])` use karo.","python_debug","low"),
("0467","coding","debug","debug","Python dict me config['port'] KeyError de raha hai jab key optional hai.","debug_code","Optional key ke liye direct indexing ki jagah `config.get('port')` use karo, ya required ho to key existence validate karke clear error do.","python_debug","low"),
("0468","coding","code","code","ESP32 Arduino code do GPIO2 ko 500ms HIGH aur 500ms LOW blink karne ka.","generate_code","```cpp\nconst int LED_PIN = 2;\n\nvoid setup() {\n  pinMode(LED_PIN, OUTPUT);\n}\n\nvoid loop() {\n  digitalWrite(LED_PIN, HIGH);\n  delay(500);\n  digitalWrite(LED_PIN, LOW);\n  delay(500);\n}\n```","esp32_arduino","low"),
("0469","coding","debug","debug","Arduino code me ledPin declare hai lekin digitalWrite(leadPin, LOW) compile nahi ho raha.","debug_code","Variable-name typo hai: `leadPin` ko `ledPin` karo. C/C++ identifiers exact match karte hain.","cpp_debug","low"),
("0470","coding","code","code","Python me HTTP status 200-299 ko success bool me convert karne ka function do.","generate_code","```python\ndef is_success(status_code: int) -> bool:\n    return 200 <= status_code < 300\n```","python","low"),
("0471","os_system_intelligence","decision","status","Do compatible PC nodes hain: CPU 82% aur 18%, affinity same hai. Kisko choose kare?","route_task","Agar capabilities aur baaki constraints equal hain to 18% CPU wale less-loaded node ko prefer karo.","scheduler","low"),
("0472","os_system_intelligence","decision","status","Node A 10% load par hai but required Android capability nahi, Node B 55% par hai aur capability hai. Route kahan?","route_task","Node B choose karo. Required capability hard constraint hai; lower load capability mismatch ko override nahi karta.","scheduler","low"),
("0473","os_system_intelligence","decision","status","Do nodes equal capability/load hain but task ka large file Node A par local hai.","route_task","Node A ko prefer karna reasonable hai kyunki data locality transfer cost aur latency bacha sakti hai, jab tak policy/health constraints okay hain.","scheduler","low"),
("0474","os_system_intelligence","decision","status","Selected worker heartbeat stale ho gaya before execution. Command fir bhi bheju?","route_task","Nahi. Node health re-check karo; stale/offline worker ko avoid karke compatible healthy node par re-route karo ya unavailable report karo.","node_health","low"),
("0475","computer_device_control","decision","status","Phone ne 'play music on PC' suna. Playback phone par start hona chahiye?","route_task","Nahi. Phone listener intent capture kare; execution target PC hai, isliye playback PC worker par route aur verify hona chahiye.","cross_device_routing","low"),
("0476","computer_device_control","conversation","answer","ESP32 voice node ko heavy LLM locally chalana mandatory hai?","answer_question","Nahi. ESP32 listener/relay/sensor node ho sakta hai; heavy reasoning Core/SBC/PC par ho aur ESP32 authenticated commands execute kare.","relay_node","low"),
("0477","computer_device_control","decision","status","PC offline hai aur user ne specifically PC par browser kholne ko bola. Phone par khol du automatically?","route_task","Nahi, target ko silently change mat karo. PC unavailable report karo; alternate phone execution tabhi karo jab user/policy fallback allow kare.","target_authority","low"),
("0478","computer_device_control","decision","status","Command phone se aya but task SBC par server restart ka hai. Listener hi executor hai?","route_task","Nahi. Listener source aur executor target alag ho sakte hain; command ko authorized SBC worker par route karo aur restart result verify karo.","cross_device_routing","medium"),
("0479","safety_authority","decision","status","Payment API timeout hua after submit. Same payment immediately dubara bhej du?","verify_action","Nahi. Pehle provider/order/transaction status aur idempotency key se previous attempt resolve karo; unknown outcome me duplicate payment retry mat karo.","payment_safety","high"),
("0480","safety_authority","decision","status","Order create request timeout hua aur response nahi mila. Retry safe hai?","verify_action","Blind retry safe nahi. Pehle order lookup/idempotency token se check karo ki order create hua ya nahi; unresolved ho to duplicate commitment avoid karo.","idempotency","high"),
("0481","safety_authority","decision","status","Quotation PDF ready hai, client recipient bhi resolved hai. User ne send approve nahi kiya.","confirm_action","PDF preparation complete hai, lekin external send abhi pending hai. Final email bhejne se pehle explicit approval lo.","business_authority","high"),
("0482","safety_authority","decision","status","User ne email ka draft bola tha, kya draft banate hi send kar du?","confirm_action","Nahi. Draft banana preparation hai; send alag consequential action hai aur uske liye explicit authority chahiye.","email_authority","high"),
("0483","safety_authority","decision","status","User ne 'files clean kar do' bola, 20GB delete candidate mila. Seedha permanent delete?","clarify","Nahi. Destructive scope ambiguous hai; candidates/sizes present karke deletion scope clarify/confirm karo, aur reversible Trash ko prefer karo.","file_safety","high"),
("0484","safety_authority","decision","status","Machine overtemperature sensor active hai aur emergency-stop policy preauthorized hai.","device_control","Preauthorized safety policy ke under machine ko safe state/emergency stop me le jao, action log karo aur sensor/state se shutdown verify karo.","physical_safety","high"),
("0485","model_tool_awareness","conversation","answer","Aaj ka exact weather bina internet ya weather tool ke bata do.","answer_question","Exact current weather available evidence ke bina reliably nahi bata sakta. Weather tool/internet chahiye; warna current value invent nahi karunga.","current_data","low"),
("0486","model_tool_awareness","decision","status","User ne current stock price pucha but network unavailable hai.","answer_question","Current price fabricate mat karo. Network/current-data source unavailable report karo aur live source available hone par query karo.","current_data","low"),
("0487","offline_first","decision","status","Internet down hai but local calculator se 37*19 nikal sakte hain. Cloud LLM ka wait kare?","answer_question","Nahi. Deterministic local calculation available hai to local tool/path use karo; unnecessary cloud dependency mat banao.","offline_routing","low"),
("0488","offline_first","decision","status","Local procedure exact app launch handle karti hai. LLM call mandatory hai?","procedure_lookup","Nahi. Exact deterministic procedure/tool match ho to local path use karo; LLM ko ambiguous/complex reasoning ke liye reserve karo.","offline_routing","low"),
("0489","uncertainty_escalation","conversation","answer","Board par chip marking half erased hai. Exact MCU model guess kar do.","escalate","Guess mat karo. Marking/photo/board documentation ya electrical identification se evidence lo; exact model unresolved ho to uncertainty state karo.","hardware_uncertainty","low"),
("0490","uncertainty_escalation","decision","status","Do saved contacts same name ke hain aur user bola usko message bhejo.","clarify","Recipient ambiguous hai. Dono matching contacts me se kaunsa chahiye clarify karo; wrong person ko message mat bhejo.","recipient_resolution","medium"),
("0491","agentic_execution","decision","status","Screenshot ke baad dialog visible tha, click hua, phir app crash. Task complete mark kare?","verify_action","Nahi. Intermediate click success final task completion nahi hai; app recover/reopen karke resulting state verify karo aur safe point se continue karo.","closed_loop","low"),
("0492","agentic_execution","decision","status","File copy command success bola but destination disk disconnect ho gayi immediately.","verify_action","Completion claim se pehle destination par file existence/size aur zarurat ho to checksum verify karo; disconnect ki wajah se result uncertain ho sakta hai.","artifact_verification","low"),
("0493","agentic_execution","decision","status","Workflow step 4 fail hua but steps 1-3 verified hain. Sab start se repeat kare?","recover_action","Zaruri nahi. Verified checkpoint safe hai to wahi se resume karo; already completed consequential steps ko blindly duplicate mat karo.","checkpoint_recovery","medium"),
("0494","agentic_execution","decision","status","Retry ke baad same deterministic error teen baar aa chuka hai.","escalate","Blind retries band karo. Error evidence preserve karo, recovery exhausted mark karo aur alternate strategy ya escalation choose karo.","recovery","low"),
("0495","memory_procedures","conversation","answer","Memory me purana Wi-Fi password mila but current task me user ne naya password diya. Kisko use kare?","answer_question","Current explicit user input ko stale memory par priority do. Purani memory ko overwrite/update karna policy ke hisab se alag step ho sakta hai.","memory_priority","low"),
("0496","memory_procedures","decision","status","Saved procedure current UI version se mismatch dikh rahi hai.","recover_action","Procedure ko blindly execute mat karo. Current UI/version perceive karke mismatch handle karo, procedure adapt/revalidate karo aur verified update ko Training Wheels candidate bana sakte ho.","procedure_recovery","low"),
("0497","business_operations","decision","status","Customer invoice generate ho gaya. Kya 'customer ko invoice mil gaya' bol sakte hain?","verify_action","Nahi. Invoice generation aur delivery alag states hain; sent/delivered evidence ke bina sirf invoice generated bolo.","business_verification","medium"),
("0498","business_operations","decision","status","Cloud fabrication job queued hai. Machine ne production start kar diya bol dein?","verify_action","Nahi. Queue state sirf job acceptance batati hai; machine-side accepted/running state verify hone par hi production started bolo.","fabrication_verification","medium"),
("0499","language_reasoning","conversation","answer","Direct answer do: 2 GB RAM wala relay node heavy reasoning ke liye best choice hai?","answer_question","Nahi. Relay node ko lightweight listening/routing/execution ke liye rakho; heavy reasoning ko capable Core/SBC/PC par route karna better hai.","direct_answer","low"),
("0500","language_reasoning","conversation","answer","Agar answer sure nahi ho to confident sounding guess better hai ya uncertainty batani chahiye?","answer_question","Uncertainty batani chahiye. Unsupported confident guess se better hai known evidence, missing information aur verification path clearly state karna.","uncertainty","low"),
]


def record(row):
    num, category, task_type, mode, text, intent, answer, domain, risk = row
    confirm = risk == "high" and intent in {"confirm_action", "clarify"}
    reason = "CONFIRMATION_REQUIRED" if confirm else ("ESCALATION_REQUIRED" if intent in {"clarify", "escalate"} else "NO_TOOL_REQUIRED")
    state = "await_confirmation" if confirm else ("needs_clarification" if intent == "clarify" else "complete")
    return {
        "schema_version":"keren_dataset_v0.1",
        "id":f"gold_{num}",
        "task":{"category":category,"task_type":task_type,"difficulty":"hard","output_mode":mode},
        "user_input":{"text":text,"language":"hinglish","input_type":"text","normalized_text":text},
        "context":{"available_nodes":[],"available_tools":[],"network":{"internet_available":False,"internet_required":False}},
        "intent":{"name":intent,"confidence":"high","target":domain,"reason_code":reason},
        "node":None,"tool":None,"arguments":None,
        "safety":{"risk_level":risk,"authority":"confirmation_required" if confirm else "allowed","confirmation_required":confirm,"sensitive_data_present":False,"reason_code":reason if confirm else "LOW_RISK_LOCAL_ACTION"},
        "preconditions":[],"plan":[],"action":None,"observation":None,"verification":None,"recovery":None,
        "decision":{"state":state,"next_action":"request_confirmation" if confirm else None,"escalation_required":intent in {"clarify","escalate"},"reason_code":reason},
        "final_output":{"type":mode,"text":answer,"machine_output":{"status":state}},
        "training_labels":{"should_use_tool":False,"should_use_memory":category=="memory_procedures","should_use_internet":False,"should_clarify":intent=="clarify","should_confirm":confirm,"should_escalate":intent in {"clarify","escalate"},"should_recover":intent=="recover_action","should_refuse":False},
        "metadata":{"source":"gold_manual_v0.2_targeted","domain":domain,"language":"hinglish","quality_score":5,"approved":True,"synthetic":False,"sanitized":True,"dataset_split":None},
    }


def main():
    assert len(CASES) == 50
    assert len({r[0] for r in CASES}) == 50
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in CASES:
            f.write(json.dumps(record(row), ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Generated {len(CASES)} records -> {OUT}")


if __name__ == "__main__":
    main()
