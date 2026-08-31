# KEREN Student V0.1 Syllabus

This syllabus defines the capability coverage for Dataset V0.1 and the first specialist student model.

1. **Language & reasoning** — English, Hindi, Hinglish, spelling/STT errors, instruction understanding, basic maths/reasoning and conversation.
2. **KEREN self-knowledge & self-engineering** — understand architecture/code, audit, debug, root-cause, patch, test, optimize and detect regressions.
3. **Coding** — Python/FastAPI, JSON/API/WebSocket, Windows automation, PowerShell/shell, Git/GitHub, Flutter/Dart, Kotlin/Android, C/C++, Arduino/ESP32 and Termux/Linux basics.
4. **Embedded & electronics** — ESP32/Arduino/SBC, GPIO, UART/I2C/SPI/PWM/ADC, BLE/Wi-Fi, sensors, displays, motors, relays, firmware, serial, SRAM/PSRAM/flash and embedded debugging/networking.
5. **OS/system intelligence** — Windows, Android and Linux/Termux; processes, apps, files, services, browser, networking, device discovery, permissions and diagnostics.
6. **Computer/device control** — app/browser/UI control, semantic targets, multi-device routing and PC↔phone↔ESP32/SBC operations.
7. **Agentic execution** — understand → plan → select node/tool → precondition → act → observe → verify → recover/replan → complete.
8. **Safety & authority** — low-risk autonomy; consequential-action confirmation; credentials, money, deletion and permission protection; deterministic safety outside model; no fake execution.
9. **Memory & procedures** — context retrieval and reusable workflows; memory remains separate from model training.
10. **Business/operations intelligence** — routine engineering/business operations, reports, task delegation and exception escalation. Product-specific knowledge should enter through approved context/datasets rather than uncontrolled assumptions.
11. **Training Wheels Protocol** — sanitized candidate collection → owner-controlled dataset → retrain → benchmark → approve → deploy/rollback. No uncontrolled online weight modification.
12. **Offline-first behavior** — local model + memory + execution should work without internet; explicitly identify steps that truly require network access.
13. **Vision/perception interface** — reason over screenshots, accessibility, OCR and vision observations. General-purpose vision-model training is not a core V0.1 objective.
14. **Model/tool awareness** — distinguish: I know / I inferred / I planned / I executed / I verified.
15. **Uncertainty & escalation** — answer when known; use tools when checkable; clarify genuine ambiguity; use internet only for current/network data; admit inability; recover/escalate on execution failure; never claim success without evidence.

## Runtime output policy

- ACTION: ultra-compact structured output.
- STATUS: one short useful status.
- ANSWER: concise explanation.
- DEBUG: root cause + patch + verification.
- CODE: complete code required by the task.

The model should learn minimum-sufficient planning, not verbose reasoning essays.
