"""Free local Egyptian Arabic voice assistant."""

from __future__ import annotations

import json
import html
import base64
import io
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import wave
from datetime import date, datetime, timedelta
from pathlib import Path
from queue import Empty
from types import SimpleNamespace

import numpy as np
import requests
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
TEMP_DIR = Path(r"D:\Mo\whisper_temp")
ASR_PYTHON = Path(r"D:\Mo\qwencleo_env\Scripts\python.exe")
ASR_WORKER = APP_DIR / "qwencleo_worker.py"
LOCAL_LLM_SERVER = Path(r"D:\Mo\nilechat\runner\llama-server.exe")
LOCAL_LLM_MODEL = Path(r"D:\Mo\qwen3-4b\Qwen3-4B-Q4_K_M.gguf")
LOCAL_LLM_URL = "http://127.0.0.1:11435"
SARA_PROMPT_FILE = APP_DIR / "prompts" / "sara_local_core.txt"
PAYMENT_DB = Path(r"D:\Mo\WhisperApp\payment_promises.db")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
ENABLE_GEMINI_TTS = False
ENABLE_LOCAL_TTS = True
LOCAL_TTS_WORKER = APP_DIR / "windows_hoda_tts.ps1"
LOCAL_TTS_SPEAKER = "Microsoft Hoda (Arabic - Egypt)"
CALL_OPENER = "\u0623\u0647\u0644\u0627 \u064a\u0627 \u0641\u0646\u062f\u0645\u060c \u0645\u0639 \u062d\u0636\u0631\u062a\u0643 \u0633\u0627\u0631\u0629\u060c \u0627\u0644\u0645\u0633\u0627\u0639\u062f\u0629 \u0627\u0644\u0622\u0644\u064a\u0629 \u0645\u0646 \u062a\u0645\u0648\u064a\u0644\u064a. \u0628\u0643\u0644\u0645 \u062d\u0636\u0631\u062a\u0643 \u0639\u0634\u0627\u0646 \u0627\u0644\u0642\u0633\u0637 \u0627\u0644\u0645\u0633\u062a\u062d\u0642. \u0647\u0648 \u0623\u0646\u0627 \u0628\u0643\u0644\u0645 \u0627\u0644\u0634\u062e\u0635 \u0627\u0644\u0635\u062d\u061f"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

SYSTEM_PROMPT = (
    "أنت مساعد محادثة مصري ذكي ودقيق وودود. تابع نفس الموضوع عبر الرسائل وافهم كل رسالة "
    "في ضوء المحادثة السابقة، ولا تتعامل معها كموضوع جديد. ساعد في الشرح والتحليل والعصف "
    "الذهني والكتابة والتلخيص والترجمة. لو المستخدم يصحح جزءاً من طلب سابق، عدّل الإجابة "
    "السابقة بناءً على التصحيح ولا تبدأ من الصفر. النص قد يكون ناتجاً من تحويل صوت وفيه "
    "أخطاء بسيطة؛ افهم المعنى العام ولا ترفض الطلب كله بسبب كلمة واحدة. لا تخترع أسماء أو "
    "شركات أو أرقام أو حقائق، ولا تدّعِ تنفيذ أفعال خارجية مثل إرسال إيميل. إذا كانت معلومة "
    "مهمة غير واضحة، اذكر [غير واضح] واسأل سؤالاً واحداً قصيراً. رد بالمصري الطبيعي، إلا إذا "
    "طلب المستخدم لغة أو صيغة أخرى. عندما يطلب كتابة رسالة أو إيميل، اكتب النص النهائي فوراً "
    "ولا تقل فقط إنك ستكتبه أو تعدله. أحدث تصحيح من المستخدم يلغي المعلومة القديمة المتعارضة "
    "معه تماماً؛ لا تستخدم المعلومة الملغاة في الإجابة. إذا طلب النص بالإنجليزية، اكتب النص "
    "النهائي كله بالإنجليزية فقط من غير مقدمة أو شرح بالعربي."
)


def stop_llm() -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/IM", "llama-server.exe", "/F"],
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        time.sleep(0.8)


def transcribe_asr(path: Path) -> tuple[str, float, float]:
    """Run QwenCleo alone, after unloading the LLM."""
    stop_llm()
    env = os.environ.copy()
    env.update({
        "HF_HOME": r"D:\Mo\huggingface",
        "HUGGINGFACE_HUB_CACHE": r"D:\Mo\huggingface\hub",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "TEMP": str(TEMP_DIR),
        "TMP": str(TEMP_DIR),
    })
    result = subprocess.run(
        [str(ASR_PYTHON), str(ASR_WORKER), str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=600,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[-1000:] or "QwenCleo failed")
    line = next((x for x in reversed(result.stdout.splitlines()) if x.startswith("{")), None)
    if not line:
        raise RuntimeError("QwenCleo returned no transcription")
    payload = json.loads(line)
    with wave.open(str(path), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
    return payload.get("text", "").strip(), float(payload.get("elapsed", 0)), duration


def is_repetitive_answer(text: str, answer: str) -> bool:
    def words(value: str) -> list[str]:
        return re.findall(r"[\w\u0600-\u06ff]+", value.lower())
    source_words = words(text)
    answer_words = words(answer)
    if not answer_words:
        return True
    source_set = set(source_words)
    overlap = sum(word in source_set for word in answer_words) / len(answer_words)
    source_compact = "".join(source_words)
    answer_compact = "".join(answer_words)
    return (
        overlap >= 0.72
        or (len(source_compact) > 15 and source_compact in answer_compact)
        or answer_compact == source_compact
    )


def clean_requested_language(text: str, answer: str) -> str:
    request = text.lower()
    wants_english = any(
        marker in request for marker in ("english", "بالانجليزي", "بالإنجليزي")
    )
    if not wants_english:
        return answer.strip()
    positions = [
        position for marker in ("subject:", "dear ")
        if (position := answer.lower().find(marker)) >= 0
    ]
    if positions:
        return answer[min(positions):].strip()
    return answer.strip()


def ensure_local_llm() -> None:
    """Start Qwen3 after ASR releases memory and wait until it is ready."""
    try:
        if requests.get(f"{LOCAL_LLM_URL}/health", timeout=2).json().get("status") == "ok":
            return
    except (requests.RequestException, ValueError):
        pass
    if not LOCAL_LLM_SERVER.exists() or not LOCAL_LLM_MODEL.exists():
        raise RuntimeError("The local Qwen3 model or llama server is missing.")
    log_dir = LOCAL_LLM_MODEL.parent
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(
        [
            str(LOCAL_LLM_SERVER), "-m", str(LOCAL_LLM_MODEL),
            "--host", "127.0.0.1", "--port", "11435", "-c", "8192",
            "-t", "6", "--jinja", "--no-webui",
        ],
        cwd=str(LOCAL_LLM_SERVER.parent),
        stdout=open(log_dir / "app-server-out.log", "a", encoding="utf-8"),
        stderr=open(log_dir / "app-server-error.log", "a", encoding="utf-8"),
        creationflags=creationflags,
    )
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            if requests.get(f"{LOCAL_LLM_URL}/health", timeout=2).json().get("status") == "ok":
                return
        except (requests.RequestException, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError("Local Qwen3 could not start within 90 seconds.")


def ask_llm(text: str) -> str:
    """Ask the fully local Qwen3-4B conversation model."""
    normalized = re.sub(r"[ًٌٍَُِّْـ]", "", text.lower())
    hardship_words = ("فقدت شغلي", "الشغل", "المرتب", "مريض", "مرض", "ظروف", "مش قادر", "معنديش فلوس")
    domain_words = ("قسط", "دفع", "ادفع", "أدفع", "سداد", "ميعاد", "موعد", "فلوس", "مبلغ", "فاتورة", "ايصال", "إيصال", "قرض")
    if any(word in normalized for word in hardship_words):
        intent = "hardship"
    elif not any(word in normalized for word in domain_words):
        intent = "off_topic"
    else:
        intent = "collection"
    if intent == "off_topic":
        return "نقدر نتكلم في موضوع القسط وميعاد الدفع. هيتحدد يوم مناسب؟"
    ensure_local_llm()
    base_prompt = (
        "إنت سارة وكيلة تحصيل مصرية. ردي بالمصري الطبيعي في جملة واحدة أو جملتين قصيرين. "
        "ممنوع تبدأي بتحية أو ترجعي تسألي عن الهوية. ما تكرريش كلام العميل وما تخترعيش بيانات. /no_think\n"
    )
    tasks = {
        "off_topic": (
            "المهمة الحالية: السؤال بعيد عن القسط. ما تجاوبيش على محتواه. اعترفي باختصار "
            "وارجعي لموضوع القسط واسألي عن ميعاد الدفع."
        ),
        "hardship": (
            "المهمة الحالية: العميل ذكر ظرف مادي أو صحي. ابدئي بتعاطف مصري قصير، وبعده "
            "اسألي إذا فيه وقت متوقع للدفع أو يفضل متابعة موظف. ممنوع تقولي هل يمكنك أو هل ترغب."
        ),
        "collection": (
            "المهمة الحالية: كملي نفس مناقشة القسط بشكل طبيعي. لو معلومة ناقصة اسألي عنها، "
            "ولو محتاجة بيانات مش موجودة قولي إنها مش متاحة."
        ),
    }
    system_prompt = base_prompt + tasks[intent]
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(st.session_state.chat_history[-6:])
    messages.append({"role": "user", "content": text})
    response = requests.post(
        f"{LOCAL_LLM_URL}/v1/chat/completions",
        json={
            "model": LOCAL_LLM_MODEL.name,
            "messages": messages,
            "temperature": 0.45,
            "top_p": 0.8,
            "presence_penalty": 1.2,
            "max_tokens": 180,
            "stream": False,
        },
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(f"Local Qwen3 error {response.status_code}: {response.text[:300]}")
    try:
        answer = response.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Local Qwen3 returned no answer.") from exc
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    if not answer:
        raise RuntimeError("Local Qwen3 returned an empty answer.")
    malformed = "�" in answer or answer.count("?") > max(3, len(answer) // 8)
    if intent == "hardship":
        has_empathy = any(word in answer for word in ("متفهم", "فاهم", "ظروف", "حصل خير", "ولا يهم"))
        too_formal = any(word in answer for word in ("هل يمكنك", "هل ترغب", "يرجى"))
        if malformed or not has_empathy or too_formal:
            return "متفهمة الظروف. فيه وقت متوقع يقدر يتم فيه الدفع، ولا يتسجل طلب متابعة من موظف؟"
    return answer


def ask_gemini_legacy(text: str) -> str:
    api_key = (
        st.session_state.get("gemini_api_key")
        or st.secrets.get("GEMINI_API_KEY", "")
        or os.getenv("GEMINI_API_KEY", "")
    )
    if not api_key.strip():
        raise RuntimeError("Enter your Gemini API key in the sidebar before speaking")

    contents = []
    for message in st.session_state.chat_history[-20:]:
        role = "model" if message["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": message["content"]}]})
    contents.append({"role": "user", "parts": [{"text": text}]})
    language_context = " ".join(
        message["content"] for message in st.session_state.chat_history[-8:]
        if message["role"] == "user"
    ) + " " + text
    http = requests.Session()
    http.trust_env = False

    def generate(system_prompt: str) -> str:
        response = http.post(
            GEMINI_URL,
            headers={"x-goog-api-key": api_key.strip()},
            json={
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": contents,
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 1000,
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            },
            timeout=90,
        )
        if not response.ok:
            detail = response.text[:500]
            raise RuntimeError(f"Gemini API error {response.status_code}: {detail}")
        payload = response.json()
        try:
            parts = payload["candidates"][0]["content"]["parts"]
            return "".join(part.get("text", "") for part in parts).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Gemini returned no answer: {payload}") from exc

    answer = clean_requested_language(language_context, generate(SYSTEM_PROMPT))
    if not is_repetitive_answer(text, answer):
        return answer

    retry = clean_requested_language(language_context, generate(
        SYSTEM_PROMPT
        + " الإجابة السابقة كررت كلام المستخدم وتم رفضها. أجب على المقصود مباشرة، "
          "ولا تنسخ أو تلخص جملة المستخدم."
    ))
    if not is_repetitive_answer(text, retry):
        return retry
    return "فاهم إنك بتكمل نفس الموضوع، بس محتاج أعرف: تحب أساعدك بإيه تحديداً دلوقتي؟"


def ask_local_agent(text: str) -> str:
    """Let local Qwen reason over the full conversation under business guardrails."""
    ensure_local_llm()
    if not SARA_PROMPT_FILE.exists():
        raise RuntimeError("Sara's local agent prompt is missing.")
    system_prompt = SARA_PROMPT_FILE.read_text(encoding="utf-8-sig").strip()
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(st.session_state.chat_history[-12:])
    messages.append({"role": "user", "content": text})

    def generate(extra_instruction: str = "") -> str:
        request_messages = list(messages)
        if extra_instruction:
            request_messages.append({"role": "system", "content": extra_instruction})
        response = requests.post(
            f"{LOCAL_LLM_URL}/v1/chat/completions",
            json={
                "model": LOCAL_LLM_MODEL.name,
                "messages": request_messages,
                "temperature": 0.25,
                "top_p": 0.75,
                "presence_penalty": 0.10,
                "repetition_penalty": 1.08,
                "max_tokens": 180,
                "stream": False,
            },
            timeout=120,
        )
        if not response.ok:
            raise RuntimeError(
                f"Local Qwen3 error {response.status_code}: {response.text[:300]}"
            )
        try:
            answer = response.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Local Qwen3 returned no answer.") from exc
        return re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    answer = generate()
    if not answer:
        raise RuntimeError("Local Qwen3 returned an empty answer.")
    if is_repetitive_answer(text, answer):
        answer = generate(
            "ردّي على معنى كلام العميل بشكل طبيعي ومباشر. لا تكرري كلامه ولا ردًا سابقًا، "
            "واسألي سؤال توضيح واحد فقط إذا كان المعنى غير واضح. /no_think"
        )
    if not answer:
        raise RuntimeError("Local Qwen3 returned an empty retry.")
    return answer


def valid_transcript(text: str, duration: float) -> bool:
    compact = "".join(text.split())
    return bool(compact) and len(compact) >= 2 and len(compact) <= max(35, duration * 20)


def write_wav(path: Path, samples: np.ndarray) -> None:
    if len(samples):
        samples = samples - float(np.mean(samples))
        peak = float(np.max(np.abs(samples)))
        if 0.001 < peak < 0.85:
            samples *= min(0.85 / peak, 6.0)
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm.tobytes())


ARABIC_WEEKDAYS = {
    "الاثنين": 0, "الاتنين": 0, "الثلاثاء": 1, "التلات": 1, "التلاتاء": 1,
    "الأربعاء": 2, "الاربعاء": 2, "الخميس": 3, "الجمعة": 4,
    "السبت": 5, "الأحد": 6, "الاحد": 6,
}
ARABIC_DAY_WORDS = {
    1: "واحد", 2: "اتنين", 3: "تلاتة", 4: "اربعة", 5: "خمسة", 6: "ستة",
    7: "سبعة", 8: "تمانية", 9: "تسعة", 10: "عشرة", 11: "حداشر", 12: "اتناشر",
    13: "تلتاشر", 14: "اربعتاشر", 15: "خمستاشر", 16: "ستاشر", 17: "سبعتاشر",
    18: "تمنتاشر", 19: "تسعتاشر", 20: "عشرين", 21: "واحد وعشرين",
    22: "اتنين وعشرين", 23: "تلاتة وعشرين", 24: "اربعة وعشرين",
    25: "خمسة وعشرين", 26: "ستة وعشرين", 27: "سبعة وعشرين",
    28: "تمانية وعشرين", 29: "تسعة وعشرين", 30: "تلاتين", 31: "واحد وتلاتين",
}
ARABIC_MONTHS = (
    "", "يناير", "فبراير", "مارس", "ابريل", "مايو", "يونيو",
    "يوليو", "اغسطس", "سبتمبر", "اكتوبر", "نوفمبر", "ديسمبر",
)


def spoken_date(value: date) -> str:
    return f"{ARABIC_DAY_WORDS[value.day]} {ARABIC_MONTHS[value.month]}"


def extract_payment_date(text: str) -> date | None:
    normalized = re.sub(r"[ًٌٍَُِّْـ]", "", text.lower())
    today = date.today()
    if "بعد بكرة" in normalized or "بعد بكره" in normalized:
        return today + timedelta(days=2)
    if "بكرة" in normalized or "بكره" in normalized or "غدا" in normalized:
        return today + timedelta(days=1)
    numeric = re.search(r"(?<!\d)([0-3]?\d)[/-]([01]?\d)(?:[/-](\d{2,4}))?", normalized)
    if numeric:
        day, month = int(numeric.group(1)), int(numeric.group(2))
        year = int(numeric.group(3) or today.year)
        year += 2000 if year < 100 else 0
        try:
            candidate = date(year, month, day)
            if not numeric.group(3) and candidate < today:
                candidate = date(year + 1, month, day)
            return candidate
        except ValueError:
            return None
    for word, weekday in ARABIC_WEEKDAYS.items():
        if word in normalized:
            delta = (weekday - today.weekday()) % 7
            if delta == 0 or "الجاي" in normalized or "القادم" in normalized:
                delta = delta or 7
            return today + timedelta(days=delta)
    return None


def save_payment_promise(payment_date: date, source_text: str) -> None:
    PAYMENT_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(PAYMENT_DB) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS payment_promises (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                payment_date TEXT NOT NULL,
                confirmed_at TEXT NOT NULL,
                source_text TEXT NOT NULL
            )"""
        )
        connection.execute(
            "INSERT INTO payment_promises(customer_id,payment_date,confirmed_at,source_text) VALUES(?,?,?,?)",
            (
                st.session_state.get("customer_id", "demo-customer"),
                payment_date.isoformat(), datetime.now().isoformat(timespec="seconds"), source_text,
            ),
        )


def payment_workflow_reply(text: str) -> str | None:
    """Handle critical collection steps without depending on LLM judgment."""
    clear_text = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", text.lower())

    history = st.session_state.get("chat_history", [])
    awaiting_identity = bool(
        history
        and history[-1].get("role") == "assistant"
        and history[-1].get("content") == CALL_OPENER
    )
    identity_yes = any(
        phrase in clear_text
        for phrase in ("اه", "ايوه", "أيوه", "تمام", "انا هو", "أنا هو", "معاك")
    )
    identity_no = any(
        phrase in clear_text
        for phrase in ("مش انا", "مش أنا", "رقم غلط", "مش الشخص", "لا مش")
    )
    if awaiting_identity and identity_no:
        return "متأسفة على الإزعاج يا فندم. مش هذكر أي تفاصيل، وهراجع بيانات الرقم."
    if awaiting_identity and identity_yes:
        return "تمام يا فندم. حضرتك متوقع تقدر تدفع القسط إمتى؟"

    relative_delay = re.search(
        r"(?:مش\s+(?:هقدر\s+|هاقدر\s+|هدفع\s+)?قبل|بعد|كمان)\s+"
        r"(سنتين|سنه|سنة|شهرين|شهر|اسبوعين|أسبوعين|اسبوع|أسبوع)",
        clear_text,
    )
    if relative_delay:
        period = relative_delay.group(1)
        return (
            f"فاهمة إن حضرتك مش هتقدر تدفع قبل {period}. "
            "أنا أقدر أسجل وعد دفع خلال يومين بس، فالمدة دي خارج المواعيد المتاحة عندي. "
            "تحب أسجل طلب متابعة مع موظف؟"
        )

    if any(greeting in clear_text for greeting in ("السلام عليكم", "سلام عليكم")):
        return "وعليكم السلام يا فندم. بالنسبة للقسط، حضرتك متوقع تقدر تدفع إمتى؟"

    next_year = any(
        phrase in clear_text
        for phrase in ("السنه الجايه", "السنة الجاية", "العام الجاي", "بعد سنه", "بعد سنة")
    )
    if next_year:
        return (
            "فاهمة إن حضرتك بتقول إن الدفع مش هيكون غير السنة الجاية. "
            "المدة دي خارج المواعيد المتاحة عندي، تحب أسجل طلب متابعة مع موظف؟"
        )

    if any(phrase in clear_text for phrase in ("حاسس بسنه", "حاسس بسنة", "هستنى سنه", "هستنى سنة")):
        return "معلش يا فندم، هل تقصد إنك مش هتقدر تدفع غير بعد سنة؟"

    two_month_delay = any(
        phrase in clear_text
        for phrase in (
            "مش هدفع قبل شهرين", "مش هقدر ادفع قبل شهرين", "مش هقدر أدفع قبل شهرين",
            "بعد شهرين", "كمان شهرين",
        )
    )
    if two_month_delay:
        return (
            "فاهمة إن حضرتك مش هتقدر تدفع قبل ما يعدي شهرين. "
            "المدة دي خارج المواعيد المتاحة عندي، تحب أسجل طلب متابعة مع موظف؟"
        )

    unable_to_pay = any(
        phrase in clear_text
        for phrase in ("مش عارف ادفع", "مش عارف أدفع", "مش قادر ادفع", "مش قادر أدفع")
    )
    if unable_to_pay:
        return (
            "متفهمة يا فندم إن حضرتك مش قادر تدفع دلوقتي. "
            "تحب أسجل طلب متابعة مع موظف عشان يناقش معاك الحلول المتاحة؟"
        )
    normalized = re.sub(r"[ًٌٍَُِّْـ]", "", text.lower())
    st.session_state.setdefault("pending_payment_date", None)
    if any(value in normalized for value in ("مش الشخص", "رقم غلط", "مش انا", "مش أنا")):
        st.session_state.pending_payment_date = None
        return "متأسف على الإزعاج يا فندم. مش هذكر أي تفاصيل، وهراجع بيانات الرقم."
    if any(value in normalized for value in ("موظف", "حد من خدمة", "خدمة العملاء", "بني ادم")):
        return "تمام يا فندم، هسجل إن حضرتك طالب تتكلم مع موظف."
    if any(value in normalized for value in ("القسط كام", "المبلغ كام", "عليا كام", "علي كام")):
        return "قيمة القسط مش متاحة عندي في بيانات الاختبار، ومش هقول رقم غير مؤكد."
    candidate = extract_payment_date(normalized)
    if candidate:
        today = date.today()
        last_allowed = today + timedelta(days=2)
        if candidate < today:
            return "اليوم ده فات، فلازم يتحدد يوم من النهاردة لحد بعد بكرة."
        if candidate > last_allowed:
            return f"النظام بيقبل معاد لحد يوم {spoken_date(last_allowed)} بس. هيتحدد يوم في المدة دي؟"
        st.session_state.pending_payment_date = candidate.isoformat()
        return f"يبقى كده قلنا يوم {spoken_date(candidate)}، تمام؟"
    pending = st.session_state.pending_payment_date
    words = set(re.findall(r"[\w\u0600-\u06ff]+", normalized))
    confirms = any(value in normalized for value in ("ايوه", "أيوه", "اه", "آه", "تمام", "صح", "موافق"))
    rejects = bool(words.intersection({"لا", "لأ"})) or any(
        value in normalized for value in ("غير الموعد", "غير اليوم", "استنى", "استني")
    )
    if pending and confirms and not rejects:
        confirmed_date = date.fromisoformat(pending)
        save_payment_promise(confirmed_date, text)
        st.session_state.pending_payment_date = None
        confirmation = f"تمام، اتسجل ميعاد الدفع يوم {spoken_date(confirmed_date)}."
        if any(value in normalized for value in ("فاتورة", "إيصال", "ايصال")):
            return confirmation + " وبالنسبة للفاتورة، بياناتها مش متاحة عندي دلوقتي ومش هأكد معلومة من غير بيانات."
        return confirmation + " شكرًا لحضرتك."
    if pending and rejects:
        st.session_state.pending_payment_date = None
        return "ولا يهمك يا فندم، قولّي اليوم الجديد اللي يناسبك."
    if any(value in normalized for value in ("ايام", "أيام", "مواعيد", "ادفع فيها", "أدفع فيها", "list")):
        return "قولّي اليوم اللي يناسبك للدفع، وأنا هحوّله لتاريخ واضح وأأكدّه مع حضرتك قبل التسجيل."
    if any(value in normalized for value in ("ادفع", "أدفع", "الدفع", "القسط")):
        return "تمام يا فندم، تقدر تحدد اليوم اللي ناوي تدفع فيه؟"
    return None


def add_turn(text: str) -> str:
    answer = payment_workflow_reply(text) or ask_local_agent(text)
    st.session_state.chat_history.extend([
        {"role": "user", "content": text},
        {"role": "assistant", "content": answer},
    ])
    if ENABLE_LOCAL_TTS:
        try:
            st.session_state.last_reply_audio = synthesize_reply(answer)
            st.session_state.tts_error = ""
        except Exception as exc:
            st.session_state.last_reply_audio = b""
            st.session_state.tts_error = str(exc)
    else:
        st.session_state.last_reply_audio = b""
        st.session_state.tts_error = ""
    return answer


def egyptianize_for_speech(text: str) -> str:
    """Remove display-only markup; never rewrite facts or unknown wording."""
    spoken = re.sub(r"[*_#`]+", "", text).strip()
    return re.sub(r"\s+", " ", spoken)


def synthesize_reply(text: str) -> bytes:
    """Synthesize locally with the installed Microsoft Hoda ar-EG voice."""
    if not LOCAL_TTS_WORKER.exists():
        raise RuntimeError("The local Hoda TTS worker is missing.")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    text_file = TEMP_DIR / f"sara_reply_{time.time_ns()}.txt"
    output = TEMP_DIR / f"sara_reply_{time.time_ns()}.wav"
    text_file.write_text(egyptianize_for_speech(text), encoding="utf-8")
    env = os.environ.copy()
    env.update({
        "TEMP": str(TEMP_DIR),
        "TMP": str(TEMP_DIR),
    })
    result = subprocess.run(
        [
            "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(LOCAL_TTS_WORKER),
            "-TextFile", str(text_file), "-OutputFile", str(output),
        ],
        capture_output=True,
        timeout=60,
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode(errors="replace").strip()[-500:]
        raise RuntimeError(f"Hoda TTS failed: {detail}")
    audio = output.read_bytes()
    text_file.unlink(missing_ok=True)
    output.unlink(missing_ok=True)
    return audio


def synthesize_reply_gemini_legacy(text: str) -> bytes:
    """Generate a WAV response while keeping TTS failure separate from chat."""
    key = st.session_state.get("gemini_api_key", "").strip()
    if not key:
        raise RuntimeError("Gemini API key is missing.")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_TTS_MODEL}:generateContent?key={key}"
    )
    prompt = (
        "اقرأ النص التالي فقط بصوت طبيعي وودود. استخدم نطقًا عربيًا مصريًا واضحًا، "
        "وسرعة محادثة عادية، ولا تضف أي كلمات غير موجودة في النص:\n\n" + text
    )
    response = requests.post(
        url,
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": "Kore"}
                    }
                },
            },
        },
        timeout=90,
    )
    if not response.ok:
        raise RuntimeError(f"Speech service returned {response.status_code}.")
    parts = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
    inline = next((part.get("inlineData") or part.get("inline_data") for part in parts if part.get("inlineData") or part.get("inline_data")), None)
    if not inline or not inline.get("data"):
        raise RuntimeError("Gemini returned no speech audio.")
    pcm = base64.b64decode(inline["data"])
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm)
    return output.getvalue()


def render_conversation(target=None) -> None:
    holder = target.container() if target is not None else st.container()
    with holder:
        rows = []
        for message in st.session_state.chat_history:
            role = message["role"]
            avatar = "م" if role == "user" else "✦"
            label = "You" if role == "user" else "MasriFlow"
            content = html.escape(message["content"]).replace("\n", "<br>")
            rows.append(
                f'<div class="chat-row {role}"><div class="chat-avatar">{avatar}</div>'
                f'<div class="chat-message"><div class="chat-author">{label}</div>{content}</div></div>'
            )
        if not rows:
            rows.append(
                '<div class="chat-welcome"><div class="welcome-spark">✦</div>'
                '<div><b>Call ready</b><br><span>Start speaking or send a message below.</span></div></div>'
            )
        st.markdown(
            '<div class="conversation-canvas"><div class="call-heading">Call conversation</div>'
            + "".join(rows) + '</div>',
            unsafe_allow_html=True,
        )


def render_hidden_reply_audio(holder, audio_bytes: bytes) -> None:
    """Autoplay Sara's reply without exposing a recording/player control."""
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    holder.markdown(
        f'<audio autoplay preload="auto" style="display:none">'
        f'<source src="data:audio/wav;base64,{encoded}" type="audio/wav">'
        '</audio>',
        unsafe_allow_html=True,
    )


def run_live() -> None:
    import av
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    st.session_state.setdefault("voice_active", False)

    conversation_col, voice_col = st.columns([1.12, 0.88], gap="large")
    with conversation_col:
        with st.container(border=True):
            st.markdown('<div class="section-label">LIVE WORKSPACE</div>', unsafe_allow_html=True)
            st.markdown("### Conversation")
            conversation_box = st.empty()
            if not st.session_state.chat_history:
                st.markdown(
                    """
                    <div class="empty-state">
                      <div class="empty-icon">⌁</div>
                      <div class="empty-title">Your conversation starts here</div>
                      <div class="empty-copy">Start the call and speak in Egyptian Arabic, or type below. Your transcript and the AI reply will appear together here.</div>
                      <div class="flow-steps"><span>1&nbsp; Speak</span><b>→</b><span>2&nbsp; Transcribe</span><b>→</b><span>3&nbsp; Respond</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            render_conversation(conversation_box)
            audio_caption = st.empty()
            audio_box = st.empty()
            if st.session_state.get("last_reply_audio"):
                render_hidden_reply_audio(audio_box, st.session_state.last_reply_audio)
            elif st.session_state.get("tts_error"):
                audio_box.error(f"Voice unavailable: {st.session_state.tts_error}")
            st.markdown('<div class="chat-label">CHAT WITH MASRIFLOW</div>', unsafe_allow_html=True)
            with st.form("conversation_message", clear_on_submit=True):
                typed_message = st.text_input(
                    "Message",
                    placeholder="Send a message…",
                    label_visibility="collapsed",
                )
                tools_col, send_col = st.columns([8, 1])
                with tools_col:
                    st.markdown('<div class="composer-tools"><span>⚙</span><span>🎙</span><span>📎</span><small>Arabic + English</small></div>', unsafe_allow_html=True)
                with send_col:
                    send_message = st.form_submit_button("↑", type="primary", use_container_width=True)
            if send_message and typed_message.strip():
                with st.spinner("Local Qwen3 is answering…"):
                    add_turn(typed_message.strip())
                st.rerun()

    with voice_col:
        with st.container(border=True):
            st.markdown(
                """
                <div class="voice-stage">
                  <div class="status-pill"><span></span> VOICE SESSION</div>
                  <div class="voice-orb"><div class="orb-core">☎</div></div>
                  <div class="voice-title">Speak naturally</div>
                  <div class="voice-copy">Pause for 1.2 seconds and MasriFlow will respond.</div>
                </div>
                """, unsafe_allow_html=True,
            )
            call_label = "End call" if st.session_state.voice_active else "Start call"
            if st.button(
                f"☎  {call_label}",
                key="voice_call_toggle",
                type="primary",
                use_container_width=True,
            ):
                starting = not st.session_state.voice_active
                st.session_state.voice_active = starting
                if starting and not st.session_state.get("call_opened"):
                    if not any(message.get("content") == CALL_OPENER for message in st.session_state.chat_history):
                        st.session_state.chat_history.append({"role": "assistant", "content": CALL_OPENER})
                    with st.spinner("Sara is preparing the opening voice message…"):
                        try:
                            st.session_state.last_reply_audio = synthesize_reply(CALL_OPENER)
                            st.session_state.tts_error = ""
                        except Exception as exc:
                            st.session_state.last_reply_audio = b""
                            st.session_state.tts_error = str(exc)
                    st.session_state.call_opened = True
                elif not starting:
                    st.session_state.call_opened = False
                st.rerun()
            if st.session_state.voice_active:
                st.success("🎙 Microphone active — allow browser access, then speak. Pause for 1.2 seconds when finished.")
            else:
                st.info("Click Start call. Sara speaks first; then this panel confirms that your microphone is active.")
            context = webrtc_streamer(
                key="egyptian-live-v2", mode=WebRtcMode.SENDONLY,
                media_stream_constraints={"video": False, "audio": {
                    "echoCancellation": True, "noiseSuppression": True,
                    "autoGainControl": True}},
                audio_receiver_size=256, async_processing=True,
                desired_playing_state=st.session_state.voice_active,
            )
            status = st.empty()

    if not (context.state.playing and context.audio_receiver):
        return
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    phrase = np.array([], dtype=np.float32)
    pre_roll = np.array([], dtype=np.float32)
    speech_started = False
    silent = voiced = 0
    status.success("Listening — start speaking.")
    try:
        while context.state.playing:
            try:
                frames = context.audio_receiver.get_frames(timeout=1)
            except Empty:
                continue
            for frame in frames:
                items = resampler.resample(frame)
                items = items if isinstance(items, list) else [items]
                for item in items:
                    if item is None:
                        continue
                    chunk = item.to_ndarray().reshape(-1).astype(np.float32) / 32768
                    rms = float(np.sqrt(np.mean(chunk * chunk))) if len(chunk) else 0
                    if rms >= 0.018:
                        if not speech_started and len(pre_roll):
                            phrase = np.concatenate((phrase, pre_roll))
                        speech_started = True
                        silent = 0
                        voiced += len(chunk)
                        phrase = np.concatenate((phrase, chunk))
                        status.info("Speech detected — listening until you pause.")
                    elif speech_started:
                        phrase = np.concatenate((phrase, chunk))
                        silent += len(chunk)
                    else:
                        pre_roll = np.concatenate((pre_roll, chunk))[-6400:]
                    finished = (speech_started and voiced >= 12800 and silent >= 19200) or len(phrase) >= 480000
                    if not finished:
                        continue
                    usable = phrase[:-silent] if silent else phrase
                    TEMP_DIR.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(prefix="voice_", dir=TEMP_DIR) as folder:
                        wav_path = Path(folder) / "phrase.wav"
                        write_wav(wav_path, usable)
                        shutil.copy2(wav_path, TEMP_DIR / "last_live_phrase.wav")
                        status.warning("Transcribing with QwenCleo…")
                        text, elapsed, duration = transcribe_asr(wav_path)
                    if valid_transcript(text, duration):
                        st.session_state.transcripts.append(text)
                        status.warning("Transcript ready — local Qwen3 is following the conversation…")
                        audio_caption.caption("Generating Sara's new voice reply…")
                        audio_box.empty()
                        add_turn(text)
                        render_conversation(conversation_box)
                        if st.session_state.get("last_reply_audio"):
                            audio_caption.empty()
                            render_hidden_reply_audio(audio_box, st.session_state.last_reply_audio)
                        elif st.session_state.get("tts_error"):
                            audio_box.error(f"Voice unavailable: {st.session_state.tts_error}")
                        status.success(f"Completed in {elapsed:.1f}s")
                    else:
                        status.error("The audio was too short or unreliable. Please repeat it.")
                    phrase = pre_roll = np.array([], dtype=np.float32)
                    speech_started = False
                    silent = voiced = 0
                    for _ in range(300):
                        try:
                            context.audio_receiver.get_frames(timeout=0.01)
                        except Empty:
                            break
    except Exception as exc:
        st.error(f"Automatic listening failed: {exc}")


TEMP_DIR.mkdir(parents=True, exist_ok=True)
st.set_page_config(page_title="MasriFlow Voice", page_icon="🎙️", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      :root { --ink:#18202f; --muted:#778092; --line:#e6e9f0; --cyan:#40d9d2; --blue:#3157d5; --violet:#7457f5; }
      .stApp { background:radial-gradient(circle at 82% 0%,#e7f7ff 0,transparent 29%),#f6f7fb; color:var(--ink); }
      [data-testid="stHeader"] { background:transparent; }
      .block-container { max-width:1480px; padding:1.25rem 2.3rem 3rem; }
      [data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] { display:none !important; }
      .brandbar { display:flex; align-items:center; justify-content:space-between; background:rgba(255,255,255,.86); border:1px solid var(--line); box-shadow:0 12px 35px rgba(31,45,80,.06); border-radius:22px; padding:16px 20px; margin-bottom:20px; backdrop-filter:blur(16px); }
      .brand { display:flex; gap:12px; align-items:center; }
      .brandmark { width:42px; height:42px; border-radius:14px; display:grid; place-items:center; color:white; font-size:22px; font-weight:800; background:linear-gradient(145deg,var(--violet),var(--blue) 55%,var(--cyan)); box-shadow:0 8px 18px rgba(64,87,213,.25); }
      .brandname { font-size:20px; font-weight:800; letter-spacing:-.4px; }
      .brandsub { color:var(--muted); font-size:12px; margin-top:1px; }
      .engine-row { display:flex; gap:8px; flex-wrap:wrap; }
      .engine { background:#f2f4f9; color:#4f596b; border:1px solid #e4e8f0; padding:7px 11px; border-radius:999px; font-size:12px; font-weight:650; }
      .engine.live { background:#e8fbf7; border-color:#c3f1e8; color:#07816d; }
      div[data-testid="stVerticalBlockBorderWrapper"] { background:rgba(255,255,255,.96); border:1px solid var(--line); border-radius:24px; box-shadow:0 18px 50px rgba(24,32,47,.07); padding:8px; min-height:535px; }
      .section-label { color:#7b8496; font-size:11px; font-weight:800; letter-spacing:1.6px; margin-bottom:4px; }
      .voice-stage { min-height:385px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding:18px 10px 8px; }
      .status-pill { font-size:11px; font-weight:800; letter-spacing:1.2px; color:#596276; background:#f3f5f9; border:1px solid #e5e8ef; padding:7px 11px; border-radius:999px; margin-bottom:30px; }
      .status-pill span { display:inline-block; width:7px; height:7px; border-radius:50%; background:#27c59b; margin-right:7px; box-shadow:0 0 0 4px rgba(39,197,155,.12); }
      .voice-orb { width:245px; height:245px; border-radius:46% 54% 51% 49% / 54% 44% 56% 46%; padding:14px; display:grid; place-items:center; background:conic-gradient(from 20deg,#173fba,#58e3db,#e0ffff,#6458f5,#173fba,#58e3db); box-shadow:0 28px 70px rgba(35,79,190,.24),inset 0 0 35px rgba(255,255,255,.5); animation:floatOrb 6s ease-in-out infinite,spinHue 13s linear infinite; }
      .orb-core { width:82px; height:82px; border-radius:50%; display:grid; place-items:center; background:#0c1020; color:white; font-size:34px; font-weight:800; border:7px solid rgba(255,255,255,.95); box-shadow:0 12px 25px rgba(10,17,40,.28); }
      .voice-title { font-size:24px; font-weight:800; margin-top:27px; letter-spacing:-.5px; }
      .voice-copy { color:var(--muted); font-size:14px; margin-top:5px; max-width:330px; }
      .empty-state { min-height:235px; border:1px dashed #cfd6e6; border-radius:20px; background:linear-gradient(145deg,#fbfcff,#f2f7ff); display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding:26px; margin:14px 0 18px; }
      .empty-icon { width:48px; height:48px; border-radius:16px; display:grid; place-items:center; color:white; font-size:28px; font-weight:800; background:linear-gradient(145deg,var(--violet),var(--blue),var(--cyan)); box-shadow:0 12px 24px rgba(49,87,213,.22); }
      .empty-title { margin-top:15px; font-size:19px; font-weight:800; color:var(--ink); }
      .empty-copy { max-width:520px; margin-top:7px; color:var(--muted); font-size:13px; line-height:1.55; }
      .flow-steps { margin-top:18px; display:flex; align-items:center; gap:10px; color:#697387; font-size:11px; font-weight:750; }
      .flow-steps span { background:white; border:1px solid #e0e5ef; border-radius:999px; padding:7px 10px; }
      .conversation-canvas { min-height:270px; max-height:430px; overflow-y:auto; margin-top:14px; padding:18px; border:1px solid #e3e7ee; border-radius:22px; background:#fff; box-shadow:0 14px 38px rgba(28,43,75,.07); }
      .call-heading { text-align:center; color:#667084; font-size:13px; font-weight:750; margin:0 0 16px; }
      .chat-row { display:flex; align-items:flex-start; gap:11px; margin:14px 0; }
      .chat-row.user { flex-direction:row-reverse; }
      .chat-avatar { flex:0 0 34px; width:34px; height:34px; border-radius:50%; display:grid; place-items:center; color:#fff; font-weight:850; background:linear-gradient(145deg,#ff4f65,#ff8b55); }
      .chat-row.assistant .chat-avatar { background:linear-gradient(145deg,#7457f5,#38d5d0); }
      .chat-message { max-width:78%; padding:11px 14px; border-radius:16px; background:#f4f6fa; color:#20283a; line-height:1.55; font-size:14px; overflow-wrap:anywhere; }
      .chat-row.user .chat-message { background:#eef2ff; }
      .chat-author { color:#7d8698; font-size:10px; font-weight:850; letter-spacing:.7px; margin-bottom:4px; }
      .chat-welcome { min-height:190px; display:flex; flex-direction:column; gap:10px; align-items:center; justify-content:center; text-align:center; color:#31394a; }
      .chat-welcome span { color:#858e9e; font-size:13px; }
      .welcome-spark { width:42px; height:42px; border-radius:50%; display:grid; place-items:center; color:white; background:linear-gradient(145deg,#ff4f65,#7457f5,#40d9d2); }
      .voice-reply { display:flex; align-items:center; gap:12px; margin:16px 0 2px 45px; padding:10px 12px; border-radius:16px; background:#f4f8ff; border:1px solid #e0e9f8; }
      .voice-reply span { color:#526078; font-size:11px; font-weight:800; white-space:nowrap; }
      .voice-reply audio { width:100%; height:34px; }
      .voice-note { margin:12px 0 0 45px; color:#8a6570; font-size:11px; }
      @keyframes floatOrb { 0%,100%{transform:translateY(0) scale(1)} 50%{transform:translateY(-8px) scale(1.025)} }
      @keyframes spinHue { 0%{filter:hue-rotate(0deg)} 100%{filter:hue-rotate(22deg)} }
      [data-testid="stChatMessage"] { background:#f8f9fc; border:1px solid #e8ebf2; border-radius:18px; padding:10px 14px; margin:.55rem 0; }
      [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) { background:#f0f7ff; border-color:#dceafb; }
      .stTextArea textarea,[data-testid="stChatInput"],[data-testid="stChatInput"] > div { border-radius:16px !important; border-color:#e1e5ed !important; background:#fff !important; }
      [data-testid="stChatInput"] textarea { color:#18202f !important; caret-color:#18202f !important; }
      [data-testid="stCustomComponentV1"], iframe[title*="streamlit_webrtc"], iframe[src*="streamlit_webrtc"] { height:1px !important; min-height:1px !important; max-height:1px !important; overflow:hidden !important; opacity:0 !important; pointer-events:none !important; margin:0 !important; padding:0 !important; }
      .st-key-voice_call_toggle button { max-width:260px; height:56px; margin:4px auto 18px; display:flex; border:0 !important; border-radius:999px !important; background:#111827 !important; color:white !important; font-size:16px !important; box-shadow:0 14px 28px rgba(17,24,39,.2); }
      .chat-label { color:#6f798c; font-size:11px; font-weight:850; letter-spacing:1.35px; margin:18px 0 8px; }
      [data-testid="stForm"] { border:1px solid #e0e4eb !important; border-radius:26px !important; background:#fff !important; padding:10px 12px !important; box-shadow:0 14px 36px rgba(30,45,80,.09); }
      [data-testid="stForm"] div[data-baseweb="input"], [data-testid="stForm"] div[data-baseweb="base-input"], [data-testid="stForm"] input { border:0 !important; outline:0 !important; box-shadow:none !important; background:#fff !important; color:#18202f !important; }
      [data-testid="stForm"] input::placeholder { color:#8b94a5 !important; opacity:1 !important; }
      .composer-tools { display:flex; align-items:center; gap:17px; min-height:38px; color:#202838; padding-left:5px; font-size:17px; }
      .composer-tools small { color:#98a0af; font-size:10px; margin-left:2px; }
      [data-testid="stForm"] [data-testid="stFormSubmitButton"] button { width:44px !important; min-width:44px !important; height:44px !important; padding:0 !important; border-radius:50% !important; font-size:22px !important; background:#ff4f57 !important; border:0 !important; margin-left:auto !important; }
      [data-testid="stAlert"] { background:#fff !important; color:#263044 !important; border:1px solid #e1e6ee !important; border-left:4px solid #7457f5 !important; border-radius:14px !important; box-shadow:0 10px 28px rgba(30,45,80,.06); }
      [data-testid="stAlert"] * { color:#263044 !important; }
      .stButton button,[data-testid="stBaseButton-primary"] { border-radius:999px; font-weight:750; }
      div[role="radiogroup"] { background:white; border:1px solid var(--line); border-radius:999px; padding:5px 9px; width:fit-content; box-shadow:0 8px 20px rgba(20,30,60,.04); }
      @media(max-width:900px){ .block-container{padding:1rem}.brandbar{align-items:flex-start;gap:12px}.engine-row{justify-content:flex-end}.voice-orb{width:190px;height:190px} div[data-testid="stVerticalBlockBorderWrapper"]{min-height:auto} }
    </style>
    <div class="brandbar">
      <div class="brand"><div class="brandmark">م</div><div><div class="brandname">MasriFlow</div><div class="brandsub">Egyptian voice workspace</div></div></div>
      <div class="engine-row"><span class="engine live">● LOCAL</span><span class="engine">QwenCleo · ASR</span><span class="engine">Qwen3-4B · LLM</span><span class="engine">Hoda · ar-EG voice</span></div>
    </div>
    """, unsafe_allow_html=True,
)
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("transcripts", [])
st.session_state.setdefault("last_reply_audio", b"")
st.session_state.setdefault("tts_error", "")
st.session_state.setdefault("call_opened", False)
_deduplicated_history = []
for _message in st.session_state.chat_history:
    if _deduplicated_history and _message == _deduplicated_history[-1]:
        continue
    _deduplicated_history.append(_message)
st.session_state.chat_history = _deduplicated_history
st.session_state.setdefault(
    "gemini_api_key",
    st.secrets.get("GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY", ""),
)
mode = "Auto-listen"
if mode == "Auto-listen":
    run_live()
else:
    upload = st.file_uploader("Upload WAV audio", type=["wav"])
    recording = st.audio_input("Or record a message")
    media = recording or upload
    if media and st.button("Transcribe and ask", type="primary"):
        with tempfile.TemporaryDirectory(prefix="voice_", dir=TEMP_DIR) as folder:
            path = Path(folder) / "input.wav"
            path.write_bytes(media.getvalue())
            with st.spinner("Transcribing…"):
                text, _, duration = transcribe_asr(path)
            if valid_transcript(text, duration):
                st.session_state.transcripts.append(text)
                with st.spinner("Local Qwen3 is answering…"):
                    add_turn(text)
            else:
                st.error("The recording was too short or unreliable.")
    render_conversation()
