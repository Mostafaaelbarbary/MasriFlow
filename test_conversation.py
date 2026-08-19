"""Smoke tests for conversation continuity and anti-echo behavior."""

from streamlit.testing.v1 import AppTest
import requests


calls = []


class FakeResponse:
    ok = True
    status_code = 200
    text = ""

    def __init__(self, answer):
        self.answer = answer

    def json(self):
        return {"candidates": [{"content": {"parts": [{"text": self.answer}]}}]}


def fake_post(self, url, **kwargs):
    calls.append(kwargs["json"])
    latest = kwargs["json"]["contents"][-1]["parts"][0]["text"]
    system = kwargs["json"]["systemInstruction"]["parts"][0]["text"]
    if "كرر الجملة" in latest:
        if "تم رفضها" in system:
            return FakeResponse("أكيد، تحب أساعدك في إيه بخصوص الشغل؟")
        return FakeResponse(latest)
    if "صحح الكلام" in latest:
        return FakeResponse(
            "تمام، دي المسودة بالإنجليزي:\n\nDear Manager,\n\n"
            "I have an important appointment tomorrow, so I will "
            "be working from home.\n\nBest regards"
        )
    return FakeResponse("Sure, please send me the details.")


requests.Session.post = fake_post


app = AppTest.from_file("transcribe_arabic.py", default_timeout=180)
app.run()
app.radio[0].set_value("Upload / record").run()
app.session_state["gemini_api_key"] = "test-api-key"

# Normal multi-turn request: the second message corrects the first one.
app.chat_input[0].set_value(
    "عايز إيميل للمدير بالإنجليزي أقوله إني مش جاي بكرة عشان تعبان"
).run()
app.chat_input[0].set_value(
    "صحح الكلام: أنا مش تعبان، عندي ميعاد مهم وهشتغل من البيت"
).run()

history = app.session_state["chat_history"]
assert len(history) == 4, history
normal_reply = history[-1]["content"]
lower_reply = normal_reply.lower()
assert "dear" in lower_reply, normal_reply
assert normal_reply.lower().startswith("dear"), normal_reply
assert "work from home" in lower_reply or "working from home" in lower_reply, normal_reply
assert "sick" not in lower_reply and "cold" not in lower_reply, normal_reply

# Deliberately request verbatim repetition. The application must reject an echo.
echo_source = "كرر الجملة دي حرفياً: أنا محتاج مساعدة في الشغل"
app.chat_input[0].set_value(echo_source).run()
echo_reply = app.session_state["chat_history"][-1]["content"]
assert echo_reply.strip() != echo_source.strip(), echo_reply
assert "تحب أساعدك" in echo_reply, echo_reply
assert len(calls) == 4, len(calls)  # 3 turns plus one anti-echo retry
assert calls[1]["contents"][-2]["role"] == "model", calls[1]

print("NORMAL_REPLY:", normal_reply)
print("ANTI_ECHO_REPLY:", echo_reply)
print("PASS")
