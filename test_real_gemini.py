"""Live Gemini acceptance test using the project-local secret."""

from streamlit.testing.v1 import AppTest


app = AppTest.from_file("transcribe_arabic.py", default_timeout=180)
app.run()
app.radio[0].set_value("Upload / record").run()

app.chat_input[0].set_value(
    "عايز إيميل للمدير بالإنجليزي أقوله إني مش جاي بكرة عشان تعبان"
).run()
app.chat_input[0].set_value(
    "صحح الكلام: أنا مش تعبان، عندي ميعاد مهم وهشتغل من البيت"
).run()

history = app.session_state["chat_history"]
reply = history[-1]["content"]
lower = reply.lower()
assert reply.lower().startswith(("subject:", "dear ")), reply
assert any(phrase in lower for phrase in (
    "work from home", "working from home", "work remotely", "working remotely"
)), reply
assert "sick" not in lower and "cold" not in lower, reply

echo_source = "كرر كلامي: أنا محتاج مساعدة في الشغل"
app.chat_input[0].set_value(echo_source).run()
echo_reply = app.session_state["chat_history"][-1]["content"]
assert echo_reply.strip() != echo_source.strip(), echo_reply

print("REAL_GEMINI_REPLY:", reply)
print("REAL_ANTI_ECHO_REPLY:", echo_reply)
print("PASS")
