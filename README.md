# MasriFlow

MasriFlow is a local-first Egyptian Arabic voice collections-agent prototype.

The project is designed to conduct a structured conversation with a customer about an outstanding installment, understand objections, obtain a proposed payment date, confirm it and save the payment promise.

> MasriFlow is currently a research prototype. It is not approved for real customer calls or production financial decisions.

---

## Project Objective

The target is to build an automated agent that can:

- Initiate a collections conversation.
- Communicate in Egyptian Arabic.
- Verify that it is speaking with the correct person.
- Explain the purpose of the call.
- Understand objections and unexpected responses.
- Ask when the customer expects to pay.
- Convert relative expressions into exact dates.
- Confirm the payment date with the customer.
- Save the confirmed promise.
- Escalate unsupported or sensitive cases to a human employee.

---

## System Workflow

```text
Customer speaks
      ↓
Browser speech and silence detection
      ↓
QwenCleo-ASR transcribes Egyptian Arabic
      ↓
Python conversation manager tracks the call stage
      ↓
Qwen3-4B understands the message and drafts a response
      ↓
Python guardrails validate the action and payment date
      ↓
Microsoft Hoda converts the approved response into speech
      ↓
Confirmed payment promise is saved in SQLite
