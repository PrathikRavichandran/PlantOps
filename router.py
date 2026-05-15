"""
router.py — Classify a user question into (doc_type, equipment) using Claude.
"""
import json
import re
import anthropic
from dotenv import load_dotenv

load_dotenv()

VALID_DOC_TYPES = ["safety_procedures", "maintenance_manual", "quality_control"]
VALID_EQUIPMENT = ["APR", "BDP", "HLX", "NXS", "TM7", "all"]
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a document routing assistant for an industrial plant operations Q&A system.
Classify each question into exactly one doc_type and one equipment code.

Available doc_types:
- safety_procedures: hazards, PPE, permits, emergency response, LOTO, confined space, fire watch, alarms
- maintenance_manual: equipment maintenance, PM schedules, inspections, vibration, lubrication, work orders, reliability
- quality_control: product specifications, QC tests, hold/release procedures, allergen control, HACCP, sampling, lab

Available equipment codes:
- APR: Aurora Petrochemical Refinery (Pasadena TX) — refinery, crude, hydrotreater, reformer
- BDP: Brookhaven Dairy Processing (Madison WI) — dairy, milk, pasteurization, cultured products
- HLX: Helix Pharmaceuticals Building 4 — pharma, cleanroom, sterile manufacturing
- NXS: Nexus Semiconductor Fab 3 — semiconductor, wafer, cleanroom, fab
- TM7: Tide Motors Assembly Plant 7 — automotive, assembly, body shop, paint

Rules:
1. If the question clearly names or implies a single site, use that code.
2. If the question is general, spans multiple sites, or you cannot determine the site, use "all".
3. Pick the doc_type that best matches the subject matter.
4. Respond ONLY with valid JSON, no other text."""


def classify_question(question: str, client: anthropic.Anthropic) -> dict:
    user_prompt = (
        f'Classify this question:\n"{question}"\n\n'
        'Respond with this exact JSON format:\n'
        '{"doc_type": "<safety_procedures|maintenance_manual|quality_control>", '
        '"equipment": "<APR|BDP|HLX|NXS|TM7|all>"}'
    )

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=100,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                raise ValueError("No JSON found in response")

        if result.get("doc_type") not in VALID_DOC_TYPES:
            raise ValueError(f"Invalid doc_type: {result.get('doc_type')}")
        if result.get("equipment") not in VALID_EQUIPMENT:
            raise ValueError(f"Invalid equipment: {result.get('equipment')}")

        return result

    except Exception:
        return {"doc_type": "safety_procedures", "equipment": "all"}
