import os

from dotenv import load_dotenv
from groq import Groq

from rag.retriever import search_scheme
from rag.rules import check_pm_kisan_eligibility


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY is not set. "
        "Add it to backend/.env"
    )

client = Groq(api_key=api_key)


def determine_eligibility(
    user_profile: dict,
    scheme: str,
    language: str = "en"
):
    # --------------------------------------------------
    # 1. RULE ENGINE
    # --------------------------------------------------

    if scheme == "PM-KISAN":
        rule_result = check_pm_kisan_eligibility(
            user_profile
        )

    else:
        return {
            "eligibility": "Cannot determine",
            "answer": (
                "This scheme has not been configured yet."
            ),
            "reasoning": [],
            "exclusions": [],
            "missing_information": [],
            "sources": [],
        }

    # --------------------------------------------------
    # 2. RETRIEVE OFFICIAL GOVERNMENT DOCUMENT
    # --------------------------------------------------

    question = f"""
Determine the official government-document rules
relevant to this PM-KISAN eligibility assessment.

User profile:
{user_profile}

Rule engine result:
{rule_result}

Retrieve evidence concerning:

- landholding farmer eligibility
- cultivable land
- institutional landholders
- government employee exclusions
- pensioner exclusions
- income-tax exclusions
- professional exclusions
- NRI exclusions
"""

    retrieved_results = search_scheme(
        question,
        k=6
    )

    context = "\n\n".join(
        result["content"]
        for result in retrieved_results
    )

    # --------------------------------------------------
    # 3. LANGUAGE
    # --------------------------------------------------

    if language == "hi":
        response_language = """
Respond completely in Hindi.

Use simple Hindi that a rural or small-town
user can easily understand.

Keep official scheme names such as PM-KISAN
in their original form.

Avoid unnecessarily technical or bureaucratic
language.
"""
    else:
        response_language = """
Respond completely in English.

Use simple language that a rural or small-town
user can easily understand.
"""

    # --------------------------------------------------
    # 4. LLM PROMPT
    # --------------------------------------------------

    prompt = f"""
You are a government-scheme eligibility assistant.

{response_language}

Your job is to explain an eligibility assessment.

IMPORTANT:
The rule engine has already performed the basic
logical eligibility checks.

Do NOT override the rule engine.

Use ONLY the official government-document context
provided below when explaining the scheme rules.

========================
USER PROFILE
========================

{user_profile}

========================
RULE ENGINE RESULT
========================

Eligibility:
{rule_result["eligibility"]}

Reasons:
{rule_result["reasons"]}

Exclusions:
{rule_result["exclusions"]}

Missing information:
{rule_result["missing_information"]}

========================
OFFICIAL DOCUMENT CONTEXT
========================

{context}

========================
RESPONSE FORMAT
========================

Eligibility:
{rule_result["eligibility"]}

Reasoning:

For each important requirement or exclusion:

- Requirement/rule:
- User information:
- Assessment:

Explain the result using the official government
document.

Missing information:

List any information that is actually missing.

Next steps:

Give practical next steps based ONLY on the
information available in the official document.

Source:

Mention that the assessment is based on the
retrieved PM-KISAN Operational Guidelines.

IMPORTANT:

- Do not invent eligibility rules.
- Do not change the rule engine's eligibility result.
- Do not claim guaranteed government approval.
- Clearly distinguish between "Likely eligible",
  "Likely not eligible", and "Cannot determine".
"""

    # --------------------------------------------------
    # 5. CALL GROQ
    # --------------------------------------------------

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
    )

    # --------------------------------------------------
    # 6. RETURN RESULT
    # --------------------------------------------------

    return {
        "eligibility": rule_result["eligibility"],
        "answer": response.choices[0].message.content,
        "reasoning": rule_result["reasons"],
        "exclusions": rule_result["exclusions"],
        "missing_information": rule_result[
            "missing_information"
        ],
        "sources": retrieved_results,
    }