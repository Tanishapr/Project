from rag.retriever import search_scheme


user_profile = {
    "state": "Bihar",
    "occupation": "Farmer",
    "landholding": "2 acres",
    "annual_income": "12000",
}


question = f"""
Find all government-document information relevant to determining
whether this person is eligible for PM-KISAN.

User information:
{user_profile}

Focus on eligibility requirements, landholding requirements,
income-related conditions, income-tax exclusions, government
employee exclusions, pensioner exclusions, and other exclusion
categories.
"""


results = search_scheme(question, k=6)


for i, result in enumerate(results, start=1):
    print(f"\n{'=' * 60}")
    print(f"RESULT {i}")
    print(f"{'=' * 60}")
    print(result["content"])