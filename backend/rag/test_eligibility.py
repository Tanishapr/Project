from rag.eligibility import determine_eligibility


user_profile = {
    "state": "Bihar",
    "occupation": "Farmer",
    "landholding": "2 acres",
    "annual_income": "120000",
}


result = determine_eligibility(
    user_profile,
    "PM-KISAN",
)

print("\n===== ELIGIBILITY RESULT =====")
print(result["answer"])

print("\n===== RETRIEVED SOURCES =====")

for source in result["sources"]:
    print("\n---")
    print(source["content"])