from rag.rules import check_pm_kisan_eligibility


user_profile = {
    "state": "Bihar",
    "cultivable_land": "yes",
    "institutional_landholder": "no",
    "government_employee": "no",
    "retired_pensioner": "no",
    "monthly_pension": "",
    "paid_income_tax": "yes",
    "registered_professional": "no",
    "nri": "no",
}


result = check_pm_kisan_eligibility(user_profile)

print("\nEligibility:")
print(result["eligibility"])

print("\nReasons:")
for reason in result["reasons"]:
    print("-", reason)

print("\nExclusions:")
for exclusion in result["exclusions"]:
    print("-", exclusion)

print("\nMissing information:")
for item in result["missing_information"]:
    print("-", item)