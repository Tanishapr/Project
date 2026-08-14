def check_pm_kisan_eligibility(user_profile: dict):
    reasons = []
    exclusions = []
    missing_information = []

    # 1. Cultivable land
    cultivable_land = user_profile.get("cultivable_land")

    if cultivable_land == "yes":
        reasons.append(
            "User has stated that they own cultivable agricultural land."
        )
    elif cultivable_land == "no":
        exclusions.append(
            "User does not have cultivable agricultural land."
        )
    else:
        missing_information.append(
            "Whether the user owns cultivable agricultural land."
        )

    # 2. Institutional landholder
    institutional_landholder = user_profile.get(
        "institutional_landholder"
    )

    if institutional_landholder == "yes":
        exclusions.append(
            "User has identified themselves as an institutional landholder."
        )
    elif institutional_landholder == "no":
        reasons.append(
            "User is not an institutional landholder."
        )
    else:
        missing_information.append(
            "Whether the user is an institutional landholder."
        )

    # 3. Government employee
    government_employee = user_profile.get(
        "government_employee"
    )

    if government_employee == "yes":
        exclusions.append(
            "User is a government employee and may fall under "
            "the government-employee exclusion category."
        )
    elif government_employee == "no":
        reasons.append(
            "User is not a government employee."
        )
    else:
        missing_information.append(
            "Whether the user is a government employee."
        )

    # 4. Retired pensioner
    retired_pensioner = user_profile.get(
        "retired_pensioner"
    )

    if retired_pensioner == "yes":

        pension = user_profile.get("monthly_pension")

        if pension:
            try:
                pension_amount = float(pension)

                if pension_amount >= 10000:
                    exclusions.append(
                        "User receives a monthly pension of "
                        "₹10,000 or more."
                    )
                else:
                    reasons.append(
                        "User's monthly pension is below ₹10,000."
                    )

            except ValueError:
                missing_information.append(
                    "A valid monthly pension amount."
                )

        else:
            missing_information.append(
                "Monthly pension amount."
            )

    elif retired_pensioner == "no":
        reasons.append(
            "User is not a retired/superannuated pensioner."
        )

    else:
        missing_information.append(
            "Whether the user is a retired/superannuated pensioner."
        )

    # 5. Income tax
    paid_income_tax = user_profile.get(
        "paid_income_tax"
    )

    if paid_income_tax == "yes":
        exclusions.append(
            "User paid income tax in the last assessment year."
        )
    elif paid_income_tax == "no":
        reasons.append(
            "User did not pay income tax in the last assessment year."
        )
    else:
        missing_information.append(
            "Whether the user paid income tax in the last assessment year."
        )

    # 6. Registered professional
    registered_professional = user_profile.get(
        "registered_professional"
    )

    if registered_professional == "yes":
        exclusions.append(
            "User is a registered professional and may fall "
            "under the professional exclusion category."
        )
    elif registered_professional == "no":
        reasons.append(
            "User is not a registered professional."
        )
    else:
        missing_information.append(
            "Whether the user is a registered professional."
        )

    # 7. NRI
    nri = user_profile.get("nri")

    if nri == "yes":
        exclusions.append(
            "User is an NRI and may be excluded under the scheme."
        )
    elif nri == "no":
        reasons.append(
            "User is not an NRI."
        )
    else:
        missing_information.append(
            "Whether the user is an NRI."
        )

    # Final decision
    if exclusions:
        eligibility = "Likely not eligible"

    elif missing_information:
        eligibility = "Cannot determine"

    else:
        eligibility = "Likely eligible"

    return {
        "eligibility": eligibility,
        "reasons": reasons,
        "exclusions": exclusions,
        "missing_information": missing_information,
    }