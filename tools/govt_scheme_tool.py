"""
KisanAgent — Government Scheme Tool
======================================
Real-world analog: Karnataka Raitha Seva Kendra
government scheme portal.

Tracks PM-KISAN, drip irrigation subsidy,
crop insurance (PMFBY), and Tomato Support schemes.
Deadlines are hard — missing them loses real money.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("kisanagent.tools.govt_scheme")

# Canonical Karnataka 2024-25 scheme definitions
SCHEME_DEFINITIONS = [
    {
        "name": "PM-KISAN Input Supplement",
        "benefit_inr": 2000,
        "eligibility": "Smallholder farmers (<5 acres), Kolar district",
        "deadline_day": 30,
        "required_documents": ["Aadhaar card", "Land records (RTC)", "Bank passbook"],
        "application_url": "https://pmkisan.gov.in/BeneficiaryStatus.aspx",
        "type": "direct_benefit_transfer",
    },
    {
        "name": "Crop Insurance PMFBY",
        "benefit_inr": 3000,
        "eligibility": "All notified crop farmers — premium ₹500 deducted",
        "deadline_day": 20,
        "required_documents": [
            "Aadhaar card", "Land records (RTC)",
            "Crop sowing certificate", "Bank account"
        ],
        "application_url": "https://pmfby.gov.in",
        "type": "insurance",
    },
    {
        "name": "Drip Irrigation Subsidy",
        "benefit_inr": 5000,
        "eligibility": "Farmers with drip/sprinkler system installed, Karnataka",
        "deadline_day": 50,
        "required_documents": [
            "Drip system purchase invoice",
            "Aadhaar card", "Land records", "Caste certificate"
        ],
        "application_url": "https://raitamitra.karnataka.gov.in/micro_irrigation",
        "type": "capital_subsidy",
    },
    {
        "name": "Tomato Special Support Price",
        "benefit_inr": 1500,
        "eligibility": "Kolar / Chikkaballapur district tomato growers",
        "deadline_day": 70,
        "required_documents": [
            "Mandi sale receipt", "Aadhaar card",
            "Bank account linked to Aadhaar"
        ],
        "application_url": "https://krishimela.karnataka.gov.in/tomato-support",
        "type": "price_support",
    },
]


class GovtSchemeTool:
    """
    Karnataka Raitha Seva Kendra scheme portal analog.

    Exposes active schemes, deadlines, and eligibility.
    Records applied schemes to prevent double-dipping.
    """

    CLOSING_SOON_THRESHOLD = 5  # days before deadline

    def __init__(
        self,
        scheme_schedule: Optional[List[Dict[str, Any]]] = None,
        captured_schemes_ref: Optional[List[str]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.scheme_schedule = scheme_schedule or SCHEME_DEFINITIONS
        self.applied_schemes = captured_schemes_ref if captured_schemes_ref is not None else []
        self.rng = rng or np.random.default_rng(33)

    def call(
        self,
        state: str = "Karnataka",
        crop: str = "tomato",
        current_day: int = 0,
    ) -> Dict[str, Any]:
        """
        Return all active government schemes for Karnataka tomato farmers.

        Returns:
            {
              "active_schemes": [...],
              "already_applied": list[str],
              "total_available_benefit_inr": int,
              "advisory": str
            }
        """
        active_schemes = []
        total_available = 0

        for scheme in self.scheme_schedule:
            deadline_day = scheme["deadline_day"]
            days_remaining = deadline_day - current_day
            already_applied = scheme["name"] in self.applied_schemes

            if days_remaining < 0:
                status = "expired"
            elif days_remaining <= self.CLOSING_SOON_THRESHOLD:
                status = "closing_soon"
            else:
                status = "open"

            scheme_info = {
                "name": scheme["name"],
                "benefit_inr": scheme["benefit_inr"],
                "eligibility": scheme["eligibility"],
                "deadline_day": deadline_day,
                "days_remaining": max(0, days_remaining),
                "application_url": scheme["application_url"],
                "required_documents": scheme["required_documents"],
                "type": scheme.get("type", "benefit"),
                "status": status,
                "already_applied": already_applied,
            }
            active_schemes.append(scheme_info)

            if not already_applied and status != "expired":
                total_available += scheme["benefit_inr"]

        # Sort by deadline urgency
        active_schemes.sort(key=lambda s: s["days_remaining"])

        advisory = _scheme_advisory(active_schemes, current_day)

        return {
            "active_schemes": active_schemes,
            "already_applied": list(self.applied_schemes),
            "total_available_benefit_inr": total_available,
            "state": state,
            "crop": crop,
            "advisory": advisory,
        }


def _scheme_advisory(schemes: List[Dict[str, Any]], day: int) -> str:
    urgent = [s for s in schemes if s["status"] == "closing_soon" and not s["already_applied"]]
    open_schemes = [s for s in schemes if s["status"] == "open" and not s["already_applied"]]

    if urgent:
        names = ", ".join(s["name"] for s in urgent)
        total = sum(s["benefit_inr"] for s in urgent)
        return f"URGENT: {names} closing within 5 days. Apply now — ₹{total:,} at stake."
    elif open_schemes:
        names = ", ".join(s["name"] for s in open_schemes[:2])
        total = sum(s["benefit_inr"] for s in open_schemes)
        return f"Available: {names}. Total unclaimed benefit: ₹{total:,}."
    return "All available schemes applied. No action needed."
