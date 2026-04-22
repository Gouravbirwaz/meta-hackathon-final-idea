"""
KisanAgent — Credit Tool
==========================
Real-world analog: Kisan Credit Card (KCC) and
NABARD microfinance portal.

Reflects actual rural credit access constraints in Karnataka.
  - KCC max: ₹25,000 for 2-acre farmer
  - Subsidised rate: 7% per annum
  - Repayment: 180 days
  - Eligibility: no existing KCC debt, balance ≥ ₹2,000
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("kisanagent.tools.credit")

ALTERNATIVE_OPTIONS = [
    "Self Help Group (SHG) microfinance loan — ₹10,000 at 12% p.a.",
    "NABARD Joint Liability Group loan — ₹15,000 at 10% p.a.",
    "Cooperative bank agri-loan — ₹20,000 at 9% p.a. (14-day processing)",
    "Input dealer credit — fertiliser/pesticide on 60-day credit",
]


class CreditTool:
    """
    KCC (Kisan Credit Card) and NABARD microfinance portal analog.

    Approval rules:
      ✓ Max loan: ₹25,000 (KCC 2-acre limit)
      ✗ Rejected if existing KCC loan unpaid
      ✗ Rejected if bank_balance < ₹2,000
      ✓ Interest: 7% p.a. (KCC subsidised rate)
      ✓ Repayment: 180 days
    """

    KCC_MAX_LOAN_INR = 25_000
    KCC_INTEREST_ANNUAL = 0.07
    KCC_REPAYMENT_DAYS = 180
    MIN_BALANCE_FOR_LOAN = 2_000

    def __init__(
        self,
        farmer_profile: Dict[str, Any],
        simulator_ref: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.farmer_profile = farmer_profile
        self.simulator = simulator_ref
        self.rng = rng or np.random.default_rng(22)
        self.active_loans: List[Dict[str, Any]] = []

    def call(
        self,
        amount_inr: int = 10_000,
        purpose: str = "farming_input",
    ) -> Dict[str, Any]:
        """
        Check loan eligibility and return KCC credit assessment.

        Returns:
            {
              "approved": bool,
              "amount_approved": int,
              "interest_rate_annual": float,
              "repayment_days": int,
              "monthly_emi": float,
              "total_repayable": float,
              "rejection_reason": str | None,
              "alternative_options": list[str],
              "credit_score": int,   # 300-900 CIBIL proxy
              "kcc_limit_remaining": int
            }
        """
        # Get current state from simulator ref
        bank_balance = 15_000.0
        active_debt = 0.0
        if self.simulator is not None:
            bank_balance = getattr(self.simulator, "bank_balance_inr", 15_000.0)
            active_debt = getattr(self.simulator, "active_debt_inr", 0.0)

        # Also check active_loans tracked here
        unpaid_loans = [l for l in self.active_loans if not l.get("repaid", False)]
        has_existing_loan = len(unpaid_loans) > 0 or active_debt > 0

        # Compute CIBIL-proxy credit score (300-900)
        base_score = 650
        if bank_balance > 10_000:
            base_score += 50
        if has_existing_loan:
            base_score -= 150
        credit_score = int(np.clip(
            base_score + int(self.rng.normal(0, 30)), 300, 900
        ))

        # Approval checks
        rejection_reason = None
        approved = True
        amount_approved = 0

        if has_existing_loan:
            approved = False
            rejection_reason = (
                f"Existing KCC loan of ₹{active_debt:,.0f} unpaid. "
                "Repay before applying for new credit."
            )
        elif bank_balance < self.MIN_BALANCE_FOR_LOAN:
            approved = False
            rejection_reason = (
                f"Bank balance ₹{bank_balance:,.0f} below minimum ₹{self.MIN_BALANCE_FOR_LOAN:,}. "
                "Insufficient creditworthiness."
            )
        elif credit_score < 400:
            approved = False
            rejection_reason = f"Credit score {credit_score} too low for KCC (minimum 400)."
        else:
            amount_approved = min(amount_inr, self.KCC_MAX_LOAN_INR)

        # Compute repayment terms
        total_repayable = 0.0
        monthly_emi = 0.0
        if approved and amount_approved > 0:
            interest = amount_approved * self.KCC_INTEREST_ANNUAL * (self.KCC_REPAYMENT_DAYS / 365)
            total_repayable = round(amount_approved + interest, 2)
            monthly_emi = round(total_repayable / (self.KCC_REPAYMENT_DAYS / 30), 2)

            # Log approved loan
            self.active_loans.append({
                "amount": amount_approved,
                "total_repayable": total_repayable,
                "purpose": purpose,
                "repaid": False,
            })

        kcc_limit_remaining = max(0, self.KCC_MAX_LOAN_INR - amount_approved) if approved else self.KCC_MAX_LOAN_INR

        return {
            "approved": approved,
            "amount_approved": amount_approved,
            "interest_rate_annual": self.KCC_INTEREST_ANNUAL,
            "repayment_days": self.KCC_REPAYMENT_DAYS,
            "monthly_emi": monthly_emi,
            "total_repayable": total_repayable,
            "rejection_reason": rejection_reason,
            "alternative_options": ALTERNATIVE_OPTIONS if not approved else [],
            "credit_score": credit_score,
            "kcc_limit_remaining": kcc_limit_remaining,
            "purpose": purpose,
        }
