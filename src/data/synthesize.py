"""
Schema-accurate synthetic CFPB-style complaint generator.

Used only as a fallback when the real CFPB Consumer Complaint Database
can't be reached (see acquire.py). Mirrors the real dataset's columns,
its messy multi-year "Product" taxonomy (which src/data/preprocess.py
later collapses down to a clean label set — the real project's job),
its realistic class imbalance, its XXXX-style PII redaction, and
injects a small amount of deliberate label noise, because pretending
real-world text data is clean would defeat the point of a portfolio
project about handling class imbalance and label noise.

This is template + vocabulary-bank driven, not an LLM call — it's meant
to be fast, seed-reproducible, and dependency-free.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Canonical (collapsed) label set — 8 categories, realistically imbalanced.
#    Proportions loosely track real CFPB Product-level skew (credit
#    reporting dominates; vehicle loans / money transfer are long-tail).
# ---------------------------------------------------------------------------
CANONICAL_CATEGORIES = {
    "Credit reporting or other personal consumer reports": 0.42,
    "Debt collection": 0.16,
    "Credit card or prepaid card": 0.12,
    "Checking or savings account": 0.10,
    "Mortgage": 0.08,
    "Student loan": 0.05,
    "Vehicle loan or lease": 0.04,
    "Money transfer, virtual currency, or money service": 0.03,
}
assert abs(sum(CANONICAL_CATEGORIES.values()) - 1.0) < 1e-9

# ---------------------------------------------------------------------------
# 2. Raw "Product" strings — the messy, multi-year taxonomy a real CFPB
#    pull would contain (CFPB renamed / merged several product labels
#    between 2015-2024). Each canonical category maps from 1-3 raw
#    variants. preprocess.py's LABEL_MAP does the collapsing.
# ---------------------------------------------------------------------------
RAW_VARIANTS = {
    "Credit reporting or other personal consumer reports": [
        "Credit reporting, credit repair services, or other personal consumer reports",
        "Credit reporting",
    ],
    "Debt collection": ["Debt collection"],
    "Credit card or prepaid card": [
        "Credit card or prepaid card",
        "Credit card",
        "Prepaid card",
    ],
    "Checking or savings account": [
        "Checking or savings account",
        "Bank account or service",
    ],
    "Mortgage": ["Mortgage"],
    "Student loan": ["Student loan"],
    "Vehicle loan or lease": [
        "Vehicle loan or lease",
        "Consumer Loan",
    ],
    "Money transfer, virtual currency, or money service": [
        "Money transfer, virtual currency, or money service",
        "Money transfers",
        "Virtual currency",
    ],
}

SUB_PRODUCTS = {
    "Credit reporting or other personal consumer reports": ["Credit reporting", "Credit repair services"],
    "Debt collection": ["Credit card debt", "Medical debt", "Auto debt", "Other debt"],
    "Credit card or prepaid card": ["General-purpose credit card", "Store credit card", "General-purpose prepaid card"],
    "Checking or savings account": ["Checking account", "Savings account", "CD account"],
    "Mortgage": ["Conventional home mortgage", "FHA mortgage", "Home equity loan or line of credit"],
    "Student loan": ["Federal student loan servicing", "Private student loan"],
    "Vehicle loan or lease": ["Loan", "Lease"],
    "Money transfer, virtual currency, or money service": ["Domestic money transfer", "International money transfer", "Virtual currency"],
}

ISSUES = {
    "Credit reporting or other personal consumer reports": [
        "Incorrect information on your report",
        "Problem with a credit reporting company's investigation",
        "Improper use of your report",
        "Unable to get your credit report or credit score",
    ],
    "Debt collection": [
        "Attempts to collect debt not owed",
        "Written notification about debt",
        "Communication tactics",
        "False statements or representation",
    ],
    "Credit card or prepaid card": [
        "Problem with a purchase shown on your statement",
        "Fees or interest",
        "Closing your account",
        "Getting a credit card",
    ],
    "Checking or savings account": [
        "Managing an account",
        "Problem caused by your funds being low",
        "Opening an account",
        "Closing an account",
    ],
    "Mortgage": [
        "Trouble during payment process",
        "Applying for a mortgage or refinancing an existing mortgage",
        "Struggling to pay mortgage",
    ],
    "Student loan": [
        "Dealing with your lender or servicer",
        "Struggling to repay your loan",
        "Getting a loan",
    ],
    "Vehicle loan or lease": [
        "Managing the loan or lease",
        "Problems at the end of the loan or lease",
        "Struggling to pay your loan",
    ],
    "Money transfer, virtual currency, or money service": [
        "Fraud or scam",
        "Money was not available when promised",
        "Wrong amount charged or received",
    ],
}

COMPANIES = [
    "Acme National Bank", "Meridian Credit Union", "Sterling Financial Services",
    "Horizon Card Services", "Pinnacle Mortgage Group", "Summit Auto Finance",
    "Beacon Debt Recovery LLC", "Northgate Consumer Reporting", "Fairview Student Lending",
    "Crestline Money Transfer Co", "Bluewave Bank", "Redwood Servicing Corp",
]

STATES = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI", "AZ", "WA", "MA", "VA", "NJ"]
SUBMITTED_VIA = ["Web", "Phone", "Referral", "Postal mail", "Fax"]
COMPANY_RESPONSE = [
    "Closed with explanation", "Closed with non-monetary relief",
    "Closed with monetary relief", "Closed", "In progress",
]

# ---------------------------------------------------------------------------
# 3. Narrative templates, per category, with slot-fillable entities.
#    A shared "noise bank" of generic sentences is mixed in ~35% of the
#    time regardless of category, because in the real data a lot of
#    narratives are generic ("nobody called me back") and don't carry
#    much category signal — this is what makes the classification task
#    non-trivial instead of a keyword-matching exercise.
# ---------------------------------------------------------------------------
TEMPLATES = {
    "Credit reporting or other personal consumer reports": [
        "I pulled my credit report on {date} and found an account from {company} that I do not recognize, "
        "account number ending in {digits}. I have disputed this {n} times with no resolution.",
        "{company} is reporting a late payment on my account ending in {digits} that was actually paid on time. "
        "I sent a dispute letter on {date} and never received a written response.",
        "My credit report shows a collections account from {company} for {amount} that was already paid off in "
        "{year}. This is affecting my ability to get approved for a loan.",
    ],
    "Debt collection": [
        "{company} has been calling me multiple times a day about a debt of {amount} that I do not believe is mine. "
        "I asked for debt validation on {date} and never received anything in writing.",
        "I am being contacted by {company} regarding a medical bill of {amount} from {year} that I already settled. "
        "They are threatening to report it to the credit bureaus.",
        "A collector from {company} called my workplace after I told them not to, in violation of the FDCPA. "
        "This happened on {date}.",
    ],
    "Credit card or prepaid card": [
        "I noticed a charge of {amount} on my {company} card statement from {date} that I did not authorize. "
        "I called to dispute it and was told it would take {n} business days to investigate.",
        "{company} closed my credit card account without notice after {n} years, and I lost my rewards balance. "
        "This happened around {date}.",
        "I was charged an annual fee of {amount} by {company} that was never disclosed when I opened the account "
        "on {date}.",
    ],
    "Checking or savings account": [
        "{company} charged me {amount} in overdraft fees on {date} even though I had opted out of overdraft "
        "coverage. I have called customer service {n} times about this.",
        "My savings account with {company} was frozen on {date} without explanation, and I could not access "
        "{amount} for over a week.",
        "I opened a checking account with {company} and was charged a monthly maintenance fee of {amount} "
        "that contradicts what I was told in the branch.",
    ],
    "Mortgage": [
        "{company} has been applying my mortgage payments incorrectly since {date}, and now shows me as {n} "
        "months behind when I have proof of payment for {amount} each month.",
        "I applied for a mortgage modification with {company} on {date} and have not received a decision "
        "after {n} months, during which they continued charging late fees.",
        "{company} force-placed insurance on my home costing {amount} even though I had my own policy active "
        "as of {date}.",
    ],
    "Student loan": [
        "{company} is my student loan servicer and has not applied my payments toward the correct loans since "
        "{date}, resulting in {amount} in extra interest.",
        "I submitted an income-driven repayment recertification to {company} on {date} and my payment jumped "
        "to {amount} without explanation.",
        "{company} reported my loan as delinquent even though I was in an approved forbearance starting {date}.",
    ],
    "Vehicle loan or lease": [
        "{company} repossessed my vehicle on {date} despite my payment of {amount} clearing two days earlier.",
        "At the end of my lease with {company}, I was charged {amount} in fees that were never mentioned in my "
        "original lease agreement signed in {year}.",
        "{company} continues to report my auto loan as past due after I paid it in full on {date}.",
    ],
    "Money transfer, virtual currency, or money service": [
        "I sent {amount} through {company} on {date} and the recipient never received the funds. Support has "
        "not responded in {n} days.",
        "{company} charged me a hidden fee of {amount} on a transfer I made on {date} that was not disclosed "
        "up front.",
        "My {company} account was frozen after a transfer of {amount} on {date}, and I have not been able to "
        "reach anyone to explain why.",
    ],
}

NOISE_SENTENCES = [
    "I have called customer service {n} times over the past {weeks} weeks and keep getting transferred with no resolution.",
    "This has caused significant stress and I am asking the CFPB to intervene, reference #{ref}.",
    "I have attached documentation supporting my complaint, including {n} emails from the company.",
    "I would like this resolved and my account corrected as soon as possible.",
    "No one from the company has followed up with me despite {n} promises to do so over {weeks} weeks.",
    "I feel like I am being ignored and this is not the first time I've had issues with this company.",
    "I first noticed this problem about {weeks} weeks ago and it still has not been fixed.",
    "I spoke with a representative named XXXX who told me it would be resolved within {n} business days; it was not.",
    "I filed a complaint with my state attorney general's office as well, case #{ref}.",
    "This is affecting my ability to {consequence}.",
    "I was told to wait {weeks} weeks for a callback that never came.",
    "I have been a customer for {years} years and have never experienced anything like this before.",
]

CONSEQUENCES = [
    "get approved for a car loan", "refinance my home", "rent an apartment",
    "get a new credit card", "cosign for my child's loan", "qualify for a lower interest rate",
]

# Generic, cross-category dispute language. Real CFPB complaints are often
# this vague — "there's a wrong charge / this company won't fix it / I
# want it corrected" without a distinguishing noun in sight. Mixing these
# in (see AMBIGUOUS_MIX_RATE) keeps the classification task honest instead
# of degenerating into keyword matching, and is what makes the baseline
# floor meaningfully beatable rather than already saturated.
AMBIGUOUS_TEMPLATES = [
    "{company} has an incorrect charge of {amount} on my account from {date} and refuses to correct it.",
    "I have been disputing an issue with {company} since {date} involving {amount} and have gotten nowhere "
    "after {n} phone calls.",
    "{company} will not provide me with a clear explanation of a {amount} charge that appeared on {date}.",
    "I contacted {company} about a problem with my account on {date} and was given conflicting information "
    "by every representative I spoke to.",
    "There is an error on my account with {company} that has been unresolved since {date}, involving {amount}.",
    "{company} keeps sending me conflicting statements about my account, and I don't know if I owe {amount} "
    "or if it's already been resolved.",
    "I believe {company} is not handling my account correctly and it has cost me {amount} as of {date}.",
    "My account with {company} shows an issue dated {date} for {amount} that customer service could not explain.",
]
AMBIGUOUS_MIX_RATE = 0.25


def _random_date(rng: random.Random) -> str:
    start = date(2022, 1, 1)
    d = start + timedelta(days=rng.randint(0, 900))
    return f"XX/XX/{d.year}"


def _random_amount(rng: random.Random) -> str:
    dollars = rng.randint(15, 9850)
    cents = rng.choice([0, 25, 50, 75, 99])
    return f"${dollars:,}.{cents:02d}"


def _fill(template: str, rng: random.Random) -> str:
    s = template.format(
        date=_random_date(rng),
        company="XXXX",  # company name redacted in narrative text, like real CFPB data
        digits="XXXX",
        n=rng.randint(2, 14),
        amount=_random_amount(rng),
        year=rng.choice([2022, 2023, 2024, 2025]),
        weeks=rng.randint(1, 26),
        years=rng.randint(1, 22),
        ref=rng.randint(100000, 999999),
        consequence=rng.choice(CONSEQUENCES),
    )
    return s


def _make_narrative(category: str, rng: random.Random) -> str:
    if rng.random() < AMBIGUOUS_MIX_RATE:
        template = rng.choice(AMBIGUOUS_TEMPLATES)
    else:
        template = rng.choice(TEMPLATES[category])
    sentence = _fill(template, rng)
    parts = [sentence]
    # 1-3 generic filler sentences appended (realistic overlap noise), sampled
    # without replacement so repeated narratives need to collide on every slot
    # across every sentence, not just one.
    k = rng.choice([0, 1, 1, 2, 2, 3])
    if k:
        chosen = rng.sample(NOISE_SENTENCES, k=min(k, len(NOISE_SENTENCES)))
        for noise in chosen:
            parts.append(_fill(noise, rng))
    return " ".join(parts)


def generate_dataset(n_rows: int = 120_000, seed: int = 42, label_noise_rate: float = 0.05) -> pd.DataFrame:
    """Generate a schema-accurate synthetic CFPB-style complaint dataset.

    Columns mirror the real CFPB Consumer Complaint Database export:
    Date received, Product, Sub-product, Issue, Consumer complaint narrative,
    Company, State, Submitted via, Company response to consumer,
    Timely response?, Consumer disputed?, Complaint ID.

    `label_noise_rate` fraction of rows get their *raw Product* field
    swapped to a plausible-but-wrong variant post-hoc, mimicking the
    real dataset's inconsistent taxonomy application across years —
    the thing Section 4.1 of the project scope calls out as real, not
    theoretical.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    categories = list(CANONICAL_CATEGORIES.keys())
    weights = list(CANONICAL_CATEGORIES.values())
    canonical_choices = np_rng.choice(categories, size=n_rows, p=weights)

    rows = []
    start_date = date(2022, 1, 1)
    for i, canon in enumerate(canonical_choices):
        raw_product = rng.choice(RAW_VARIANTS[canon])
        sub_product = rng.choice(SUB_PRODUCTS[canon])
        issue = rng.choice(ISSUES[canon])
        narrative = _make_narrative(canon, rng)
        received = start_date + timedelta(days=rng.randint(0, 900))

        rows.append(
            {
                "Complaint ID": 1_000_000 + i,
                "Date received": received.isoformat(),
                "Product": raw_product,  # raw, messy taxonomy — collapsed in preprocess.py
                "Sub-product": sub_product,
                "Issue": issue,
                "Consumer complaint narrative": narrative,
                "Company": rng.choice(COMPANIES),
                "State": rng.choice(STATES),
                "Submitted via": rng.choice(SUBMITTED_VIA),
                "Company response to consumer": rng.choice(COMPANY_RESPONSE),
                "Timely response?": rng.choices(["Yes", "No"], weights=[0.96, 0.04])[0],
                "Consumer disputed?": rng.choices(["Yes", "No", "N/A"], weights=[0.15, 0.35, 0.50])[0],
                "_canonical_label": canon,  # ground-truth for synthetic generation only;
                # a real CFPB pull would NOT have this column — preprocess.py derives
                # the working label from Product/Sub-product the same way for both.
            }
        )

    df = pd.DataFrame(rows)

    # Inject label noise directly into the raw Product field for a subset of rows,
    # so the collapsed label derived from it is genuinely sometimes wrong —
    # this is what a class-imbalance-aware, noise-tolerant training loop has to
    # actually contend with, not a cosmetic detail.
    n_noisy = int(len(df) * label_noise_rate)
    noisy_idx = np_rng.choice(df.index, size=n_noisy, replace=False)
    all_raw = [v for variants in RAW_VARIANTS.values() for v in variants]
    for idx in noisy_idx:
        df.loc[idx, "Product"] = rng.choice(all_raw)

    df = df.drop(columns=["_canonical_label"])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=120_000)
    p.add_argument("--out", default="data/raw/complaints.csv")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    df = generate_dataset(args.n, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
