"""Score example transactions with the saved prototype models. No dataset needed.

    python prototype/demo.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fraud_scorer import FraudScorer  # noqa: E402

ARTIFACTS = os.path.join(HERE, "..", "results", "prototype")

examples = {
    "TRANSFER that drains the sender, receiver balance unchanged": dict(
        type="TRANSFER", amount=181.0, oldbalanceOrg=181.0, newbalanceOrig=0.0,
        oldbalanceDest=0.0, newbalanceDest=0.0),
    "Ordinary PAYMENT": dict(
        type="PAYMENT", amount=9839.64, oldbalanceOrg=170136.0, newbalanceOrig=160296.36,
        oldbalanceDest=0.0, newbalanceDest=0.0),
}

scorer = FraudScorer.load(ARTIFACTS)  # verifies SHA-256 of every artifact first
for label, tx in examples.items():
    print(f"\n=== {label}")
    print(json.dumps(scorer.score(tx, explain=True), indent=2))
