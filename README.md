# Ambient Week 5 — Web2 Boundary Splitter

This project implements **Micro-Challenge #5**:
> Split model output into verifiable vs non-verifiable layers.

It calls Ambient via OpenAI-compatible API and labels response segments as:
- `deterministic` (checkable math / explicit logic),
- `interpretive` (inference, advice, hedging),
- `unknown` (no explicit checkable structure).

## Files

- `ambient_week5_web2.py` — main script
- `week5_web2_report.json` — machine-readable run report
- `week5_web2_summary.md` — human-readable summary

## Requirements

- Python 3.10+
- `openai` package
- Ambient API key

## Setup

### PowerShell (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install openai
$env:AMBIENT_API_KEY="YOUR_NEW_KEY"
