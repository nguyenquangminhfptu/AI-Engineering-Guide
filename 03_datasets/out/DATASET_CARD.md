# Dataset card — helpdesk SFT v1

- **Built**: 2026-10-15 by `build_sft_dataset.py`
- **Source**: `sample_raw.jsonl` (32 raw rows)
- **Purpose**: teach the model to answer ML-platform helpdesk questions in a practical, step-by-step style
- **Format**: chat `messages` (system / user / assistant), JSONL
- **Splits**: train 20 · eval 5 (split key: `cluster`, eval fraction 0.2)
- **Topics**: {'ft': 9, 'data': 5, 'agent': 5, 'eval': 3, 'serving': 3}

## Processing
Unicode NFC → HTML strip → filters → PII masking (4 replacements) → exact dedup → MinHash near-dup (threshold 0.7).

| Drop reason | Rows |
|---|---|
| exact_duplicate | 3 |
| question_too_short | 1 |
| boilerplate_refusal | 1 |
| empty_field | 1 |
| near_duplicate | 1 |
| **kept** | **25** |

## Known gaps
- Small sample; topic coverage is uneven. Single-turn only.
- License / consent: internal helpdesk export — confirm approval before external use.
