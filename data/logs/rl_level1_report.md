# Gaiia Prime Radiant -- Level 1 RAG-Doll Retrieval Report

**Date:** 2026-07-17 12:30:35
**Duration:** 4015.4s
**Total Tests:** 11

---

## Gate Result: GATE FAILED

| Metric | Value |
|---|---|
| Pass Rate (EXACT + FAITHFUL) | **81.8%** |
| Gate Threshold | 95.0% |
| Gate Status | BLOCKED -- fix corpus gaps before implementation |
| Errata Priority Tests | All errata tests passed |

---

## Score Breakdown

| Grade | Count | % |
|---|---|---|
| EXACT | 6 | 55% |
| FAITHFUL | 3 | 27% |
| CITED | 2 | 18% |
| FAIL | 0 | 0% |

---

## By Game System

| System | Total | EXACT | FAITHFUL | CITED | FAIL | Pass% |
|---|---|---|---|---|---|---|
| CENTURION | 4 | 2 | 1 | 1 | 0 | 75% |
| INTERCEPTOR | 4 | 1 | 2 | 1 | 0 | 75% |
| LEVIATHAN | 2 | 2 | 0 | 0 | 0 | 100% |
| PREFECT | 1 | 1 | 0 | 0 | 0 | 100% |

---

## Individual Results

| ID | System | Diff | Grade | Has Citation | Reason |
|---|---|---|---|---|---|
| C-R-01 | CENTURION | EASY | FAITHFUL | YES | The AI's answer is correct but uses slightly different wording and expands on th |
| C-R-02 | CENTURION | MEDIUM | CITED | YES | The AI correctly states that it cannot provide an answer based on the given text |
| C-R-03 | CENTURION | EASY | EXACT | YES | The AI's answer precisely matches the reference answer, includes all required ke |
| C-R-04 | CENTURION | HARD | EXACT | YES | The AI's answer is exact to the reference answer, providing all necessary inform |
| I-R-01 | INTERCEPTOR | EASY | EXACT | YES | The AI's answer is an exact match to the reference answer, including all require |
| I-R-02 | INTERCEPTOR | MEDIUM | FAITHFUL | YES | The answer is correct but uses slightly different wording, particularly in descr |
| I-R-03 | INTERCEPTOR | HARD | CITED | NO | The AI's answer does not match the reference answer and lacks specific keywords. |
| I-R-04 | INTERCEPTOR | EASY | FAITHFUL | YES | The AI's answer is correct and includes all required keywords, but it adds speci |
| L-R-01 | LEVIATHAN | MEDIUM | EXACT | YES | The AI's answer matches the reference answer precisely, includes all required ke |
| L-R-02 | LEVIATHAN | HARD | EXACT | YES | The AI's answer precisely matches the reference answer and correctly cites the L |
| P-R-01 | PREFECT | EASY | EXACT | YES | The AI's answer is an exact match to the reference answer, includes all required |

---

## Failures and Corpus Gaps

### C-R-02 -- MEDIUM (CENTURION)
**Query:** What happens when a Centurion unit becomes suppressed?

**Grade:** CITED -- The AI correctly states that it cannot provide an answer based on the given text, but there is a lack of information from the sources to accurately determine the effects of suppression on a Centurion unit. The response does not include required keywords and lacks the exact phrasing or content needed for EXACT/FAITHFUL grading.

**Required Keywords:** suppression, suppressed, fire

**Keywords Found:** suppression

---

### I-R-03 -- HARD (INTERCEPTOR)
**Query:** What does it mean when an Interceptor ship is Seriously Out of Control (SOC)?

**Grade:** CITED -- The AI's answer does not match the reference answer and lacks specific keywords. The AI mentions 'seriously out of control' but does not provide an accurate definition or include all required keywords.

**Required Keywords:** seriously out of control, damage, maneuvering functions

**Keywords Found:** seriously out of control

---


*Level 1 gate: FAILED -- resolve corpus gaps first*
