# Gaiia RAG Doll Cloud — Up Front 100-Benchmark Scorecard

**Execution Date**: 2026-09-07 03:33:06 UTC  
**Production Endpoint**: `https://8ifjmds7mk.execute-api.ap-southeast-2.amazonaws.com/api/ask`  
**Test Cases Evaluated**: #1 to #60 (Total: 60)  
**Engine / LLM**: AWS Bedrock Nova Pro + DynamoDB Rule Index

## 1. Executive Summary

| Metric | Production Cloud Value | Target / Benchmark |
| :--- | :---: | :---: |
| **Rule Hit Rate** | **89.4%** (42/47) | > 85.0% |
| **Average Rule Recall** | **0.79** | > 0.70 |
| **Average RTT Latency** | **5.55s** | < 6.0s |
| **P50 RTT Latency** | **5.46s** | < 5.0s |
| **P90 RTT Latency** | **6.9s** | < 8.0s |
| **Average Server Exec Time** | **5226.75ms** | < 3500ms |
| **HTTP / Pipeline Errors** | **0** | 0 |

## 2. Latency Profile

| Measurement | Mean | P50 (Median) | P90 | P95 | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Round-Trip Time (RTT)** | 5.55s | 5.46s | 6.9s | 7.22s | 3.52s | 7.39s |
| **Cloud Lambda Exec Time** | 5226.75ms | 5140ms | 6403ms | 6666ms | 3293ms | 6987ms |

## 3. Query Intent Breakdown

| Intent Category | Total Inquiries | Inquiries w/ Rules | Hit Rate (%) | Avg Recall | Avg RTT Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `clarification` | 33 | 25 | **84.0%** | 0.71 | 5.26s |
| `concept` | 2 | 1 | **100.0%** | 1.0 | 6.81s |
| `direct_rule` | 1 | 1 | **100.0%** | 1.0 | 5.11s |
| `errata_comparison` | 3 | 3 | **100.0%** | 0.64 | 6.19s |
| `scenario` | 18 | 15 | **93.3%** | 0.9 | 5.81s |
| `situation` | 3 | 2 | **100.0%** | 1.0 | 5.83s |

## 4. Adjudication Verdict Distribution

| Verdict | Count | Percentage |
| :--- | :---: | :---: |
| `PERMITTED` | 42 | 70.0% |
| `PROHIBITED` | 18 | 30.0% |

## 5. Retrieval Misses Deep Dive

The following **5 questions** had expected rule citations where no strict match was found:

| # | ID | Title | Expected Rules | Retrieved Rules | Verdict |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 14 | `bgg_uf_3589681` | Does ordnance get any advantage fro | `28.46` | `10.45, 14.2, 15.2, 17.1, 17.3` | `PERMITTED` |
| 25 | `bgg_uf_3371642` | Half track malfunction repair | `19.2` | `10.45, 12.13, 14.2, 14.4, 16.4` | `PERMITTED` |
| 34 | `bgg_uf_3206922` | Odd situation - Group behind opposi | `23.7` | `11.11, 13.3, 15.6, 16.5, 17.21` | `PERMITTED` |
| 37 | `bgg_uf_3199417` | Serman tank, for example. | `4.2` | `10.45, 12.1, 14.2, 18.21, 18.4` | `PERMITTED` |
| 54 | `bgg_uf_2984341` | Missing a single card | `20.6` | `10.21, 10.4, 12.13, 13.2, 14.2` | `PROHIBITED` |

## 6. Itemized Test Results (First 100 Cases)

| # | ID | Intent | Verdict | Expected Rules | Hits | Recall | Latency | Exec (ms) |
| :---: | :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | `bgg_uf_3720028` | `scenario` | `PERMITTED` | - | - | - | 7.39s | 6287ms |
| 2 | `bgg_uf_3713707` | `scenario` | `PROHIBITED` | 17.7 | 17.7 | 1.00 | 6.90s | 6533ms |
| 3 | `bgg_uf_3679123` | `scenario` | `PROHIBITED` | 16.42, 32.12 | 16.42, 32.12 | 1.00 | 7.39s | 6987ms |
| 4 | `bgg_uf_3675113` | `clarification` | `PERMITTED` | 17.1 | 17.1 | 1.00 | 5.57s | 5284ms |
| 5 | `bgg_uf_3645496` | `clarification` | `PERMITTED` | 25.4, 26.5, 28.3, 28.43, 28.431, 28.45, 6.5 | 26.5, 28.3, 28.43, 28.431, 28.45, 6.5 | 0.86 | 6.23s | 5984ms |
| 6 | `bgg_uf_3645331` | `clarification` | `PROHIBITED` | 41.7 | 41.7 | 1.00 | 4.97s | 4637ms |
| 7 | `bgg_uf_3629431` | `clarification` | `PERMITTED` | 17.3, 17.4 | 17.3, 17.4 | 1.00 | 6.19s | 5859ms |
| 8 | `bgg_uf_3623562` | `clarification` | `PERMITTED` | 20.39, 20.73, 20.9, 20.91, 45.4, 45.422 | 20.73, 45.4, 45.422 | 0.50 | 5.94s | 5594ms |
| 9 | `bgg_uf_3623559` | `clarification` | `PERMITTED` | 15.2, 4.1, 4.5 | 15.2, 4.1, 4.5 | 1.00 | 5.79s | 5516ms |
| 10 | `bgg_uf_3616976` | `clarification` | `PERMITTED` | 14.5, 7.3, 7.32 | 14.5, 7.3, 7.32 | 1.00 | 6.79s | 6403ms |
| 11 | `bgg_uf_3610189` | `scenario` | `PERMITTED` | 2.22, 41.3 | 2.22, 41.3 | 1.00 | 5.00s | 4772ms |
| 12 | `bgg_uf_3610186` | `clarification` | `PERMITTED` | 17.2, 17.3, 41.4, 41.54, 41.57 | 17.2, 17.3, 41.4, 41.54, 41.57 | 1.00 | 6.25s | 6000ms |
| 13 | `bgg_uf_3595997` | `scenario` | `PERMITTED` | 13.1, 13.2, 13.29, 25.8, 7.321 | 13.1, 13.2, 13.29, 25.8, 7.321 | 1.00 | 5.06s | 4655ms |
| 14 | `bgg_uf_3589681` | `clarification` | `PERMITTED` | 28.46 | ❌ | 0.00 | 4.63s | 4397ms |
| 15 | `bgg_uf_3573599` | `clarification` | `PERMITTED` | 11.12, 11.13, 17.0, 18.2, 3.3 | 11.12, 18.2, 3.3 | 0.60 | 4.93s | 4561ms |
| 16 | `bgg_uf_3488991` | `concept` | `PERMITTED` | - | - | - | 6.40s | 6177ms |
| 17 | `bgg_uf_3487478` | `clarification` | `PERMITTED` | - | - | - | 5.23s | 4973ms |
| 18 | `bgg_uf_3483939` | `direct_rule` | `PERMITTED` | 45.4, 45.43 | 45.4, 45.43 | 1.00 | 5.11s | 4880ms |
| 19 | `bgg_uf_3482507` | `scenario` | `PROHIBITED` | 16.4, 16.4.2, 16.42, 16.421, 16.422, 16.423 | 16.4, 16.4.2, 16.42, 16.421, 16.422, 16.423 | 1.00 | 6.38s | 6087ms |
| 20 | `bgg_uf_3479566` | `clarification` | `PROHIBITED` | 17.7, 17.8, 17.9 | 17.7, 17.8 | 0.67 | 4.14s | 3782ms |
| 21 | `bgg_uf_3436068` | `scenario` | `PERMITTED` | 17.6, 17.613 | 17.6, 17.613 | 1.00 | 4.78s | 4493ms |
| 22 | `bgg_uf_3430569` | `clarification` | `PERMITTED` | - | - | - | 5.10s | 4845ms |
| 23 | `bgg_uf_3393252` | `scenario` | `PERMITTED` | 11.13, 18.1 | 11.13, 18.1 | 1.00 | 4.84s | 4599ms |
| 24 | `bgg_uf_3377439` | `clarification` | `PERMITTED` | 13.1, 33.1 | 33.1 | 0.50 | 6.37s | 6013ms |
| 25 | `bgg_uf_3371642` | `scenario` | `PERMITTED` | 19.2 | ❌ | 0.00 | 5.15s | 4809ms |
| 26 | `bgg_uf_3332374` | `scenario` | `PERMITTED` | - | - | - | 5.51s | 5144ms |
| 27 | `bgg_uf_3276875` | `scenario` | `PROHIBITED` | 3.2 | 3.2 | 1.00 | 5.57s | 5252ms |
| 28 | `bgg_uf_3243425` | `scenario` | `PROHIBITED` | 15.6, 16.5 | 15.6, 16.5 | 1.00 | 5.82s | 5581ms |
| 29 | `bgg_uf_3224790` | `scenario` | `PERMITTED` | 38.3 | 38.3 | 1.00 | 5.29s | 4928ms |
| 30 | `bgg_uf_3215988` | `scenario` | `PROHIBITED` | 16.422 | 16.422 | 1.00 | 5.13s | 4717ms |
| 31 | `bgg_uf_3215900` | `errata_comparison` | `PERMITTED` | 11.1, 19.11, 19.13 | 19.11, 19.13 | 0.67 | 5.31s | 5070ms |
| 32 | `bgg_uf_3215100` | `scenario` | `PERMITTED` | 16.1, 2.22 | 16.1 | 0.50 | 7.04s | 6666ms |
| 33 | `bgg_uf_3212008` | `scenario` | `PERMITTED` | 5.6 | 5.6 | 1.00 | 5.83s | 5503ms |
| 34 | `bgg_uf_3206922` | `clarification` | `PERMITTED` | 23.7 | ❌ | 0.00 | 5.87s | 5616ms |
| 35 | `bgg_uf_3205614` | `situation` | `PERMITTED` | 18.1, 20.8, 4.3 | 18.1, 20.8, 4.3 | 1.00 | 5.46s | 5140ms |
| 36 | `bgg_uf_3203445` | `clarification` | `PERMITTED` | - | - | - | 3.72s | 3448ms |
| 37 | `bgg_uf_3199417` | `clarification` | `PERMITTED` | 4.2 | ❌ | 0.00 | 5.18s | 4811ms |
| 38 | `bgg_uf_3175698` | `clarification` | `PERMITTED` | - | - | - | 5.96s | 5737ms |
| 39 | `bgg_uf_3175363` | `errata_comparison` | `PROHIBITED` | 17.6, 17.62, 47.2, 47.3 | 17.6, 17.62 | 0.50 | 6.25s | 5932ms |
| 40 | `bgg_uf_3172033` | `clarification` | `PERMITTED` | 20.2, 20.3 | 20.2, 20.3 | 1.00 | 4.36s | 3968ms |
| 41 | `bgg_uf_3167913` | `clarification` | `PERMITTED` | 25.3, 25.6 | 25.3, 25.6 | 1.00 | 4.93s | 4609ms |
| 42 | `bgg_uf_3164229` | `clarification` | `PERMITTED` | 14.6 | 14.6 | 1.00 | 4.24s | 3891ms |
| 43 | `bgg_uf_3159799` | `clarification` | `PERMITTED` | - | - | - | 5.05s | 4819ms |
| 44 | `bgg_uf_3153137` | `clarification` | `PROHIBITED` | 25.3, 28.32 | 25.3, 28.32 | 1.00 | 3.52s | 3293ms |
| 45 | `bgg_uf_3144329` | `clarification` | `PROHIBITED` | - | - | - | 5.24s | 4902ms |
| 46 | `bgg_uf_3129211` | `situation` | `PERMITTED` | 25.43, 25.6 | 25.43, 25.6 | 1.00 | 6.07s | 5849ms |
| 47 | `bgg_uf_3094327` | `errata_comparison` | `PROHIBITED` | 17.6, 20.5, 29.2, 29.5 | 17.6, 20.5, 29.5 | 0.75 | 7.00s | 6590ms |
| 48 | `bgg_uf_3094195` | `concept` | `PROHIBITED` | 20.71 | 20.71 | 1.00 | 7.22s | 6835ms |
| 49 | `bgg_uf_3090589` | `clarification` | `PERMITTED` | 19.4, 20.6, 20.74 | 19.4, 20.6, 20.74 | 1.00 | 6.33s | 6017ms |
| 50 | `bgg_uf_3073277` | `scenario` | `PROHIBITED` | 17.6, 5.51, 5.6 | 17.6, 5.51, 5.6 | 1.00 | 6.27s | 5929ms |
| 51 | `bgg_uf_3072777` | `clarification` | `PROHIBITED` | 5.51 | 5.51 | 1.00 | 5.77s | 5416ms |
| 52 | `bgg_uf_3063831` | `clarification` | `PERMITTED` | 10.12, 7.3 | 10.12, 7.3 | 1.00 | 4.46s | 4237ms |
| 53 | `bgg_uf_2988827` | `scenario` | `PROHIBITED` | - | - | - | 5.20s | 4863ms |
| 54 | `bgg_uf_2984341` | `clarification` | `PROHIBITED` | 20.6 | ❌ | 0.00 | 4.25s | 4068ms |
| 55 | `bgg_uf_2973875` | `clarification` | `PROHIBITED` | - | - | - | 5.10s | 4896ms |
| 56 | `bgg_uf_2952275` | `clarification` | `PERMITTED` | 28.46, 28.51, 28.7 | 28.46 | 0.33 | 6.63s | 6324ms |
| 57 | `bgg_uf_2940194` | `clarification` | `PERMITTED` | - | - | - | 4.73s | 4403ms |
| 58 | `bgg_uf_2933911` | `situation` | `PERMITTED` | - | - | - | 5.97s | 5623ms |
| 59 | `bgg_uf_2932122` | `clarification` | `PERMITTED` | 5.1, 8.11 | 8.11 | 0.50 | 5.21s | 4920ms |
| 60 | `bgg_uf_2929984` | `clarification` | `PERMITTED` | 13.1, 13.25, 7.2 | 13.25, 7.2 | 0.67 | 4.77s | 4481ms |