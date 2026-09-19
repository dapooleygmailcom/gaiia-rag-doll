# Gaiia RAG Doll Cloud — Up Front 100-Benchmark Scorecard

**Execution Date**: 2026-09-07 03:41:07 UTC  
**Production Endpoint**: `https://8ifjmds7mk.execute-api.ap-southeast-2.amazonaws.com/api/ask`  
**Test Cases Evaluated**: #1 to #100 (Total: 100)  
**Engine / LLM**: AWS Bedrock Nova Pro + DynamoDB Rule Index

## 1. Executive Summary

| Metric | Production Cloud Value | Target / Benchmark |
| :--- | :---: | :---: |
| **Rule Hit Rate** | **86.5%** (64/74) | > 85.0% |
| **Average Rule Recall** | **0.72** | > 0.70 |
| **Average RTT Latency** | **5.63s** | < 6.0s |
| **P50 RTT Latency** | **5.39s** | < 5.0s |
| **P90 RTT Latency** | **7.04s** | < 8.0s |
| **Average Server Exec Time** | **5304.97ms** | < 3500ms |
| **HTTP / Pipeline Errors** | **0** | 0 |

## 2. Latency Profile

| Measurement | Mean | P50 (Median) | P90 | P95 | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Round-Trip Time (RTT)** | 5.63s | 5.39s | 7.04s | 7.79s | 3.52s | 9.3s |
| **Cloud Lambda Exec Time** | 5304.97ms | 5140ms | 6666ms | 7510ms | 3293ms | 8806ms |

## 3. Query Intent Breakdown

| Intent Category | Total Inquiries | Inquiries w/ Rules | Hit Rate (%) | Avg Recall | Avg RTT Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `clarification` | 49 | 34 | **85.3%** | 0.71 | 5.27s |
| `concept` | 4 | 2 | **100.0%** | 1.0 | 6.59s |
| `direct_rule` | 5 | 5 | **80.0%** | 0.66 | 5.34s |
| `errata_comparison` | 8 | 7 | **85.7%** | 0.49 | 6.22s |
| `scenario` | 29 | 22 | **90.9%** | 0.83 | 6.02s |
| `situation` | 5 | 4 | **75.0%** | 0.62 | 5.55s |

## 4. Adjudication Verdict Distribution

| Verdict | Count | Percentage |
| :--- | :---: | :---: |
| `PERMITTED` | 67 | 67.0% |
| `PROHIBITED` | 33 | 33.0% |

## 5. Retrieval Misses Deep Dive

The following **10 questions** had expected rule citations where no strict match was found:

| # | ID | Title | Expected Rules | Retrieved Rules | Verdict |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 14 | `bgg_uf_3589681` | Does ordnance get any advantage fro | `28.46` | `10.45, 14.2, 15.2, 17.1, 17.3` | `PERMITTED` |
| 25 | `bgg_uf_3371642` | Half track malfunction repair | `19.2` | `10.45, 12.13, 14.2, 14.4, 16.4` | `PERMITTED` |
| 34 | `bgg_uf_3206922` | Odd situation - Group behind opposi | `23.7` | `11.11, 13.3, 15.6, 16.5, 17.21` | `PERMITTED` |
| 37 | `bgg_uf_3199417` | Serman tank, for example. | `4.2` | `10.45, 12.1, 14.2, 18.21, 18.4` | `PERMITTED` |
| 54 | `bgg_uf_2984341` | Missing a single card | `20.6` | `10.21, 10.4, 12.13, 13.2, 14.2` | `PROHIBITED` |
| 79 | `bgg_uf_2688478` | Order of resolution of multiple CC | `16.3, 16.5, 20.52, 20.62, 20.72, 4.2` | `11.11, 14.2, 14.4, 14.6, 17.1` | `PERMITTED` |
| 83 | `bgg_uf_2633076` | Wire Cards | `25.8` | `13.3, 13.33, 14.2, 15.2, 16.1` | `PROHIBITED` |
| 89 | `bgg_uf_2479530` | I'm so bad at maths (and english) | `19.11` | `10.45, 12.13, 14.2, 14.4, 18.2` | `PERMITTED` |
| 90 | `bgg_uf_2475533` | Infiltration to an opposing Group ¿ | `20.51` | `12.13, 17.1, 17.3, 17.6, 17.63` | `PROHIBITED` |
| 98 | `bgg_uf_2324650` | Basic game - two questions | `10.12` | `10.4, 16.5, 17.1, 17.3, 17.6` | `PROHIBITED` |

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
| 61 | `bgg_uf_2928709` | `scenario` | `PROHIBITED` | 34.3, 8.52 | 8.52 | 0.50 | 5.14s | 4789ms |
| 62 | `bgg_uf_2910183` | `scenario` | `PERMITTED` | 42.6, 42.7, 42.72 | 42.6, 42.7, 42.72 | 1.00 | 9.30s | 8806ms |
| 63 | `bgg_uf_2909876` | `situation` | `PROHIBITED` | 20.21, 20.52 | 20.21 | 0.50 | 5.14s | 4778ms |
| 64 | `bgg_uf_2909528` | `clarification` | `PERMITTED` | 27.1 | 27.1 | 1.00 | 4.28s | 3944ms |
| 65 | `bgg_uf_2865400` | `direct_rule` | `PROHIBITED` | 20.24, 20.25, 7.2, 8.2, 8.22 | 20.24, 20.25, 8.2, 8.22 | 0.80 | 5.19s | 4577ms |
| 66 | `bgg_uf_2859449` | `clarification` | `PERMITTED` | - | - | - | 5.27s | 4915ms |
| 67 | `bgg_uf_2855553` | `scenario` | `PROHIBITED` | - | - | - | 5.23s | 4300ms |
| 68 | `bgg_uf_2838588` | `errata_comparison` | `PROHIBITED` | 8.4 | 8.4 | 1.00 | 8.28s | 7921ms |
| 69 | `bgg_uf_2834181` | `errata_comparison` | `PROHIBITED` | 25.34, 28.61, 44.3, 50.3, 9.5 | 44.3 | 0.20 | 5.94s | 5672ms |
| 70 | `bgg_uf_2828147` | `clarification` | `PERMITTED` | 46.4 | 46.4 | 1.00 | 5.58s | 5170ms |
| 71 | `bgg_uf_2826539` | `clarification` | `PERMITTED` | 20.3, 20.52 | 20.3 | 0.50 | 5.06s | 4713ms |
| 72 | `bgg_uf_2817909` | `scenario` | `PERMITTED` | 45.422, 45.43, 5.6, 5.61 | 45.422, 45.43 | 0.50 | 8.19s | 7769ms |
| 73 | `bgg_uf_2797691` | `clarification` | `PROHIBITED` | - | - | - | 6.55s | 6178ms |
| 74 | `bgg_uf_2795341` | `clarification` | `PERMITTED` | 22.5, 28.7 | 22.5, 28.7 | 1.00 | 5.96s | 5705ms |
| 75 | `bgg_uf_2740282` | `clarification` | `PERMITTED` | - | - | - | 5.02s | 4699ms |
| 76 | `bgg_uf_2737943` | `scenario` | `PERMITTED` | 16.4 | 16.4 | 1.00 | 5.55s | 5263ms |
| 77 | `bgg_uf_2736919` | `clarification` | `PERMITTED` | - | - | - | 5.74s | 5424ms |
| 78 | `bgg_uf_2690087` | `errata_comparison` | `PERMITTED` | 1.5, 10.45, 17.4, 22.1, 33.2, 7.5 | 10.45, 17.4 | 0.33 | 5.47s | 5157ms |
| 79 | `bgg_uf_2688478` | `situation` | `PERMITTED` | 16.3, 16.5, 20.52, 20.62, 20.72, 4.2 | ❌ | 0.00 | 5.12s | 4763ms |
| 80 | `bgg_uf_2661857` | `direct_rule` | `PROHIBITED` | 6.5 | 6.5 | 1.00 | 6.10s | 5826ms |
| 81 | `bgg_uf_2651429` | `clarification` | `PERMITTED` | 3.6 | 3.6 | 1.00 | 4.65s | 4303ms |
| 82 | `bgg_uf_2646142` | `scenario` | `PERMITTED` | - | - | - | 6.98s | 6756ms |
| 83 | `bgg_uf_2633076` | `clarification` | `PROHIBITED` | 25.8 | ❌ | 0.00 | 6.17s | 5889ms |
| 84 | `bgg_uf_2632337` | `clarification` | `PERMITTED` | - | - | - | 3.86s | 3680ms |
| 85 | `bgg_uf_2543271` | `concept` | `PERMITTED` | 19.2 | 19.2 | 1.00 | 4.96s | 4604ms |
| 86 | `bgg_uf_2543063` | `clarification` | `PROHIBITED` | 20.21, 20.5, 20.51, 20.52, 20.53 | 20.21, 20.5, 20.53 | 0.60 | 4.90s | 4700ms |
| 87 | `bgg_uf_2536120` | `scenario` | `PERMITTED` | 17.3, 17.6, 5.6, 5.6.2, 5.61 | 17.6, 5.6, 5.6.2, 5.61 | 0.80 | 4.66s | 4307ms |
| 88 | `bgg_uf_2483572` | `errata_comparison` | `PERMITTED` | - | - | - | 5.10s | 4776ms |
| 89 | `bgg_uf_2479530` | `errata_comparison` | `PERMITTED` | 19.11 | ❌ | 0.00 | 6.40s | 6033ms |
| 90 | `bgg_uf_2475533` | `direct_rule` | `PROHIBITED` | 20.51 | ❌ | 0.00 | 5.43s | 5179ms |
| 91 | `bgg_uf_2460694` | `clarification` | `PERMITTED` | - | - | - | 4.95s | 4623ms |
| 92 | `bgg_uf_2447913` | `direct_rule` | `PERMITTED` | 19.11, 19.12, 19.13, 2.67 | 19.11, 19.13 | 0.50 | 4.88s | 4526ms |
| 93 | `bgg_uf_2433348` | `clarification` | `PERMITTED` | 20.39, 20.51, 20.73 | 20.39 | 0.33 | 7.73s | 7408ms |
| 94 | `bgg_uf_2431930` | `clarification` | `PROHIBITED` | 33.1 | 33.1 | 1.00 | 5.38s | 5061ms |
| 95 | `bgg_uf_2416386` | `scenario` | `PROHIBITED` | 24.7, 45.423, 6.52 | 24.7, 45.423, 6.52 | 1.00 | 4.86s | 4474ms |
| 96 | `bgg_uf_2390951` | `clarification` | `PERMITTED` | - | - | - | 3.77s | 3483ms |
| 97 | `bgg_uf_2376834` | `scenario` | `PERMITTED` | - | - | - | 8.06s | 7792ms |
| 98 | `bgg_uf_2324650` | `scenario` | `PROHIBITED` | 10.12 | ❌ | 0.00 | 6.63s | 6256ms |
| 99 | `bgg_uf_2324200` | `concept` | `PERMITTED` | - | - | - | 7.79s | 7510ms |
| 100 | `bgg_uf_2321437` | `scenario` | `PROHIBITED` | - | - | - | 5.39s | 5163ms |