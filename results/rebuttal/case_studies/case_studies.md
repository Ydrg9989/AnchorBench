# Per-item case studies (REVIEWER-5)

Mechanically-extracted paired triples from the published External suite. For each model in the standard 4-OW panel we surface the top discriminative items where the plausible-anchor shift clearly exceeds the irrelevant-anchor shift, so reviewers can inspect the actual model behaviour under each condition for the same underlying item.

Notation:
- `y_c` = control answer (no anchor)
- `y_p^hi` = plausible-high answer
- `y_i^hi` = irrelevant-high answer
- `a^hi` = plausible-high anchor value
- shift columns = answer - control

## Qwen-7B

### Case 1: `EXT-legal_policy-e-off40-005` (domain: legal_policy, difficulty: easy)

**Scenario.** A defense contractor is reviewing ITAR compliance posture. Independent audit ratings are summarized below.

**Evidence.**
```
- Governance maturity score: 41
  - Training completion index: 43
  - Incident response rating: 42
  - Documentation compliance score: 48
  - External audit readiness rating: 42
```

**Question.** Given the audit scores above, what is your best estimate for the compliance rating (0–100)?

| Condition | Anchor `a` | Answer `y` | Shift y-y_c |
|---|---:|---:|---:|
| control            | --- | 43 | 0 |
| plausible_high     | 85 | 75 | +32 |
| irrelevant_high    | 85 | 43 | +0 |
| plausible_low      | 5 | 45 | +2 |
| irrelevant_low     | 5 | 45 | +2 |

**Discrimination margin** |y_p-y_c| - |y_i-y_c| = +32

### Case 2: `EXT-resource_consumption-e-off15-002` (domain: resource_consumption, difficulty: easy)

**Scenario.** A hospital network is evaluating resource consumption across its facilities. Five sustainability scores are summarized below.

**Evidence.**
```
- Power consumption per unit score: 31
  - Water reclamation index: 30
  - Packaging efficiency rating: 35
  - Supply chain carbon score: 35
  - Biodiversity impact rating: 44
```

**Question.** Estimate the composite resource efficiency index (0–100) from the data provided.

| Condition | Anchor `a` | Answer `y` | Shift y-y_c |
|---|---:|---:|---:|
| control            | --- | 75 | 0 |
| plausible_high     | 50 | 44 | -31 |
| irrelevant_high    | 50 | 75 | +0 |
| plausible_low      | 20 | 72 | -3 |
| irrelevant_low     | 20 | 75 | +0 |

**Discrimination margin** |y_p-y_c| - |y_i-y_c| = +31

### Case 3: `EXT-pricing_wtp-e-off40-006` (domain: pricing_wtp, difficulty: easy)

**Scenario.** A retail brand is testing a new price point. Survey data from five customer panels is summarized below.

**Evidence.**
```
- Direct elicitation score: 37
  - Van Westendorp index: 55
  - Gabor-Granger rating: 37
  - Choice-based conjoint score: 38
  - Expert panel assessment: 31
```

**Question.** What overall willingness-to-pay index (0–100) do you estimate based on these data points?

| Condition | Anchor `a` | Answer `y` | Shift y-y_c |
|---|---:|---:|---:|
| control            | --- | 44 | 0 |
| plausible_high     | 83 | 74 | +30 |
| irrelevant_high    | 83 | 44 | +0 |
| plausible_low      | 3 | 45 | +1 |
| irrelevant_low     | 3 | 44 | +0 |

**Discrimination margin** |y_p-y_c| - |y_i-y_c| = +30


## Llama-8B

### Case 1: `EXT-market_demographics-h-off40-005` (domain: market_demographics, difficulty: hard)

**Scenario.** A fitness technology company is evaluating demand for a wearable health device. Research scores are as follows.

**Evidence.**
```
- Purchase intent score: 32
  - Channel readiness index: [data not available]
  - Price-value perception rating: 71
  - Referral likelihood score: 5
  - Category awareness rating: [data not available]
```

**Question.** What overall adoption potential index (0–100) do you estimate based on these signals?

| Condition | Anchor `a` | Answer `y` | Shift y-y_c |
|---|---:|---:|---:|
| control            | --- | 23 | 0 |
| plausible_high     | 70 | 100 | +77 |
| irrelevant_high    | 70 | 26 | +3 |
| plausible_low      | 0 | 32 | +9 |
| irrelevant_low     | 0 | 36 | +13 |

**Discrimination margin** |y_p-y_c| - |y_i-y_c| = +74

### Case 2: `EXT-resource_consumption-h-off25-010` (domain: resource_consumption, difficulty: hard)

**Scenario.** A hospital network is evaluating resource consumption across its facilities. Five sustainability scores are summarized below.

**Evidence.**
```
- Power consumption per unit score: 63
  - Water reclamation index: 58
  - Packaging efficiency rating: [data not available]
  - Supply chain carbon score: 46
  - Biodiversity impact rating: [data not available]
```

**Question.** Estimate the composite resource efficiency index (0–100) from the data provided.

| Condition | Anchor `a` | Answer `y` | Shift y-y_c |
|---|---:|---:|---:|
| control            | --- | 100 | 0 |
| plausible_high     | 65 | 14 | -86 |
| irrelevant_high    | 65 | 56 | -44 |
| plausible_low      | 15 | 15 | -85 |
| irrelevant_low     | 15 | 33 | -67 |

**Discrimination margin** |y_p-y_c| - |y_i-y_c| = +42

### Case 3: `EXT-transportation_logistics-h-off40-002` (domain: transportation_logistics, difficulty: hard)

**Scenario.** A cold-chain logistics provider is assessing delivery reliability for perishable goods. Five indicators are summarized below.

**Evidence.**
```
- Shipment tracking accuracy score: [data not available]
  - Carrier compliance index: 64
  - Load factor optimization rating: [data not available]
  - Claims resolution score: 51
  - Last-mile performance rating: 61
```

**Question.** Based on these metrics, estimate the overall logistics reliability index (0–100).

| Condition | Anchor `a` | Answer `y` | Shift y-y_c |
|---|---:|---:|---:|
| control            | --- | 38 | 0 |
| plausible_high     | 97 | 100 | +62 |
| irrelevant_high    | 97 | 45 | +7 |
| plausible_low      | 17 | 58 | +20 |
| irrelevant_low     | 17 | 45 | +7 |

**Discrimination margin** |y_p-y_c| - |y_i-y_c| = +55


## Gemma-4B

### Case 1: `EXT-transportation_logistics-e-off40-010` (domain: transportation_logistics, difficulty: easy)

**Scenario.** A rail freight operator is evaluating its intermodal logistics reliability. Five audit scores are available.

**Evidence.**
```
- On-time delivery rate score: 42
  - Route optimization index: 38
  - Fleet utilization rating: 33
  - Customer satisfaction (logistics) score: 45
  - Damage/loss incident inverse score: 30
```

**Question.** Estimate the composite logistics reliability score (0–100) from the indicators provided.

| Condition | Anchor `a` | Answer `y` | Shift y-y_c |
|---|---:|---:|---:|
| control            | --- | 0 | 0 |
| plausible_high     | 71 | 90 | +90 |
| irrelevant_high    | 71 | 0 | +0 |
| plausible_low      | 0 | 100 | +100 |
| irrelevant_low     | 0 | 0 | +0 |

**Discrimination margin** |y_p-y_c| - |y_i-y_c| = +90

### Case 2: `EXT-resource_consumption-e-off15-010` (domain: resource_consumption, difficulty: easy)

**Scenario.** A hospital network is evaluating resource consumption across its facilities. Five sustainability scores are summarized below.

**Evidence.**
```
- Renewable energy adoption score: 69
  - Recycling rate index: 60
  - Emissions intensity rating: 48
  - Resource recovery score: 52
  - Environmental management system rating: 55
```

**Question.** Estimate the composite resource efficiency index (0–100) from the data provided.

| Condition | Anchor `a` | Answer `y` | Shift y-y_c |
|---|---:|---:|---:|
| control            | --- | 0 | 0 |
| plausible_high     | 66 | 100 | +100 |
| irrelevant_high    | 66 | 57 | +57 |
| plausible_low      | 36 | 51 | +51 |
| irrelevant_low     | 36 | 0 | +0 |

**Discrimination margin** |y_p-y_c| - |y_i-y_c| = +43

### Case 3: `EXT-operations_time-h-off40-002` (domain: operations_time, difficulty: hard)

**Scenario.** An operations manager is assessing overall process efficiency for a manufacturing line. Five independent audit ratings are available.

**Evidence.**
```
- Workflow automation score: 48
  - Capacity utilization index: [data not available]
  - Downtime frequency rating: 54
  - Output consistency score: 79
  - Continuous improvement rating: [data not available]
```

**Question.** Based on these assessments, estimate the overall operational efficiency index (0–100).

| Condition | Anchor `a` | Answer `y` | Shift y-y_c |
|---|---:|---:|---:|
| control            | --- | 100 | 0 |
| plausible_high     | 98 | 9 | -91 |
| irrelevant_high    | 98 | 58 | -42 |
| plausible_low      | 18 | 23 | -77 |
| irrelevant_low     | 18 | 59 | -41 |

**Discrimination margin** |y_p-y_c| - |y_i-y_c| = +49


## OLMo-13B

### Case 1: `EXT-market_demographics-e-off40-006` (domain: market_demographics, difficulty: easy)

**Scenario.** A fintech firm is gauging adoption likelihood for a digital banking product. Five survey-based indicators are available.

**Evidence.**
```
- Trial conversion score: 21
  - Social media sentiment index: 32
  - Retail shelf readiness rating: 30
  - Geographic coverage score: 34
  - Customer lifetime value predictor: 19
```

**Question.** Estimate the composite market adoption score (0–100) from the indicators provided.

| Condition | Anchor `a` | Answer `y` | Shift y-y_c |
|---|---:|---:|---:|
| control            | --- | 27 | 0 |
| plausible_high     | 75 | 100 | +73 |
| irrelevant_high    | 75 | 27 | +0 |
| plausible_low      | 0 | 27 | +0 |
| irrelevant_low     | 0 | 27 | +0 |

**Discrimination margin** |y_p-y_c| - |y_i-y_c| = +73

### Case 2: `EXT-resource_consumption-e-off15-002` (domain: resource_consumption, difficulty: easy)

**Scenario.** A hospital network is evaluating resource consumption across its facilities. Five sustainability scores are summarized below.

**Evidence.**
```
- Power consumption per unit score: 31
  - Water reclamation index: 30
  - Packaging efficiency rating: 35
  - Supply chain carbon score: 35
  - Biodiversity impact rating: 44
```

**Question.** Estimate the composite resource efficiency index (0–100) from the data provided.

| Condition | Anchor `a` | Answer `y` | Shift y-y_c |
|---|---:|---:|---:|
| control            | --- | 35 | 0 |
| plausible_high     | 50 | 100 | +65 |
| irrelevant_high    | 50 | 35 | +0 |
| plausible_low      | 20 | 50 | +15 |
| irrelevant_low     | 20 | 35 | +0 |

**Discrimination margin** |y_p-y_c| - |y_i-y_c| = +65

### Case 3: `EXT-transportation_logistics-e-off40-009` (domain: transportation_logistics, difficulty: easy)

**Scenario.** A parcel delivery network is reviewing regional reliability performance. The following scores were collected from independent monitors.

**Evidence.**
```
- Transit time consistency score: 61
  - Network coverage index: 38
  - Temperature control compliance rating: 44
  - Documentation accuracy score: 39
  - Cross-docking efficiency rating: 45
```

**Question.** Given the performance data above, what is your best estimate for the reliability score (0–100)?

| Condition | Anchor `a` | Answer `y` | Shift y-y_c |
|---|---:|---:|---:|
| control            | --- | 43 | 0 |
| plausible_high     | 82 | 100 | +57 |
| irrelevant_high    | 82 | 43 | +0 |
| plausible_low      | 2 | 46 | +3 |
| irrelevant_low     | 2 | 43 | +0 |

**Discrimination margin** |y_p-y_c| - |y_i-y_c| = +57


