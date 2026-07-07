# Section 6.4 Aviation Rule Relation Balanced v13 150 Rule-Library Comparison

The first table reports raw LLM-generated rule-library quality. The second table evaluates downstream replacement under a controlled setting where strong-aligned valid rules are supplied and rule compilation is handled by the backend. The backend first solves the linearized feasible region and then applies a local nonlinear polish for residual nonlinear constraints.

## Table 1. Rule-library generation quality

| Domain | Generator | Rules | Provenance valid | Constraint grounding | Relation grounding |
| --- | --- | --- | --- | --- | --- |
| Aviation rule relation balanced v13 150 | Qwen-plus | 111 | 100.0% | 100.0% | 100.0% |
| Aviation rule relation balanced v13 150 | DeepSeek-Pro | 142 | 100.0% | 99.1% | 99.6% |
| Aviation rule relation balanced v13 150 | Xiaomi MIMO | 186 | 100.0% | 100.0% | 100.0% |

## Table 2. Downstream replacement performance

| Domain | Generator | Rule Precision | Rule Recall | Formal CSR | Sem-CSR | False accept | Candidate zero | CTHR no valid | Invalid cases | Unsupported tasks | Relation templates added |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aviation rule relation balanced v13 150 | Qwen-plus | 100.0% | 100.0% | 96.7% | 96.7% | 0.0% | 0 | 0 | 5/150 | 0 | 616 |
| Aviation rule relation balanced v13 150 | DeepSeek-Pro | 86.8% | 100.0% | 96.7% | 96.7% | 0.0% | 0 | 0 | 5/150 | 0 | 616 |
| Aviation rule relation balanced v13 150 | Xiaomi MIMO | 88.9% | 100.0% | 94.0% | 94.0% | 0.0% | 0 | 0 | 9/150 | 0 | 616 |
