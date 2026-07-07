# Section 6.3 Sensitivity Summary Table

| Experiment Block | Setting | Candidate/Ref | Filtered/Ref | Predicted/Ref | Rule Precision | Rule Recall | Exact Match | Sem-CSR | Takeaway |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Default deterministic baseline | threshold 8.0, no LLM | 9.608 | 1.301 | 1.129 | 0.936 | 0.975 | 0.850 | 0.967 | Strong recovery without LLM filtering |
| Candidate threshold grid | threshold 6 | 10.733 | 1.301 | 1.121 | 0.936 | 0.967 | 0.833 | 0.950 | Wider candidates do not improve recovery |
| Candidate threshold grid | threshold 7-8 | 10.156-9.608 | 1.301 | 1.129 | 0.936 | 0.975 | 0.850 | 0.967 | Stable near-default plateau |
| Candidate threshold grid | threshold 9-10 | 8.934-7.611 | 1.281-1.274 | 1.109-1.102 | 0.941 | 0.963-0.956 | 0.833-0.817 | 0.950 | Higher threshold trades recall for slight precision gain |
| Task scoring weights | +/-20% one-factor | 9.522-9.608 | 1.298-1.301 | 1.126-1.129 | 0.936 | 0.972-0.975 | 0.833-0.850 | 0.950-0.967 | Candidate scorer weights are not brittle |
| Rule/profile matching weights | +/-20% one-factor | 9.608 | 1.378-1.445 | 1.206-1.273 | 0.899 | 0.975 | 0.750 | 0.967 | More conservative profile variants preserve recall and Sem-CSR |
| LLM switch check | submission cached profile-auto | 9.608 | 1.185 | 1.046 | 0.952 | 0.975 | 0.883 | 0.967 | Cached architecture-only LLM reranking improves compactness and precision |

