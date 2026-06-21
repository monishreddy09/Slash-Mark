# Monitoring Checklist

- Track fraud rate drift weekly.
- Recompute PR-AUC, recall, and cost on recent labeled data.
- Monitor feature distributions for `Amount`, `Time`, and engineered fields.
- Calibrate or retune the threshold when base rates shift.
- Review false negatives first; they are usually the most expensive errors.
- Retrain on fresh data when drift or performance degradation is detected.
