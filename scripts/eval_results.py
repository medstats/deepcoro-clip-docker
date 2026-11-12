import pandas as pd

df = pd.read_csv('/workspace/results/summary.csv')

study_map = {
    'STUDY_E5A7C532': ('POS_001_SYNTAX45', 'Severe CAD (SYNTAX 45.17)'),
    'STUDY_1D1A4141': ('POS_002_SYNTAX30', 'Severe CAD (SYNTAX 29.67)'),
    'STUDY_D601092B': ('NEG_001_SYNTAX0', 'Normal Coronaries (SYNTAX 0.0)'),
    'STUDY_72A426F9': ('NEG_002_SYNTAX0', 'Normal Coronaries (SYNTAX 0.0)')
}

key_segments = ['left_main', 'prox_lad', 'mid_lad', 'prox_lcx', 'prox_rca', 'mid_rca']

fmt = "{:<18} | {:<12} | {:<10} | {:<10} | {:<12} | {:<10} | {}"
header = fmt.format("Study / Case", "Segment", "Pred Prob", "Threshold", "Cont Score", "Model Call", "Ground Truth")
sep = "=" * len(header)
line_sep = "-" * len(header)

print(sep)
print(header)
print(sep)

for study_id, (case_name, gt_desc) in study_map.items():
    df_s = df[df['StudyID'] == study_id]
    for seg in key_segments:
        row = df_s[df_s['CoronarySegment'] == seg]
        if not row.empty:
            prob = float(row['SignificantStenosisProb'].values[0])
            th = float(row['SignificantStenosisThreshold'].values[0])
            score = float(row['ContinuousStenosisScore'].values[0])
            is_pos = bool(row['ThresholdPositive'].values[0])
            call = 'POSITIVE' if is_pos else 'Negative'
            print(fmt.format(case_name, seg, f"{prob:.4f}", f"{th:.4f}", f"{score:.2f}", call, gt_desc))
    print(line_sep)
