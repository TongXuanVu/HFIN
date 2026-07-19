import pandas as pd
import os

wa_path = r'C:\FederatedLearning\HFIN\IDPS\logs\wa\03-07-26_08-43\metrics_wa.csv'
der_path = r'C:\FederatedLearning\HFIN\IDPS\logs\der\03-07-26_08-43\metrics_der.csv'

def process_180(df):
    df = df.rename(columns={'task': 'task_id', 'round': 'round_in_task'})
    cols = ['task_id', 'round_in_task', 'global_round', 'acc', 'prec_mic', 'prec_mac', 'prec_wei', 'rec_mic', 'rec_mac', 'rec_wei', 'f1_mic', 'f1_mac', 'f1_wei', 'loss']
    df = df[cols]
    
    metric_cols = ['acc', 'prec_mic', 'prec_mac', 'prec_wei', 'rec_mic', 'rec_mac', 'rec_wei', 'f1_mic', 'f1_mac', 'f1_wei']
    for c in metric_cols:
        df[c] = df[c].apply(lambda x: f"{float(x):.2f}")
    df['loss'] = df['loss'].apply(lambda x: f"{float(x):.4f}")
    return df

def process_30(df):
    df_30 = df[df['round_in_task'].astype(int) == 30].copy()
    df_30 = df_30.rename(columns={
        'acc': 'accuracy',
        'f1_mic': 'f1_micro', 'f1_mac': 'f1_macro', 'f1_wei': 'f1_weight',
        'prec_mic': 'precision_micro', 'prec_mac': 'precision_macro', 'prec_wei': 'precision_weight',
        'rec_mic': 'recall_micro', 'rec_mac': 'recall_macro', 'rec_wei': 'recall_weight'
    })
    # Kèm thêm task_id ở đầu cho rõ ràng
    cols = ['task_id', 'accuracy', 'f1_micro', 'f1_macro', 'f1_weight', 'precision_micro', 'precision_macro', 'precision_weight', 'recall_micro', 'recall_macro', 'recall_weight', 'loss']
    return df_30[cols]

df_wa = pd.read_csv(wa_path)
df_der = pd.read_csv(der_path)

df_wa_180 = process_180(df_wa.copy())
df_der_180 = process_180(df_der.copy())

df_wa_30 = process_30(df_wa_180)
df_der_30 = process_30(df_der_180)

out_dir = r'C:\FederatedLearning\HFIN\IDPS\logs\summary_csvs'
os.makedirs(out_dir, exist_ok=True)

df_wa_180.to_csv(f'{out_dir}/wa_180_rounds.csv', index=False)
df_der_180.to_csv(f'{out_dir}/der_180_rounds.csv', index=False)
df_wa_30.to_csv(f'{out_dir}/wa_final_rounds.csv', index=False)
df_der_30.to_csv(f'{out_dir}/der_final_rounds.csv', index=False)

print("Exported 4 CSV files successfully to:", out_dir)
