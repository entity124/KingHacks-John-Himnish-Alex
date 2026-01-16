import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def analyze_model_performance(report_path, transactions_path):
    # 1. Load the Data
    try:
        report_df = pd.read_csv(report_path)
        transactions_df = pd.read_csv(transactions_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Preprocess Dates for Merging
    try:
        report_df['join_date'] = pd.to_datetime(report_df['Last_Txn']).dt.date
        transactions_df['join_date'] = pd.to_datetime(transactions_df['timestamp']).dt.date
    except Exception as e:
        print(f"Error processing dates: {e}")
        return

    # 3. Merge Datasets
    merged_df = pd.merge(
        report_df,
        transactions_df,
        left_on=['User', 'Merchant', 'join_date'],
        right_on=['user_id', 'merchant', 'join_date'],
        how='inner'
    )

    if merged_df.empty:
        print("Error: Merged dataset is empty. Check if user IDs and dates match between files.")
        return

    # 4. Define Labels
    # Prediction: "FLAGGED" = True (1), "Safe" = False (0)
    merged_df['pred_label'] = merged_df['Verdict'] == 'FLAGGED'
    
    # Ground Truth: Anything NOT 'normal' is Fraud (True/1)
    merged_df['actual_label'] = merged_df['pattern_label'] != 'normal'

    # 5. Handle Duplicates (Aggregation)
    grouped_results = merged_df.groupby(['User', 'Merchant', 'join_date']).agg({
        'actual_label': 'max',      # True if any transaction was fraud
        'pred_label': 'first'       # The model's verdict for that day
    }).reset_index()

    y_true = grouped_results['actual_label']
    y_pred = grouped_results['pred_label']

    # 6. Calculate Metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # 7. Print Report (Professional Format)
    print("\n" + "="*50)
    print("VETERAN MODEL PERFORMANCE EVALUATION")
    print("="*50)
    print(f"Total Evaluated Transactions: {len(y_true)}")
    print("-" * 50)
    print(f"{'Metric':<25} | {'Value':<10}")
    print("-" * 50)
    print(f"{'Overall Accuracy':<25} | {accuracy:.2%}")
    print(f"{'False Positive Rate':<25} | {fpr:.2%}")
    print(f"{'Precision':<25} | {precision:.2%}")
    print(f"{'Recall (Sensitivity)':<25} | {recall:.2%}")
    print(f"{'F1 Score':<25} | {f1:.4f}")
    print("-" * 50)
    print("CONFUSION MATRIX DETAILS")
    print("-" * 50)
    print(f"{'True Negatives':<25} : {tn} (Correctly identified as Safe)")
    print(f"{'False Positives':<25} : {fp} (Incorrectly flagged as Fraud)")
    print(f"{'False Negatives':<25} : {fn} (Fraud missed by model)")
    print(f"{'True Positives':<25} : {tp} (Fraud correctly detected)")
    print("="*50 + "\n")

# --- EXECUTION ---
analyze_model_performance(
    report_path='Final_Report_Veteran.csv', 
    transactions_path='mock_transactions-2.csv'
)