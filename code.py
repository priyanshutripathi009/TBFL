import numpy as np 
import pandas as pd 
import random 
import time 
from multiprocessing import Process, Manager 
 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import RobustScaler 
from sklearn.feature_selection import SelectKBest, f_classif 
import matplotlib.pyplot as plt 
import seaborn as sns 
 
df = pd.read_csv("parkinsons_updrs.csv") 
 
if "status" in df.columns: 
    y = df["status"].astype(int) 
    X = df.drop("status", axis=1) 
else: 
    y = (df["total_UPDRS"] > df["total_UPDRS"].median()).astype(int) 
    X = df.drop("total_UPDRS", axis=1) 
 
print("\nTotal Columns:", len(df.columns)) 
print("Total Rows:", len(df)) 
print("Column Names:\n", df.columns) 
 
print("\nTop 10 Data:\n") 
print(df.head(10).to_string()) 
 
scaler = RobustScaler() 
X_scaled = scaler.fit_transform(X) 
 
selector = SelectKBest(score_func=f_classif, k=15) 
X_selected = selector.fit_transform(X_scaled, y) 
 
X_train, X_test, y_train, y_test = train_test_split( 
    X_selected, y, test_size=0.2, random_state=42 
) 
 
num_clients = 5 
clients = [f"C{i+1}" for i in range(num_clients)] 
 
X_split = np.array_split(X_train, num_clients) 
y_split = np.array_split(y_train, num_clients) 
 
manual_history = { 
    "C1": [1,1,1,1,1,0,1,1,0,1], 
    "C2": [1,0,1,1,0,1,1,0,1,1], 
    "C3": [1,1,1,0,1,1,1,1,0,1], 
    "C4": [1,0,0,1,0,1,0,1,0,1], 
    "C5": [0,0,1,0,0,1,0,0,1,0] 
} 
 
def client_process(cid, X, y, score_dict, model_dict, time_dict): 
 
    start_time = time.perf_counter() 
 
    cpu = random.randint(2, 16) 
    ram = random.uniform(2, 16) 
    internet_speed = random.uniform(20, 100) 
    bandwidth = random.uniform(10, 50) 
    battery_percent = random.uniform(30, 100) 
 
    score = (0.25 * cpu + 0.20 * ram + 0.20 * bandwidth + 
             0.15 * battery_percent + 0.20 * internet_speed) 
 
    score_dict[cid] = score 
 
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42) 
    rf.fit(X, y) 
 
    model_dict[cid] = rf 
 
    end_time = time.perf_counter() 
    time_dict[cid] = end_time - start_time 
 
if __name__ == "__main__": 
 
    num_runs = 10 
    all_accuracies = [] 
    all_results = [] 
 
    prev_trust_scores = {cid: 0 for cid in clients} 
 
    for run in range(num_runs): 
 
        print(f"\n=========== RUN {run+1} ===========") 
 
        manager = Manager() 
        score_dict = manager.dict() 
        model_dict = manager.dict() 
        time_dict = manager.dict() 
 
        processes = [] 
 
        for i, cid in enumerate(clients): 
            p = Process( 
                target=client_process, 
                args=(cid, X_split[i], y_split[i], 
                      score_dict, model_dict, time_dict) 
            ) 
            processes.append(p) 
            p.start() 
 
        for p in processes: 
            p.join() 
 
        round_time = max(time_dict.values()) 
 
        print("\nTraining Time (per client):") 
        for cid in clients: 
            print(f"{cid} → {time_dict[cid]:.4f} sec") 
 
        print(f"\nTotal Training Time: {round_time:.4f} sec") 
        raw_scores = np.array([score_dict[cid] for cid in clients]) 
 
        norm_scores = (raw_scores - raw_scores.min()) / ( 
            raw_scores.max() - raw_scores.min() + 1e-8 
        ) 
 
        print("\nNormalized Scores:") 
        for i, cid in enumerate(clients): 
            print(f"{cid} → {norm_scores[i]:.4f}") 
 
        trust_scores = {} 
 
        for i, cid in enumerate(clients): 
            Si = norm_scores[i] 
            Hi = sum(manual_history[cid]) / len(manual_history[cid]) 
 
            current_trust =  Si * Hi 
            final_trust = (prev_trust_scores[cid] + current_trust) / 2 
 
            trust_scores[cid] = final_trust 
 
        print("\nTrust Scores:", trust_scores) 
 
        T = np.mean(list(trust_scores.values())) 
        print(f"\nThreshold Value (T): {T:.4f}") 
 
        selected_clients = [ 
            cid for cid in clients if trust_scores[cid] >= T 
        ] 
 
        if len(selected_clients) < 2: 
            print("Too few clients selected → applying fallback") 
 
            sorted_clients = sorted( 
                trust_scores.items(), 
                key=lambda x: x[1], 
                reverse=True 
            ) 
 
            selected_clients = [cid for cid, _ in sorted_clients[:3]] 
 
        if len(selected_clients) > len(clients): 
            selected_clients = selected_clients[:len(clients)] 
 
        print("Selected Clients:", selected_clients) 
 
        weights = {cid: trust_scores[cid] for cid in selected_clients} 
 
        def predict_ensemble(X): 
            final_pred = np.zeros(len(X)) 
            den = sum(weights.values()) 
            if den == 0: 
                return np.zeros(len(X)) 
 
            for cid in selected_clients: 
                model = model_dict[cid] 
                pred = model.predict_proba(X)[:, 1] 
                final_pred += weights[cid] * pred 
 
            return (final_pred / den > 0.5).astype(int) 
 
        start = time.perf_counter() 
        y_pred = predict_ensemble(X_test) 
        pred_time = time.perf_counter() - start 
 
        acc = accuracy_score(y_test, y_pred) 
 
        print(f"Accuracy: {acc:.4f}") 
        print(f"Prediction Time: {pred_time:.6f} sec") 
 
        all_accuracies.append(acc) 
 
        all_results.append({ 
            "accuracy": acc, 
            "trust_scores": trust_scores, 
            "norm_scores": norm_scores, 
            "clients": selected_clients, 
            "weights": weights, 
            "training_time": round_time, 
            "prediction_time": pred_time 
        }) 
 
        prev_trust_scores = trust_scores.copy() 
 
    selected_counts = [len(result["clients"]) for result in all_results] 
 
    runs = [f"Run {i+1}" for i in range(len(selected_counts))] 
 
    plt.figure(figsize=(10, 5)) 
    plt.bar(runs, selected_counts) 
 
    plt.xlabel("Rounds") 
    plt.ylabel("Number of Selected Clients") 
    plt.title("Selected Clients Across 10 Rounds") 
 
    plt.xticks(rotation=45) 
    plt.grid(axis='y', linestyle='--', alpha=0.6) 
 
    plt.show() 
 
import numpy as np 
import matplotlib.pyplot as plt 
 
best_index = np.argmax(all_accuracies) 
best = all_results[best_index] 
 
avg_accuracy = np.mean(all_accuracies) 
 
print("\n=========== FINAL RESULT ===========") 
print(f"Best Accuracy: {best['accuracy']:.4f}") 
print(f"Average Accuracy: {avg_accuracy:.4f}") 
 
print("\nNormalized Scores (All Clients):") 
 
all_clients = list(best["trust_scores"].keys()) 
norm_scores = best["norm_scores"] 
 
for i, cid in enumerate(all_clients): 
    print(f"{cid} → {norm_scores[i]:.4f}") 
 
norm_scores = best["norm_scores"] 
clients = list(best["trust_scores"].keys()) 
 
plt.figure(figsize=(10, 6)) 
plt.bar(clients, norm_scores) 
 
plt.xlabel("Clients") 
plt.ylabel("Normalized Score") 
plt.title("Normalized Raw Score of Each Client") 
 
plt.xticks(rotation=45) 
 
for i, v in enumerate(norm_scores): 
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center') 
 
plt.tight_layout() 
plt.show() 
 
print(f"\nSelected Clients: {best['clients']}") 
print(f"Training Time: {best['training_time']:.4f} sec") 
print(f"Prediction Time: {best['prediction_time']:.6f} sec") 
 
print("\nTrust Scores:") 
for cid, score in best["trust_scores"].items(): 
    print(f"{cid} → {score:.4f}") 
 
clients = list(best["trust_scores"].keys()) 
scores = list(best["trust_scores"].values()) 
 
plt.figure(figsize=(10, 6)) 
plt.bar(clients, scores) 
 
plt.xlabel("Clients") 
plt.ylabel("Trust Score") 
plt.title("Trust Score of Each Client") 
 
plt.xticks(rotation=45) 
 
for i, v in enumerate(scores): 
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center') 
 
plt.tight_layout() 
plt.show() 
 
clients = list(best["trust_scores"].keys()) 
scores = list(best["trust_scores"].values()) 
 
T = np.mean(scores) 
plt.figure(figsize=(10, 6)) 
plt.bar(clients, scores, color='tab:blue') 
plt.axhline(y=T, color='tab:blue', linestyle='--', linewidth=2, label=f"Threshold 
= {T:.3f}") 
 
plt.xlabel("Clients") 
plt.ylabel("Trust Score") 
plt.title("Trust Score of Each Client with Threshold Line") 
 
for i, v in enumerate(scores): 
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center') 
 
plt.legend() 
plt.tight_layout() 
plt.show() 
 
best_trust = best["trust_scores"] 
best_norm = best["norm_scores"] 
best_weights = best["weights"] 
 
clients = list(best_trust.keys()) 
T = np.mean(list(best_trust.values())) 
 
best_index = np.argmax(all_accuracies) 
    best = all_results[best_index] 
 
    avg_accuracy = np.mean(all_accuracies) 
 
    print("\n================ FINAL RESULT ================") 
 
    print(f"Best Accuracy: {best['accuracy']:.4f}") 
    print(f"Average Accuracy: {avg_accuracy:.4f}") 
 
    print(f"Selected Clients (Best Run): {best['clients']}") 
    print(f"Training Time (Best Run): {best['training_time']:.4f} sec") 
    print(f"Prediction Time (Best Run): {best['prediction_time']:.6f} sec") 
 
selected_clients = best["clients"] 
selected_trust = [best["trust_scores"][cid] for cid in selected_clients] 
 
plt.figure(figsize=(8,5)) 
plt.bar(selected_clients, selected_trust) 
 
plt.title("Selected Clients - Trust Scores") 
plt.xlabel("Clients") 
plt.ylabel("Trust Score") 
 
plt.ylim(0,1) 
 
for i, v in enumerate(selected_trust): 
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center') 
plt.show() 
plt.figure(figsize=(8,6)) 
plt.pie( 
list(best_weights.values()), 
labels=list(best_weights.keys()), 
autopct='%1.2f%%', 
startangle=90 
) 
plt.title("Best Run: Client Contribution") 
plt.show() 
import seaborn as sns 
plt.figure(figsize=(12, 8)) 
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f") 
plt.title("Feature Correlation Heatmap") 
plt.show() 
