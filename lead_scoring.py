import pandas as pd                        
import numpy as np                          
import matplotlib.pyplot as plt             
import seaborn as sns                     
import warnings
warnings.filterwarnings("ignore")         
import os                                    
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler      
from sklearn.linear_model import LogisticRegression    
from sklearn.ensemble import RandomForestClassifier    
from sklearn.metrics import (                         
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

df = pd.read_csv("facebook_ads.csv")

print("=" * 65)
print("DATASET LOADED")
print("=" * 65)
print(f"  Total leads     : {len(df)}")
print(f"  Columns         : {list(df.columns)}")
print(f"  Missing values  : {df.isnull().sum().sum()}")

print("\nFirst 5 rows of data:")
print(df.head().to_string(index=False))

converted     = df["Clicked"].sum()        
not_converted = len(df) - converted
conv_rate     = round(converted / len(df) * 100, 1)

print(f"\n  Converted (Clicked=1)     : {converted} people ({conv_rate}%)")
print(f"  Not Converted (Clicked=0) : {not_converted} people ({100 - conv_rate}%)")


print("\n")
print("=" * 65)
print("DATA EXPLORATION")
print("=" * 65)

print("\nBasic Statistics:")
print(df[["Time Spent on Site", "Salary", "Clicked"]].describe().round(2).to_string())

comparison = df.groupby("Clicked")[["Time Spent on Site", "Salary"]].mean().round(2)
comparison.index = ["Did NOT Convert (0)", "Converted (1)"]

print("\nAverage behaviour — Converted vs Not Converted:")
print(comparison.to_string())

df["engagement_score"] = (
    (df["Time Spent on Site"] - df["Time Spent on Site"].min()) /
    (df["Time Spent on Site"].max() - df["Time Spent on Site"].min()) * 100
).round(2)

df["salary_tier"] = pd.cut(
    df["Salary"],
    bins=[0, 40000, 70000, 100000, float("inf")],   
    labels=["Low", "Medium", "High", "Very High"]   
)

avg_time = df["Time Spent on Site"].mean()  
df["high_engagement"] = (df["Time Spent on Site"] > avg_time).astype(int)


print("\nNew features created:")
print(df[["Names", "Time Spent on Site", "engagement_score",
          "salary_tier", "high_engagement", "Clicked"]].head(8).to_string(index=False))

X = df[["Time Spent on Site", "Salary", "engagement_score", "high_engagement"]]

y = df["Clicked"]

print("\n")
print("=" * 65)
print("PREPARING DATA FOR ML")
print("=" * 65)
print(f"  Features (X) shape : {X.shape}")  
print(f"  Target (y) shape   : {y.shape}")  
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  Training set size  : {len(X_train)} leads")
print(f"  Testing set size   : {len(X_test)} leads")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled  = scaler.transform(X_test)


print("\n")
print("=" * 65)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 65)

lr_model = LogisticRegression(random_state=42)

lr_model.fit(X_train_scaled, y_train)

lr_predictions = lr_model.predict(X_test_scaled)

lr_probabilities = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_auc      = roc_auc_score(y_test, lr_probabilities)


print(f"  Accuracy : {round(lr_accuracy * 100, 2)}%")
print(f"  AUC Score: {round(lr_auc, 4)}")
print("\nDetailed Report:")
print(classification_report(y_test, lr_predictions,
                             target_names=["Not Converted", "Converted"]))


print("\n")
print("=" * 65)
print("MODEL 2: RANDOM FOREST")
print("=" * 65)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

rf_predictions   = rf_model.predict(X_test)
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_auc      = roc_auc_score(y_test, rf_probabilities)

print(f"  Accuracy : {round(rf_accuracy * 100, 2)}%")
print(f"  AUC Score: {round(rf_auc, 4)}")
print("\nDetailed Report:")
print(classification_report(rf_predictions, y_test,
                             target_names=["Not Converted", "Converted"]))

feature_names       = X.columns.tolist()
feature_importances = rf_model.feature_importances_ 

print("Feature Importances (what drives conversion):")
for name, importance in sorted(zip(feature_names, feature_importances),
                                key=lambda x: x[1], reverse=True):
    bar = "█" * int(importance * 50)  
    print(f"  {name:<25} : {bar} {round(importance * 100, 1)}%")

print("\n")
print("=" * 65)
print("MODEL COMPARISON")
print("=" * 65)
print(f"  Logistic Regression → Accuracy: {round(lr_accuracy*100,2)}%  |  AUC: {round(lr_auc,4)}")
print(f"  Random Forest       → Accuracy: {round(rf_accuracy*100,2)}%  |  AUC: {round(rf_auc,4)}")

if rf_auc >= lr_auc:
    best_model      = rf_model
    best_model_name = "Random Forest"
    
    all_probabilities = rf_model.predict_proba(X)[:, 1]
else:
    best_model      = lr_model
    best_model_name = "Logistic Regression"
    all_probabilities = lr_model.predict_proba(scaler.transform(X))[:, 1]

print(f"\n  Best model selected: {best_model_name}")

print("\n")
print("=" * 65)
print("LEAD SCORING — ALL 499 LEADS RANKED")
print("=" * 65)

df["conversion_probability"] = (all_probabilities * 100).round(1)

df["lead_score"] = df["conversion_probability"].round(0).astype(int)

def lead_tier(score):
    if score >= 70:
        return "🔥 Hot Lead"   
    elif score >= 45:
        return "warm Lead"        
    else:
        return "Cold Lead"    
df["lead_tier"] = df["lead_score"].apply(lead_tier)

df_scored = df.sort_values("lead_score", ascending=False).reset_index(drop=True)
df_scored["rank"] = df_scored.index + 1 

print("\nTOP 15 LEADS — Most Likely to Convert:")
print(df_scored[["rank", "Names", "Time Spent on Site", "Salary",
                  "lead_score", "lead_tier", "Clicked"]].head(15).to_string(index=False))

print("\n\nLEAD TIER SUMMARY:")
tier_summary = df_scored["lead_tier"].value_counts()
for tier, count in tier_summary.items():
    pct = round(count / len(df) * 100, 1)
    print(f"  {tier:<20} : {count} leads ({pct}%)")

print("\n")
print("=" * 65)
print("BUSINESS INSIGHTS & RECOMMENDATIONS")
print("=" * 65)

hot_leads  = df_scored[df_scored["lead_tier"] == " Hot Lead"]
warm_leads = df_scored[df_scored["lead_tier"] == "warm Lead"]
cold_leads = df_scored[df_scored["lead_tier"] == "Cold Lead"]

print(f"\n HOT LEADS ({len(hot_leads)} people — contact immediately):")
print(f"   Avg time on site : {hot_leads['Time Spent on Site'].mean():.1f} mins")
print(f"   Avg salary       : ${hot_leads['Salary'].mean():,.0f}")
print(f"   Actual conversion: {hot_leads['Clicked'].mean()*100:.1f}% of them actually converted")

print(f"\n WARM LEADS ({len(warm_leads)} people — nurture with follow-ups):")
print(f"   Avg time on site : {warm_leads['Time Spent on Site'].mean():.1f} mins")
print(f"   Avg salary       : ${warm_leads['Salary'].mean():,.0f}")
print(f"   Actual conversion: {warm_leads['Clicked'].mean()*100:.1f}% of them actually converted")

print(f"\n COLD LEADS ({len(cold_leads)} people — low priority):")
print(f"   Avg time on site : {cold_leads['Time Spent on Site'].mean():.1f} mins")
print(f"   Avg salary       : ${cold_leads['Salary'].mean():,.0f}")
print(f"   Actual conversion: {cold_leads['Clicked'].mean()*100:.1f}% of them actually converted")

print(f"\n MODEL RECOMMENDATION:")
print(f"   Use {best_model_name} for scoring")
print(f"   It correctly identifies {round(rf_accuracy*100,1)}% of leads")
print(f"   Prioritize Hot Leads — {hot_leads['Clicked'].mean()*100:.0f}% of them convert")


os.makedirs("lead_scoring_output", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots(figsize=(9, 5))

converted_scores     = df[df["Clicked"] == 1]["lead_score"]
not_converted_scores = df[df["Clicked"] == 0]["lead_score"]

ax.hist(converted_scores,     bins=20, alpha=0.6, color="#2D9E75",
        label="Actually Converted (1)",     edgecolor="white")
ax.hist(not_converted_scores, bins=20, alpha=0.6, color="#993C1D",
        label="Did Not Convert (0)", edgecolor="white")

ax.axvline(x=70, color="red",    linestyle="--", linewidth=1.5, label="Hot Lead threshold (70)")
ax.axvline(x=45, color="orange", linestyle="--", linewidth=1.5, label="Warm Lead threshold (45)")

ax.set_xlabel("Lead Score (0–100)")
ax.set_ylabel("Number of Leads")
ax.set_title("Lead Score Distribution — Converted vs Not Converted",
             fontweight="bold", fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig("lead_scoring_output/chart1_score_distribution.png", dpi=150)
plt.close()
print("\nChart 1 saved: Lead Score Distribution")

fig, ax = plt.subplots(figsize=(7, 4))

sorted_idx         = np.argsort(feature_importances)  
sorted_features    = [feature_names[i] for i in sorted_idx]
sorted_importances = [feature_importances[i] for i in sorted_idx]

ax.barh(sorted_features, sorted_importances, color="#185FA5", edgecolor="white")

for i, val in enumerate(sorted_importances):
    ax.text(val + 0.002, i, f"{val*100:.1f}%", va="center", fontsize=10)

ax.set_xlabel("Importance Score")
ax.set_title("What Drives Conversion? (Random Forest Feature Importance)",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("lead_scoring_output/chart2_feature_importance.png", dpi=150)
plt.close()
print("Chart 2 saved: Feature Importance")

fig, ax = plt.subplots(figsize=(6, 6))

tier_counts = df_scored["lead_tier"].value_counts()
colors_pie  = ["#2D9E75", "#E89B2D", "#993C1D"]   

ax.pie(
    tier_counts.values,
    labels=tier_counts.index,
    autopct="%1.1f%%",
    colors=colors_pie,
    startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
ax.set_title("Lead Tier Distribution", fontweight="bold", fontsize=14)
plt.tight_layout()
plt.savefig("lead_scoring_output/chart3_lead_tiers.png", dpi=150)
plt.close()
print("Chart 3 saved: Lead Tier Breakdown")

fig, ax = plt.subplots(figsize=(9, 6))

colors_scatter = df["Clicked"].map({1: "#2D9E75", 0: "#993C1D"})
scatter = ax.scatter(
    df["Time Spent on Site"],  
    df["Salary"],              
    c=colors_scatter,          
    alpha=0.6,                 
    s=50                       
)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#2D9E75", label="Converted (Clicked=1)"),
    Patch(facecolor="#993C1D", label="Not Converted (Clicked=0)")
]
ax.legend(handles=legend_elements)

ax.set_xlabel("Time Spent on Site (minutes)")
ax.set_ylabel("Salary ($)")
ax.set_title("Time Spent vs Salary — Who Converts?",
             fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("lead_scoring_output/chart4_scatter_conversion.png", dpi=150)
plt.close()
print("Chart 4 saved: Time Spent vs Salary Scatter")

fig, ax = plt.subplots(figsize=(7, 4))

models     = ["Logistic Regression", "Random Forest"]
accuracies = [round(lr_accuracy * 100, 2), round(rf_accuracy * 100, 2)]
aucs       = [round(lr_auc, 4), round(rf_auc, 4)]

x     = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, accuracies, width, label="Accuracy (%)",
               color="#185FA5", edgecolor="white")
bars2 = ax.bar(x + width/2, [a * 100 for a in aucs], width,
               label="AUC Score (×100)", color="#2D9E75", edgecolor="white")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%", ha="center", fontsize=10, fontweight="bold")
for bar, auc in zip(bars2, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{auc}", ha="center", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 115)
ax.set_title("Model Comparison — Accuracy & AUC", fontweight="bold", fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig("lead_scoring_output/chart5_model_comparison.png", dpi=150)
plt.close()
print("Chart 5 saved: Model Accuracy Comparison")


from datetime import date
today      = date.today().strftime("%Y-%m-%d")
excel_path = f"lead_scoring_output/lead_scoring_report_{today}.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

    df_scored[["rank", "Names", "Time Spent on Site", "Salary",
                "engagement_score", "lead_score", "lead_tier", "Clicked"]].to_excel(
        writer, sheet_name="All Leads Ranked", index=False
    )

    hot_leads[["Names", "Time Spent on Site", "Salary",
               "lead_score", "Clicked"]].to_excel(
        writer, sheet_name="Hot Leads", index=False
    )

    warm_leads[["Names", "Time Spent on Site", "Salary",
                "lead_score", "Clicked"]].to_excel(
        writer, sheet_name="Warm Leads", index=False
    )

    model_perf = pd.DataFrame({
        "Model":    ["Logistic Regression", "Random Forest"],
        "Accuracy": [round(lr_accuracy*100,2), round(rf_accuracy*100,2)],
        "AUC":      [round(lr_auc,4), round(rf_auc,4)],
        "Winner":   ["" if rf_auc >= lr_auc else "✓", "✓" if rf_auc >= lr_auc else ""]
    })
    model_perf.to_excel(writer, sheet_name="Model Performance", index=False)

    tier_df = pd.DataFrame({
        "Tier":           tier_counts.index,
        "Count":          tier_counts.values,
        "Percentage":     [f"{round(c/len(df)*100,1)}%" for c in tier_counts.values]
    })
    tier_df.to_excel(writer, sheet_name="Tier Summary", index=False)

print(f"\nExcel report saved: {excel_path}")

print("\n")
print("=" * 65)
print("ANALYSIS COMPLETE")
print("=" * 65)
print(f"  Leads analysed     : {len(df)}")
print(f"  Best model         : {best_model_name}")
print(f"  Model accuracy     : {round(rf_accuracy*100,1)}%")
print(f"  Hot leads found    : {len(hot_leads)}")
print(f"  Charts generated   : 5")
print(f"  Excel sheets       : 5")
print(f"  Output folder      : lead_scoring_output/")
print("=" * 65)