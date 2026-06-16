# 🎯 Lead Scoring System — ML-Powered Conversion Intelligence

> **An end-to-end machine learning pipeline that scores leads 0–100, segments them into Hot/Warm/Cold, and surfaces business insights — built with XGBoost, Random Forest & Streamlit.**

🔗 **[Live Demo → Try it Now](https://leadscoringproject-4zdbml2fauqqo9kwyzbc2p.streamlit.app/)**

---

## 🧠 The Problem This Solves

Most sales teams waste 60–70% of their time chasing leads that will never convert. This system fixes that.

By training on real Facebook Ads lead data, it learns *which leads actually convert* — and ranks every new lead by their probability of becoming a customer. Your sales team focuses only on 🔥 Hot leads. Revenue goes up. Time wasted goes down.

---

## 🚀 What It Does

Upload your leads CSV and get back:

- ✅ **Lead Score (0–100)** for every single lead
- ✅ **Segmentation** → 🔥 Hot · 🌤️ Warm · ❄️ Cold
- ✅ **Feature Importance** → what actually drives conversions
- ✅ **Model Comparison** → Logistic Regression vs Random Forest
- ✅ **Score Distribution Chart** → see where your pipeline stands
- ✅ **Excel Report** → scored leads ready to import into any CRM
- ✅ **92% Accuracy** · 94.6% Hot Lead Conversion Rate

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| ✅ Accuracy | **92%** |
| 🔥 Hot Lead Conversion Rate | **94.6%** |
| 📦 Dataset Size | 499 leads |
| 🧮 Score Range | 0 – 100 |

> **What 94.6% means in practice:** Out of every 100 leads the model labels as Hot, ~95 actually convert. That's not just good stats — that's real revenue impact.

---

## 🤖 ML Models Used

| Model | Role |
|---|---|
| **Random Forest** | Primary scoring model — handles non-linear patterns, feature interactions |
| **Logistic Regression** | Baseline comparison — fast, interpretable |

Both models are trained, evaluated, and compared side-by-side so you can see exactly why Random Forest wins.

---

## 🔥 Lead Segmentation Logic

| Segment | Score Range | What It Means |
|---|---|---|
| 🔥 Hot | 70 – 100 | High intent — contact immediately |
| 🌤️ Warm | 40 – 69 | Needs nurturing — add to drip campaign |
| ❄️ Cold | 0 – 39 | Low priority — don't waste sales time |

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| ML Models | Scikit-learn (Random Forest, Logistic Regression) |
| Data Processing | Pandas, NumPy |
| Visualizations | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Reports | OpenPyXL (Excel) |
| Deployment | Streamlit Community Cloud |

---

## 📁 Project Structure

```
lead_Scoring_project/
│
├── app.py                      # Streamlit dashboard
├── lead_scoring.py             # Core ML pipeline (CLI version)
├── facebook_ads.csv            # Raw lead dataset
├── requirements.txt            # Dependencies
├── lead_scoring_output/        # Generated outputs
│   ├── chart_score_distribution.png
│   ├── chart_feature_importance.png
│   ├── chart_segmentation.png
│   ├── chart_model_comparison.png
│   └── lead_scores_report.xlsx
└── README.md
```

---

## ⚡ Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/Ayesha037/lead_Scoring_project.git
cd lead_Scoring_project
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit dashboard**
```bash
streamlit run app.py
```

**4. Or run the CLI version**
```bash
python lead_scoring.py
```

Opens at `http://localhost:8501` — upload your CSV and get scored leads instantly ✅

---

## 📋 Dataset Schema

The model was trained on Facebook Ads lead data (`facebook_ads.csv`):

| Column | Description |
|---|---|
| `lead_id` | Unique identifier |
| `source` | Traffic source (Facebook, Google, Organic) |
| `engagement_score` | How engaged the lead is |
| `age` / `gender` | Demographic data |
| `campaign_type` | Which ad campaign generated the lead |
| `converted` | Target variable — 1 = converted, 0 = didn't |

---

## 🔄 ML Pipeline

```
Raw CSV
  ↓
Data Cleaning & Null Handling
  ↓
Feature Engineering & Encoding
  ↓
Train/Test Split (80/20)
  ↓
Model Training (Random Forest + Logistic Regression)
  ↓
Evaluation (Accuracy, Precision, Recall, F1)
  ↓
Probability → Lead Score (0–100 scaling)
  ↓
Segmentation (Hot / Warm / Cold)
  ↓
Charts + Excel Report Output
```

---

## 📈 Outputs Generated

| Output | Description |
|---|---|
| `chart_score_distribution.png` | Histogram of all lead scores |
| `chart_feature_importance.png` | Which features drive conversions most |
| `chart_segmentation.png` | Hot/Warm/Cold breakdown |
| `chart_model_comparison.png` | RF vs Logistic Regression head-to-head |
| `lead_scores_report.xlsx` | Every lead with score + segment, CRM-ready |

---

## 🌐 Deployment

Deployed on **Streamlit Community Cloud** — zero infrastructure, always on.

To deploy your own fork:
1. Push to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select repo → set `app.py` as main file
4. Hit **Deploy** — live in ~2 minutes 🚀

---

## 📦 Dependencies

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
openpyxl
```

---

## 💡 Key Business Insight

> The model doesn't just perform well statistically — it proves **business value**.
>
> Hot leads achieve a **94.6% actual conversion rate**, meaning sales teams using this system can focus their energy where it counts most and dramatically improve close rates without hiring more people.

---

## 🗺️ Future Roadmap

- [ ] Add XGBoost & LightGBM for even higher accuracy
- [ ] Real-time scoring API (FastAPI / Flask)
- [ ] CRM integration (HubSpot, Salesforce webhook)
- [ ] A/B testing module for lead nurturing strategies
- [ ] Auto-retraining when new lead data is uploaded

---

## 👩‍💻 Author

**Ayesha Summaiyya** — [@Ayesha037](https://github.com/Ayesha037)

Built as part of an ML portfolio project focused on real-world CRM analytics and sales intelligence.

---

## ⭐ Show Some Love

If this helped you, drop a ⭐ on the repo — it keeps the motivation going!

---

*Built with Python · Scikit-learn · Streamlit · Deployed on the cloud*
