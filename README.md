🚀 Lead Scoring Model 

📌 Overview
This project implements a Lead Scoring System using a notebook-style Python workflow.

It processes raw lead data, trains machine learning models, and generates:
📊 Lead scores (0–100)
🔥 Lead categories (Hot, Warm, Cold)
📈 Visual insights (charts)
📄 Excel reports for business use
📊 Dataset
Total Leads: 499
Source: Facebook Ads dataset (facebook_ads.csv)
Includes engagement, source, and behavioral features

⚙️ Approach
The entire pipeline is built in a single Python script (notebook-style):

Data Cleaning & Preprocessing
Exploratory Data Analysis (EDA)
Feature Selection
Model Training
Evaluation & Comparison
Lead Scoring (0–100 scaling)
Segmentation into:
🔥 Hot
🌤️ Warm
❄️ Cold
Output generation (charts + Excel)

🤖 Models Used
Logistic Regression
Random Forest

📈 Results
✅ Accuracy: 92%
🔥 Hot Leads Conversion Rate: 94.6%
📊 Each lead assigned a score from 0–100

📂 Project Files
lead_scoring.py              
facebook_ads.csv             
lead_scoring_output/         
├── charts (PNG)
├── model comparison
├── segmentation visuals
├── Excel report

📊 Outputs Generated
Score distribution chart
Feature importance chart
Lead segmentation (Hot/Warm/Cold)
Model comparison
Excel report with scored leads

▶️ How to Run
python lead_scoring.py

💡 Key Insight
The model doesn't just perform well statistically — it proves business value, with Hot leads achieving 94.6% actual conversion rate.

🛠️ Tech Stack
Python
Pandas, NumPy
Scikit-learn
Matplotlib / Seaborn

🔗 Future Improvements
Convert into modular pipeline
Build Streamlit dashboard
Deploy as API for real-time scoring

👩‍💻 Author
Ayesha Summaiyya
