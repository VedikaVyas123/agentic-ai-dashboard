# 🧠 Agentic AI Dashboard — Analyze • Predict • Reflect  

An interactive **AI-powered analytics dashboard** built with **Gradio**, **Scikit-learn**, and **OpenAI API** to help users upload datasets, analyze insights, train predictive models, and generate intelligent reflections automatically.  

---

## 🌟 Key Features  

📂 **Upload CSV or Excel files** instantly  

🎯 **Choose target variable** (e.g., revenue) for predictions  

📊 **Auto-EDA Generation** using `ydata-profiling` with descriptive statistics and visual summaries  

🤖 **Baseline Model (Linear Regression)** with evaluation metrics (**MAE**, **R²**)  

🪄 **AI Reflection:** LLM-based summary of insights and model suggestions  

💅 **Elegant dark UI** designed with Gradio for an intuitive and modern experience  

---

## 🧩 Tech Stack  

| Category | Technologies |
|-----------|--------------|
| Frontend | Gradio |
| Backend & ML | Scikit-learn, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| EDA | ydata-profiling |
| LLM Integration | OpenAI API |
| Environment | Python 3.10+, Virtualenv |

---

## 🧱 Architecture Diagram

```mermaid
flowchart TD
    A[Upload CSV/Excel File] --> B[Exploratory Data Analysis using ydata-profiling]
    B --> C[Train Baseline Linear Regression Model]
    C --> D[Evaluate Model using MAE and R2 Score]
    D --> E[Generate AI Reflections and Insights]
    E --> F[Visualize Results in Gradio Dashboard]

🗂 Workflow Summary:
1️⃣ User uploads dataset
2️⃣ Automated EDA report is generated
3️⃣ Linear regression model trains
4️⃣ Model performance & predictions are displayed
5️⃣ LLM interprets and provides actionable insights

⚙️ Installation & Setup
bash
Copy code
# 1️⃣ Clone the repository
git clone https://github.com/VedikaVyas123/agentic-ai-dashboard.git
cd agentic-ai-dashboard

# 2️⃣ Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # (on Windows)
# or source .venv/bin/activate   # (on macOS/Linux)

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the app
python app_advanced.py
Once it runs, open http://127.0.0.1:7860 in your browser 🌐

📈 Example Outputs
Visuals generated:

Revenue distribution histogram

Revenue over time (trend line)

Top correlations with target variable

Average revenue by category

Baseline Regression Results:

MAE: 3298.91

R²: 0.867

EDA report saved as eda_report.html

🧠 AI Reflection
The model explains ~86% variance in revenue, demonstrating strong predictive performance.
Key influencers include units_sold and avg_price.
Future enhancements could explore non-linear models or feature scaling for improved robustness.

👩‍💻 Author
Vedika Vyas
🎓 MS Data Analytics @ San José State University
📍 San Jose, California
🔗 LinkedIn | GitHub

💬 Acknowledgements
Special thanks to open-source contributors and the Gradio and Scikit-learn communities for empowering data-driven AI dashboards.

