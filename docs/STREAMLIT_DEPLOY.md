# Streamlit Cloud Deployment Guide

## Deploy to Streamlit Cloud (Free - 5 minutes)

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit - FinTech AI"
git remote add origin https://github.com/yourusername/fintech-ai.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Select your GitHub repo: `yourusername/fintech-ai`
4. Set **Main file path**: `streamlit_app/streamlit_app.py`
5. Set **Python version**: 3.11
6. Click **Deploy**

### Step 3: Configure (Optional)
In the Streamlit Cloud dashboard -> **Advanced settings** -> **Secrets**:

```toml
# Live AI on the Gemini free tier (use a Google AI Studio key on a project with NO billing).
[gemini]
api_key = "..."

# Optional: provider / per-role model overrides (see .streamlit/secrets.toml.template)
# [llm]
# provider = "gemini"
```

Without a key the app runs in demo mode: quant metrics are computed live and AI sections show
deterministic templates filled from the data, labelled "Demo mode".

### Step 4: Generate Demo Data
The demo uses pre-generated synthetic data. Run this once locally and commit the output:
```bash
pip install faker numpy
python3 src/etl/generate_data.py
git add data/
git commit -m "Add synthetic demo data"
git push
```

---

## Demo Mode vs. Live AI

| Feature | Demo mode (no key / fallback) | Live AI (Gemini free tier) |
|---|---|---|
| Portfolio Overview | Computed from synthetic data | Computed from synthetic data |
| Market Analysis report | Data-derived template | Market agent (LangGraph) + grader |
| Risk (VaR/CVaR/stress/backtest) | Computed | Computed |
| AI Chat | Data-derived template | HyDE + FAISS + cross-encoder + Gemini + grader |
| MRM Dashboard | Real src/mrm modules | Real src/mrm modules |
| Cost | $0 | $0 (free tier, 10 live requests per session) |

Live AI is used only for components approved by the held-out eval (`data/mrm/approvals.json`);
otherwise the UI shows "Blocked by model governance: <reason>" and falls back to demo output.
