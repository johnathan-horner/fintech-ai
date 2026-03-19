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
# Demo mode (default - no AWS needed)
FINTECH_DEMO_MODE = "true"

# Live mode (requires AWS Bedrock access)
# FINTECH_DEMO_MODE = "false"
# AWS_ACCESS_KEY_ID = "..."
# AWS_SECRET_ACCESS_KEY = "..."
# AWS_REGION = "us-east-1"
```

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

## Demo Mode vs. Live Mode

| Feature | Demo Mode | Live Mode |
|---|---|---|
| Portfolio Overview | - [x] Real synthetic data | - [x] Real synthetic data |
| Market Analysis | [DEMO] Pre-generated text | - [x] Bedrock Claude 3 Sonnet |
| Risk (VaR/CVaR) | - [x] Real calculations | - [x] Real calculations |
| Stress Tests | - [x] Real calculations | - [x] Real calculations |
| AI Chat | [DEMO] Canned responses | - [x] Full RAG pipeline |
| MRM Dashboard | - [x] Full validation suite | - [x] Full validation suite |
| AWS Required | - [ ] No | - [x] Yes |

**Demo mode runs the actual Python quant math** (VaR, CVaR, stress tests, MRM validation) - only the LLM narrative calls are mocked. This means the core financial engineering is always live and testable.
