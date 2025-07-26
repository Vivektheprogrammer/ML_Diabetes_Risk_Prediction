# Streamlit Deployment Guide

## Deploy to Streamlit Cloud

1. **Visit Streamlit Cloud**: Go to [share.streamlit.io](https://share.streamlit.io)

2. **Sign in**: Use your GitHub account to sign in

3. **Deploy App**: 
   - Click "New App"
   - Select your repository: `Vivektheprogrammer/ML_Diabetes_Risk_Prediction`
   - Choose branch: `main`
   - Main file path: `fullapp.py`
   - App URL: Choose a custom URL or use the default

4. **Configure**: The app will automatically install dependencies from `requirements.txt`

5. **Deploy**: Click "Deploy!" and wait for the build to complete

## Local Development

To run locally:

```bash
# Clone the repository
git clone https://github.com/Vivektheprogrammer/ML_Diabetes_Risk_Prediction.git
cd ML_Diabetes_Risk_Prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run fullapp.py
```

## Environment Variables (if needed)

If your app requires any environment variables, add them in the Streamlit Cloud dashboard under "Advanced settings" during deployment.

## Troubleshooting

- Ensure all model files (.joblib) are included in the repository
- Check that the file paths in your code are relative (no absolute paths)
- Verify all dependencies are listed in requirements.txt with compatible versions

## Update Process

To update your deployed app:
1. Make changes to your code locally
2. Commit and push to GitHub
3. Streamlit Cloud will automatically redeploy on push to main branch
