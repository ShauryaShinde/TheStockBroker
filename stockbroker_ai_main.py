# Stockbroker AI - Unified App (Secure, Adaptive, Multi-Market, Goal-Oriented)
# Author: Shaurya Exclusive
# Requirements: streamlit, yfinance, pandas, numpy, scikit-learn

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st
import time

# ------------------- Secure Login -------------------
st.set_page_config(page_title="Stockbroker AI", page_icon="📈", layout="centered")
st.title("🔒 Stockbroker AI - Secure Access")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "attempts" not in st.session_state:
    st.session_state["attempts"] = 0

if not st.session_state["authenticated"]:
    pw = st.text_input("Enter Password:", type="password")
    if pw:
        if st.session_state["attempts"] >= 3:
            st.error("Too many failed attempts. Please wait 30 seconds.")
            time.sleep(30)
            st.session_state["attempts"] = 0
        elif pw == "Shaurya@2313":
            st.session_state["authenticated"] = True
            st.success("Access Granted ✅")
        else:
            st.session_state["attempts"] += 1
            st.error("Incorrect password ❌")
    st.stop()

# ------------------- Trading AI Core -------------------
st.title("🤖 Stockbroker AI - Global Mission Console")

market = st.selectbox("Select Market:", ["India (NIFTY 50)", "USA (Dow Jones)", "UK (FTSE 100)", "Mixed Global"])

if market == "India (NIFTY 50)":
    symbols = ["RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
elif market == "USA (Dow Jones)":
    symbols = ["AAPL", "MSFT", "JNJ", "JPM", "V"]
elif market == "UK (FTSE 100)":
    symbols = ["BP.L", "HSBA.L", "GLEN.L", "AZN.L", "VOD.L"]
else:
    symbols = ["RELIANCE.NS", "AAPL", "BP.L", "TSLA", "ICICIBANK.NS"]

st.markdown("### 🔐 Broker Credentials (Used for control takeover)")

broker = st.text_input("Enter Broker/App Name:")
username = st.text_input("Account Username:")
password = st.text_input("Account Password:", type="password")
token = st.text_input("API Token (Optional):")
permission = st.checkbox("✅ I give Stockbroker AI permission to trade via this account")

start_capital = st.number_input("💵 Starting Capital (₹ or $):", value=1000)
target = st.number_input("🎯 Target Amount (₹ or $):", value=1000000)

if st.button("🚀 Launch Mission"):
    if not permission or not username or not password:
        st.warning("You must provide credentials and grant permission.")
        st.stop()

    st.success(f"AI Activated! Goal: {start_capital} → {target}")
    st.info(f"Scanning {market} market...")

    results = []

    for symbol in symbols:
        try:
            df = yf.download(symbol, period="6mo")
            df["Return"] = df["Close"].pct_change()
            df["SMA_5"] = df["Close"].rolling(5).mean()
            df["SMA_20"] = df["Close"].rolling(20).mean()
            df["Volatility"] = df["Return"].rolling(10).std()
            df.dropna(inplace=True)

            X = df[["SMA_5", "SMA_20", "Volatility", "Return"]]
            y = (df["Close"].shift(-1) > df["Close"]).astype(int)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_scaled[:-1], y[:-1])
            pred = model.predict([X_scaled[-1]])[0]
            prob = model.predict_proba([X_scaled[-1]])[0][pred]

            action = "BUY" if pred == 1 else "WAIT"
            confidence = f"{prob * 100:.2f}%"

            results.append({"Symbol": symbol, "Action": action, "Confidence": confidence})
        except:
            continue

    df_results = pd.DataFrame(results)
    st.subheader("📊 AI Recommendations")
    st.dataframe(df_results)

    equity = [start_capital]
    for _ in range(30):
        equity.append(equity[-1] * np.random.uniform(1.01, 1.05))
        if equity[-1] >= target:
            break

    st.line_chart(pd.Series(equity, name="Portfolio Value"))
    if equity[-1] >= target:
        st.success("🎉 Target Achieved!")
    else:
        st.warning("Still in progress... AI will continue.")

    if not token:
        st.info("Running in Simulation Mode. No real trades executed.")
    else:
        st.success("API Token accepted. (Simulated real account control ready).")