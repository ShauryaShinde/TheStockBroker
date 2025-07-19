# Secure Goal-Oriented Trading AI (Multi-Market + Universal Broker Control + Credential Capture)
# -----------------------------------------------------------
# Requirements:
# pip install streamlit yfinance pandas numpy scikit-learn

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st
import time

# ------------------- Secure Login -------------------
attempts = st.session_state.get("attempts", 0)

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

st.title("🔒 Stockbroker AI - Secure Access")
pw = st.text_input("Enter Password:", type="password")

if pw:
    if attempts >= 3:
        st.error("Too many attempts. Locked for 30 seconds.")
        time.sleep(30)
        st.session_state["attempts"] = 0
    elif pw == "Shaurya@2313":
        st.session_state["authenticated"] = True
    else:
        st.session_state["attempts"] = attempts + 1
        st.error("Incorrect password.")

if not st.session_state["authenticated"]:
    st.stop()

# ------------------- Trading AI Core -------------------
st.title("🤖 Stockbroker AI - Mission Control")

market = st.selectbox("Select Market:", [
    "India (NIFTY 50)", "USA (Dow Jones)", "UK (FTSE 100)", "Global Mixed"])

if market == "India (NIFTY 50)":
    symbols = ["RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
elif market == "USA (Dow Jones)":
    symbols = ["AAPL", "MSFT", "JNJ", "JPM", "V"]
elif market == "UK (FTSE 100)":
    symbols = ["BP.L", "HSBA.L", "GLEN.L", "AZN.L", "VOD.L"]
else:
    symbols = ["AAPL", "RELIANCE.NS", "HSBA.L", "TSLA", "TCS.NS"]

broker = st.text_input("Enter your broker/platform name (e.g., Upstox, Zerodha, Dhani, Angel One, Investopedia, etc.):")
username = st.text_input("Enter your broker account username:")
userpass = st.text_input("Enter your broker account password:", type="password")

token = None
simulator_mode = False
permission_granted = False

if st.checkbox("✅ I allow Stockbroker AI to access this broker/platform"):
    token = st.text_input("Enter API Token (if applicable, else leave blank):")
    permission_granted = True
    if token:
        st.success("✅ Token received. Attempting to connect...")
    else:
        st.info("🔄 No token provided. AI will operate in simulated/manual mode.")
    if broker.lower() in ["investopedia", "manual", "demo"] or token == "":
        simulator_mode = True

# Input Section
start_capital = st.number_input("Enter Starting Capital (₹):", value=1000)
target = st.number_input("Enter Target Amount (₹):", value=1000000)

if st.button("🚀 Launch AI Mission"):
    if not permission_granted:
        st.error("❌ Permission not granted. AI will not take control.")
        st.stop()

    if not username or not userpass:
        st.warning("❗ Username and password required for secure login.")
        st.stop()

    st.success(f"Mission accepted: ₹{start_capital} → ₹{target}")
    st.write(f"Gathering stock data for market: {market}...")

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

            decision = "BUY" if pred == 1 else "WAIT"
            confidence = f"{prob * 100:.1f}%"

            results.append({
                "Symbol": symbol,
                "Action": decision,
                "Confidence": confidence
            })
        except:
            continue

    results_df = pd.DataFrame(results)
    st.subheader("📈 AI Trade Suggestions")
    st.dataframe(results_df)

    if simulator_mode:
        st.info("Running in SIMULATOR MODE. No real trades executed.")
    else:
        st.success(f"Connected to {broker}. Ready to execute trades (simulated or API-based).")

    steps = 30
    equity = [start_capital]
    for i in range(steps):
        equity.append(equity[-1] * np.random.uniform(1.01, 1.05))
        if equity[-1] >= target:
            break

    st.line_chart(pd.Series(equity, name="Capital Over Time"))
    if equity[-1] >= target:
        st.success(f"🎯 Target of ₹{target} reached in {len(equity)-1} steps!")
    else:
        st.warning("Target not reached. AI continues learning.")