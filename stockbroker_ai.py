import os

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import bcrypt
import time
import os
try:
    with open("password_hash.txt", "rb") as f:
        hashed_password = f.read()
except FileNotFoundError:
    st.error("Missing password_hash.txt! Upload it to your app folder.")
    st.stop()

attempts = st.session_state.get("attempts", 0)

def login():
    global attempts
    st.title("🔒 Stockbroker AI - Secure Access")
    pw = st.text_input("Enter Password:", type="password")
    if pw:
        if attempts >= 3:
            st.error("Too many attempts. Locked for 30 seconds.")
            time.sleep(30)
            attempts = 0

        if bcrypt.checkpw(pw.encode(), hashed_password):
            st.session_state["authenticated"] = True
        else:
            st.session_state["attempts"] = attempts + 1
            st.error("Incorrect password.")
