# ============================================================
# SmartLoan: Loan Approval & Risk Assessment System
# app.py — Streamlit Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="SmartLoan",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load CSS ───────────────────────────────────────────────
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("artifacts/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("artifacts/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("artifacts/feature_columns.json") as f:
        features = json.load(f)
    return model, scaler, features

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    return df

model, scaler, FEATURE_COLS = load_model()
df_raw = load_data()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="brand-icon">💳</span>
        <span class="brand-name">SmartLoan</span>
        <span class="brand-sub">Risk Assessment System</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠  Dashboard", "📊  Analytics", "🔮  Prediction Tool"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-info">
        <p>📁 <b>Dataset</b><br>614 loan applications</p>
        <p>🤖 <b>Model</b><br>Random Forest (89.4% acc)</p>
        <p>📅 <b>Built by</b><br>SmartLoan ML System</p>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":

    st.markdown('<h1 class="page-title">Dashboard <span class="title-accent">Overview</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Real-time insights across all loan applications</p>', unsafe_allow_html=True)

    # ── KPI Metrics ─────────────────────────────────────────
    total      = len(df_raw)
    approved   = (df_raw["Loan_Status"] == "Y").sum()
    rejected   = (df_raw["Loan_Status"] == "N").sum()
    appr_rate  = approved / total * 100
    avg_loan   = df_raw["LoanAmount"].mean()
    med_income = df_raw["ApplicantIncome"].median()

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "Total Applications", f"{total:,}",  "📋", "#4F8EF7"),
        (c2, "Approved",           f"{approved:,}","✅", "#2ECC71"),
        (c3, "Rejected",           f"{rejected:,}","❌", "#E74C3C"),
        (c4, "Approval Rate",      f"{appr_rate:.1f}%","📈","#F39C12"),
        (c5, "Avg Loan Amount",    f"₹{avg_loan:.0f}K","💰","#9B59B6"),
    ]
    for col, label, value, icon, color in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid {color};">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Donut + Bar ───────────────────────────────────
    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<p class="chart-title">Loan Status Distribution</p>', unsafe_allow_html=True)
        status_counts = df_raw["Loan_Status"].value_counts()
        fig_donut = go.Figure(go.Pie(
            labels=["Approved", "Rejected"],
            values=[status_counts.get("Y", 0), status_counts.get("N", 0)],
            hole=0.62,
            marker=dict(colors=["#2ECC71", "#E74C3C"],
                        line=dict(color="#0D1117", width=3)),
            textinfo="percent+label",
            textfont_size=13,
        ))
        fig_donut.update_layout(
            height=300, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E0E6F0"),
            showlegend=False,
            annotations=[dict(text=f"<b>{appr_rate:.0f}%</b><br>Approved",
                              x=0.5, y=0.5, font_size=18, showarrow=False,
                              font=dict(color="#E0E6F0"))]
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<p class="chart-title">Approval Rate by Property Area</p>', unsafe_allow_html=True)
        area_stats = df_raw.groupby("Property_Area")["Loan_Status"].apply(
            lambda x: (x == "Y").mean() * 100).reset_index()
        area_stats.columns = ["Area", "Approval Rate (%)"]
        area_stats = area_stats.sort_values("Approval Rate (%)", ascending=True)
        fig_bar = px.bar(area_stats, x="Approval Rate (%)", y="Area", orientation="h",
                         color="Approval Rate (%)",
                         color_continuous_scale=["#E74C3C", "#F39C12", "#2ECC71"],
                         text=area_stats["Approval Rate (%)"].apply(lambda x: f"{x:.1f}%"))
        fig_bar.update_traces(textposition="outside", textfont_size=13)
        fig_bar.update_layout(
            height=300, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E0E6F0"), coloraxis_showscale=False,
            xaxis=dict(range=[0, 100], gridcolor="#1E2A3A"),
            yaxis=dict(gridcolor="#1E2A3A")
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 2: Loan Amount Dist + Credit History ─────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<p class="chart-title">Loan Amount Distribution</p>', unsafe_allow_html=True)
        fig_hist = px.histogram(df_raw.dropna(subset=["LoanAmount"]),
                                x="LoanAmount", nbins=35,
                                color="Loan_Status",
                                color_discrete_map={"Y": "#2ECC71", "N": "#E74C3C"},
                                barmode="overlay", opacity=0.75,
                                labels={"LoanAmount": "Loan Amount (₹K)", "Loan_Status": "Status"})
        fig_hist.update_layout(
            height=300, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E0E6F0"),
            xaxis=dict(gridcolor="#1E2A3A"), yaxis=dict(gridcolor="#1E2A3A"),
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<p class="chart-title">Credit History Impact</p>', unsafe_allow_html=True)
        ch = df_raw.dropna(subset=["Credit_History"])
        ch_group = ch.groupby(["Credit_History", "Loan_Status"]).size().reset_index(name="Count")
        ch_group["Credit_History"] = ch_group["Credit_History"].map({1.0: "Good Credit", 0.0: "No Credit"})
        fig_ch = px.bar(ch_group, x="Credit_History", y="Count", color="Loan_Status",
                        barmode="group",
                        color_discrete_map={"Y": "#2ECC71", "N": "#E74C3C"},
                        labels={"Credit_History": "Credit History", "Loan_Status": "Status"})
        fig_ch.update_layout(
            height=300, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E0E6F0"),
            xaxis=dict(gridcolor="#1E2A3A"), yaxis=dict(gridcolor="#1E2A3A"),
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_ch, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Quick Stats Row ─────────────────────────────────────
    st.markdown('<p class="chart-title" style="margin-top:1rem">Quick Statistics</p>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    stats = [
        (s1, "Median Loan",     f"₹{df_raw['LoanAmount'].median():.0f}K",   "💵"),
        (s2, "Max Loan",        f"₹{df_raw['LoanAmount'].max():.0f}K",       "📊"),
        (s3, "Graduate Apps",   f"{(df_raw['Education']=='Graduate').sum():,}","🎓"),
        (s4, "Good Credit",     f"{(df_raw['Credit_History']==1).sum():,}",   "⭐"),
    ]
    for col, label, val, icon in stats:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <span class="stat-icon">{icon}</span>
                <span class="stat-val">{val}</span>
                <span class="stat-lbl">{label}</span>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 2 — ANALYTICS
# ════════════════════════════════════════════════════════════
elif page == "📊  Analytics":

    st.markdown('<h1 class="page-title">Deep-Dive <span class="title-accent">Analytics</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Segment analysis across income, credit, and demographics</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["💰 Income", "📋 Credit History", "🏘 Property Area", "👥 Demographics"])

    # ── Tab 1: Income ────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<p class="chart-title">Applicant Income vs Loan Status</p>', unsafe_allow_html=True)
            fig = px.box(df_raw, x="Loan_Status", y="ApplicantIncome",
                         color="Loan_Status",
                         color_discrete_map={"Y": "#2ECC71", "N": "#E74C3C"},
                         labels={"Loan_Status": "Status", "ApplicantIncome": "Income (₹)"},
                         points="outliers")
            fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#E0E6F0"),
                              showlegend=False,
                              xaxis=dict(gridcolor="#1E2A3A"), yaxis=dict(gridcolor="#1E2A3A"))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<p class="chart-title">Coapplicant Income Distribution</p>', unsafe_allow_html=True)
            fig2 = px.histogram(df_raw, x="CoapplicantIncome", nbins=30,
                                color="Loan_Status",
                                color_discrete_map={"Y":"#2ECC71","N":"#E74C3C"},
                                barmode="overlay", opacity=0.75)
            fig2.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#E0E6F0"),
                               xaxis=dict(gridcolor="#1E2A3A"), yaxis=dict(gridcolor="#1E2A3A"),
                               legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<p class="chart-title">Income vs Loan Amount Scatter</p>', unsafe_allow_html=True)
        fig3 = px.scatter(df_raw.dropna(subset=["LoanAmount"]),
                          x="ApplicantIncome", y="LoanAmount",
                          color="Loan_Status",
                          color_discrete_map={"Y":"#2ECC71","N":"#E74C3C"},
                          opacity=0.6, trendline="lowess",
                          labels={"ApplicantIncome":"Applicant Income (₹)","LoanAmount":"Loan Amount (₹K)"})
        fig3.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#E0E6F0"),
                           xaxis=dict(gridcolor="#1E2A3A"), yaxis=dict(gridcolor="#1E2A3A"),
                           legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Tab 2: Credit ────────────────────────────────────────
    with tab2:
        ch = df_raw.dropna(subset=["Credit_History"]).copy()
        ch["Credit"] = ch["Credit_History"].map({1.0: "Good Credit ✅", 0.0: "No Credit ❌"})
        ch_appr = ch.groupby("Credit")["Loan_Status"].apply(
            lambda x: (x=="Y").mean()*100).reset_index()
        ch_appr.columns = ["Credit", "Approval Rate (%)"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<p class="chart-title">Approval Rate by Credit History</p>', unsafe_allow_html=True)
            fig = px.bar(ch_appr, x="Credit", y="Approval Rate (%)",
                         color="Approval Rate (%)",
                         color_continuous_scale=["#E74C3C","#F39C12","#2ECC71"],
                         text=ch_appr["Approval Rate (%)"].apply(lambda x: f"{x:.1f}%"))
            fig.update_traces(textposition="outside", textfont_size=16)
            fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#E0E6F0"),
                              coloraxis_showscale=False, showlegend=False,
                              xaxis=dict(gridcolor="#1E2A3A"), yaxis=dict(range=[0,105], gridcolor="#1E2A3A"))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<p class="chart-title">Credit History vs Loan Amount</p>', unsafe_allow_html=True)
            fig2 = px.box(ch, x="Credit", y="LoanAmount", color="Loan_Status",
                          color_discrete_map={"Y":"#2ECC71","N":"#E74C3C"})
            fig2.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#E0E6F0"),
                               xaxis=dict(gridcolor="#1E2A3A"), yaxis=dict(gridcolor="#1E2A3A"),
                               legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Insight callout
        st.markdown("""
        <div class="insight-box">
            <span class="insight-icon">💡</span>
            <div>
                <b>Key Finding:</b> Applicants with good credit history have an <b>~80% approval rate</b>
                vs only <b>~8% for those without</b>. Credit history is the single strongest predictor
                in the model — accounting for <b>22.4% of feature importance</b> in Random Forest.
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Tab 3: Property Area ─────────────────────────────────
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<p class="chart-title">Applications by Property Area</p>', unsafe_allow_html=True)
            area_cnt = df_raw["Property_Area"].value_counts().reset_index()
            area_cnt.columns = ["Area", "Count"]
            fig = px.pie(area_cnt, values="Count", names="Area",
                         color_discrete_sequence=["#4F8EF7","#2ECC71","#F39C12"],
                         hole=0.4)
            fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#E0E6F0"),
                              legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<p class="chart-title">Approval Rate by Area & Education</p>', unsafe_allow_html=True)
            grp = df_raw.groupby(["Property_Area","Education"])["Loan_Status"].apply(
                lambda x: (x=="Y").mean()*100).reset_index()
            grp.columns = ["Area","Education","Approval Rate (%)"]
            fig2 = px.bar(grp, x="Area", y="Approval Rate (%)", color="Education",
                          barmode="group",
                          color_discrete_sequence=["#4F8EF7","#F39C12"])
            fig2.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#E0E6F0"),
                               xaxis=dict(gridcolor="#1E2A3A"), yaxis=dict(gridcolor="#1E2A3A"),
                               legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Tab 4: Demographics ──────────────────────────────────
    with tab4:
        col1, col2, col3 = st.columns(3)
        for col, feat, title in [
            (col1, "Gender", "Gender Distribution"),
            (col2, "Married", "Marital Status"),
            (col3, "Education", "Education Level"),
        ]:
            with col:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown(f'<p class="chart-title">{title}</p>', unsafe_allow_html=True)
                grp = df_raw.groupby(feat)["Loan_Status"].apply(
                    lambda x: (x=="Y").mean()*100).reset_index()
                grp.columns = [feat, "Approval Rate (%)"]
                fig = px.bar(grp, x=feat, y="Approval Rate (%)",
                             color="Approval Rate (%)",
                             color_continuous_scale=["#E74C3C","#F39C12","#2ECC71"],
                             text=grp["Approval Rate (%)"].apply(lambda x: f"{x:.1f}%"))
                fig.update_traces(textposition="outside")
                fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#E0E6F0"),
                                  coloraxis_showscale=False,
                                  xaxis=dict(gridcolor="#1E2A3A"),
                                  yaxis=dict(range=[0,105], gridcolor="#1E2A3A"))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Dependents
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<p class="chart-title">Approval Rate by Dependents</p>', unsafe_allow_html=True)
        dep = df_raw.groupby("Dependents")["Loan_Status"].apply(
            lambda x: (x=="Y").mean()*100).reset_index()
        dep.columns = ["Dependents","Approval Rate (%)"]
        fig_dep = px.line(dep, x="Dependents", y="Approval Rate (%)",
                          markers=True, line_shape="spline",
                          color_discrete_sequence=["#4F8EF7"])
        fig_dep.update_traces(marker=dict(size=12, color="#F39C12"))
        fig_dep.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#E0E6F0"),
                              xaxis=dict(gridcolor="#1E2A3A"), yaxis=dict(gridcolor="#1E2A3A"))
        st.plotly_chart(fig_dep, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION TOOL
# ════════════════════════════════════════════════════════════
elif page == "🔮  Prediction Tool":

    st.markdown('<h1 class="page-title">Loan Approval <span class="title-accent">Predictor</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Enter applicant details to get an instant AI-powered decision</p>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown('<p class="section-label">👤 Personal Information</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            gender        = st.selectbox("Gender", ["Male", "Female"])
        with c2:
            married       = st.selectbox("Marital Status", ["Yes", "No"])
        with c3:
            dependents    = st.selectbox("Dependents", ["0", "1", "2", "3+"])

        c4, c5 = st.columns(2)
        with c4:
            education     = st.selectbox("Education", ["Graduate", "Not Graduate"])
        with c5:
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])

        st.markdown("---")
        st.markdown('<p class="section-label">💰 Financial Information</p>', unsafe_allow_html=True)
        c6, c7 = st.columns(2)
        with c6:
            applicant_income   = st.number_input("Applicant Income (₹/month)", min_value=0, value=5000, step=500)
        with c7:
            coapplicant_income = st.number_input("Coapplicant Income (₹/month)", min_value=0, value=0, step=500)

        c8, c9 = st.columns(2)
        with c8:
            loan_amount        = st.number_input("Loan Amount (₹ thousands)", min_value=1, value=150, step=10)
        with c9:
            loan_term          = st.selectbox("Loan Term (months)", [360, 180, 120, 240, 300, 480, 84, 60, 36, 12])

        st.markdown("---")
        st.markdown('<p class="section-label">🏡 Loan Details</p>', unsafe_allow_html=True)
        c10, c11 = st.columns(2)
        with c10:
            credit_history = st.selectbox("Credit History", [1, 0],
                                          format_func=lambda x: "Good (1)" if x == 1 else "No History (0)")
        with c11:
            property_area  = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        submitted = st.form_submit_button("🔮  Predict Loan Approval", use_container_width=True)

    # ── Prediction Logic ────────────────────────────────────
    if submitted:
        total_income     = applicant_income + coapplicant_income
        emi              = loan_amount / loan_term
        balance_income   = total_income - (emi * 1000)
        income_loan_ratio= total_income / (loan_amount + 1)

        input_dict = {
            "Gender"               : 1 if gender == "Male" else 0,
            "Married"              : 1 if married == "Yes" else 0,
            "Dependents"           : int(dependents.replace("3+", "3")),
            "Education"            : 0 if education == "Graduate" else 1,
            "Self_Employed"        : 1 if self_employed == "Yes" else 0,
            "Loan_Amount_Term"     : float(loan_term),
            "Credit_History"       : float(credit_history),
            "ApplicantIncome_log"  : np.log1p(applicant_income),
            "CoapplicantIncome_log": np.log1p(coapplicant_income),
            "TotalIncome_log"      : np.log1p(total_income),
            "LoanAmount_log"       : np.log1p(loan_amount),
            "EMI"                  : emi,
            "BalanceIncome"        : balance_income,
            "IncomeLoanRatio"      : income_loan_ratio,
            "Property_Area_Rural"  : 1 if property_area == "Rural" else 0,
            "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
            "Property_Area_Urban"  : 1 if property_area == "Urban" else 0,
        }

        input_df  = pd.DataFrame([input_dict])[FEATURE_COLS]
        pred      = model.predict(input_df)[0]
        prob      = model.predict_proba(input_df)[0][1]

        if prob >= 0.75:
            risk = "Low Risk"
            risk_color = "#2ECC71"
            risk_icon  = "🟢"
        elif prob >= 0.50:
            risk = "Medium Risk"
            risk_color = "#F39C12"
            risk_icon  = "🟡"
        else:
            risk = "High Risk"
            risk_color = "#E74C3C"
            risk_icon  = "🔴"

        approved = pred == 1
        result_color = "#2ECC71" if approved else "#E74C3C"
        result_text  = "APPROVED ✅" if approved else "REJECTED ❌"
        result_bg    = "rgba(46,204,113,0.1)" if approved else "rgba(231,76,60,0.1)"

        st.markdown(f"""
        <div class="result-card" style="border: 2px solid {result_color}; background: {result_bg};">
            <div class="result-verdict" style="color:{result_color};">{result_text}</div>
            <div class="result-grid">
                <div class="result-item">
                    <div class="result-item-val">{prob*100:.1f}%</div>
                    <div class="result-item-lbl">Approval Probability</div>
                </div>
                <div class="result-item">
                    <div class="result-item-val" style="color:{risk_color};">{risk_icon} {risk}</div>
                    <div class="result-item-lbl">Risk Category</div>
                </div>
                <div class="result-item">
                    <div class="result-item-val">₹{loan_amount}K</div>
                    <div class="result-item-lbl">Requested Amount</div>
                </div>
                <div class="result-item">
                    <div class="result-item-val">₹{emi:.1f}K</div>
                    <div class="result-item-lbl">Est. Monthly EMI</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability gauge
        st.markdown("<br>", unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 40, "color": "#E0E6F0"}},
            delta={"reference": 68.7, "suffix": "% vs avg"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#E0E6F0"},
                "bar": {"color": result_color},
                "steps": [
                    {"range": [0, 50],   "color": "rgba(231,76,60,0.2)"},
                    {"range": [50, 75],  "color": "rgba(243,156,18,0.2)"},
                    {"range": [75, 100], "color": "rgba(46,204,113,0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#4F8EF7", "width": 3},
                    "thickness": 0.75,
                    "value": 68.7
                }
            },
            title={"text": "Approval Probability Score", "font": {"color": "#E0E6F0", "size": 16}}
        ))
        fig_gauge.update_layout(
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E0E6F0"),
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Applicant summary
        with st.expander("📋 View Full Applicant Summary"):
            summary_df = pd.DataFrame({
                "Field": ["Gender", "Married", "Dependents", "Education", "Self Employed",
                          "Applicant Income", "Coapplicant Income", "Total Income",
                          "Loan Amount", "Loan Term", "Monthly EMI",
                          "Balance Income", "Credit History", "Property Area"],
                "Value": [gender, married, dependents, education, self_employed,
                          f"₹{applicant_income:,}", f"₹{coapplicant_income:,}", f"₹{total_income:,}",
                          f"₹{loan_amount}K", f"{loan_term} months", f"₹{emi:.2f}K",
                          f"₹{balance_income:,.0f}", "Good" if credit_history == 1 else "No History",
                          property_area]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
