import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="PAN · Attrition Intelligence",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark sidebar */
[data-testid="stSidebar"] {
    background: #0a0f1e !important;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label { color: #94a3b8 !important; font-size: 0.78rem !important; letter-spacing: 0.08em; text-transform: uppercase; }

/* Main background */
.main { background: #f8fafc; }
.block-container { padding: 1.5rem 2.5rem 2rem; }

/* Page header */
.pan-header {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d2147 60%, #0f3460 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.pan-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(249,115,22,0.3) 0%, transparent 70%);
    border-radius: 50%;
}
.pan-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.9rem;
    color: #ffffff;
    margin: 0 0 0.25rem;
    letter-spacing: -0.02em;
}
.pan-header p { color: #94a3b8; margin: 0; font-size: 0.9rem; }
.pan-badge {
    display: inline-block;
    background: rgba(249,115,22,0.2);
    color: #fb923c;
    border: 1px solid rgba(249,115,22,0.4);
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    height: 100%;
}
.metric-card .metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.4rem;
}
.metric-card .metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.metric-card .metric-sub { font-size: 0.8rem; color: #64748b; }
.metric-red   .metric-value { color: #ef4444; }
.metric-amber .metric-value { color: #f59e0b; }
.metric-green .metric-value { color: #22c55e; }
.metric-blue  .metric-value { color: #3b82f6; }

/* Risk badge */
.risk-high   { background:#fee2e2; color:#dc2626; padding:3px 10px; border-radius:20px; font-weight:700; font-size:0.8rem; }
.risk-medium { background:#fef3c7; color:#d97706; padding:3px 10px; border-radius:20px; font-weight:700; font-size:0.8rem; }
.risk-low    { background:#dcfce7; color:#16a34a; padding:3px 10px; border-radius:20px; font-weight:700; font-size:0.8rem; }

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f172a;
    margin: 1.5rem 0 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #f1f5f9;
}

/* Profile card */
.profile-card {
    background: white;
    border-radius: 14px;
    padding: 1.5rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.profile-risk-bar {
    height: 10px;
    border-radius: 5px;
    background: #f1f5f9;
    overflow: hidden;
    margin: 0.5rem 0;
}

/* Sidebar nav */
.nav-item {
    padding: 0.6rem 1rem;
    border-radius: 8px;
    margin: 2px 0;
    cursor: pointer;
    font-size: 0.88rem;
    font-weight: 500;
}

/* Info box */
.info-box {
    background: #eff6ff;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #1e40af;
    margin: 0.75rem 0;
}

/* Scrollable table */
.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Load Data & Model ───────────────────────────────────
@st.cache_resource
def load_model():
    with open('attrition_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv('Employee_Attrition_Risk_Scores.csv')

def get_risk_label(prob):
    if prob >= 60:   return "High", "risk-high",   "🔴"
    elif prob >= 30: return "Medium", "risk-medium", "🟡"
    else:            return "Low", "risk-low",    "🟢"

try:
    bundle = load_model()
    model  = bundle['model']
    scaler = bundle['scaler']
    features = bundle['features']
    model_name = bundle.get('model_name', 'ML Model')
    df = load_data()
    data_loaded = True
except Exception as e:
    data_loaded = False
    err = str(e)

# ─── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 1.5rem;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.3rem; font-weight: 800; color: white; letter-spacing: -0.02em;'>
            🔵 PAN Intelligence
        </div>
        <div style='font-size: 0.75rem; color: #475569; margin-top: 2px;'>Attrition Risk Platform</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        ["📊  Risk Dashboard", "👤  Employee Profile", "🏢  Department View", "🔬  What-If Explorer"],
        label_visibility="visible"
    )

    st.markdown("---")

    if data_loaded:
        st.markdown("<div style='font-size:0.72rem; color:#475569; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;'>FILTERS</div>", unsafe_allow_html=True)
        dept_filter = st.multiselect(
            "Department",
            options=sorted(df['Department'].unique()),
            default=sorted(df['Department'].unique())
        )
        risk_filter = st.multiselect(
            "Risk Category",
            options=["🔴 High Risk", "🟡 Medium Risk", "🟢 Low Risk"],
            default=["🔴 High Risk", "🟡 Medium Risk", "🟢 Low Risk"]
        )
        st.markdown("---")
        st.markdown(f"<div style='font-size:0.72rem; color:#475569;'>Model: <b style='color:#94a3b8;'>{model_name}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.72rem; color:#475569;'>Records: <b style='color:#94a3b8;'>{len(df):,}</b></div>", unsafe_allow_html=True)

# ─── Apply Filters ───────────────────────────────────────
if data_loaded:
    filtered_df = df[
        df['Department'].isin(dept_filter) &
        df['Risk_Category'].isin(risk_filter)
    ].copy()
    high_risk_df   = df[df['Risk_Category'] == '🔴 High Risk']
    medium_risk_df = df[df['Risk_Category'] == '🟡 Medium Risk']
    low_risk_df    = df[df['Risk_Category'] == '🟢 Low Risk']

# ════════════════════════════════════════════════════════
# PAGE 1 — RISK DASHBOARD
# ════════════════════════════════════════════════════════
if page == "📊  Risk Dashboard":

    st.markdown("""
    <div class="pan-header">
        <div class="pan-badge">Live Analytics</div>
        <h1>Attrition Risk Dashboard</h1>
        <p>Real-time workforce risk intelligence powered by machine learning</p>
    </div>
    """, unsafe_allow_html=True)

    if not data_loaded:
        st.error(f"⚠️ Could not load files. Error: {err}")
        st.info("Make sure `attrition_model.pkl` and `Employee_Attrition_Risk_Scores.csv` are in the same folder as `app.py`")
        st.stop()

    # ── KPI Cards ──
    c1, c2, c3, c4 = st.columns(4)
    total = len(df)
    n_high   = len(high_risk_df)
    n_medium = len(medium_risk_df)
    n_low    = len(low_risk_df)
    actual_attrition = df['Attrition'].sum() if 'Attrition' in df.columns else "N/A"

    with c1:
        st.markdown(f"""
        <div class="metric-card metric-blue">
            <div class="metric-label">Total Employees</div>
            <div class="metric-value">{total:,}</div>
            <div class="metric-sub">Across all departments</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card metric-red">
            <div class="metric-label">🔴 High Risk</div>
            <div class="metric-value">{n_high}</div>
            <div class="metric-sub">{n_high/total*100:.1f}% of workforce</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card metric-amber">
            <div class="metric-label">🟡 Medium Risk</div>
            <div class="metric-value">{n_medium}</div>
            <div class="metric-sub">{n_medium/total*100:.1f}% of workforce</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card metric-green">
            <div class="metric-label">🟢 Low Risk</div>
            <div class="metric-value">{n_low}</div>
            <div class="metric-sub">{n_low/total*100:.1f}% of workforce</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Row 2: Donut + Histogram ──
    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.markdown("<div class='section-title'>Risk Distribution</div>", unsafe_allow_html=True)
        fig_donut = go.Figure(go.Pie(
            labels=['High Risk', 'Medium Risk', 'Low Risk'],
            values=[n_high, n_medium, n_low],
            hole=0.62,
            marker=dict(colors=['#ef4444', '#f59e0b', '#22c55e'],
                        line=dict(color='white', width=2)),
            textinfo='percent',
            textfont=dict(size=13, family='DM Sans'),
            hovertemplate="<b>%{label}</b><br>%{value} employees<br>%{percent}<extra></extra>"
        ))
        fig_donut.add_annotation(text=f"<b>{total}</b><br><span style='font-size:11px'>Employees</span>",
                                  x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig_donut.update_layout(
            height=280, margin=dict(t=10,b=10,l=10,r=10),
            legend=dict(orientation='h', y=-0.1, font=dict(size=11)),
            paper_bgcolor='white', plot_bgcolor='white', showlegend=True
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Attrition Probability Distribution</div>", unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df['Attrition_Prob'], nbinsx=40,
            marker=dict(
                color=df['Attrition_Prob'],
                colorscale=[[0,'#22c55e'],[0.3,'#22c55e'],[0.3,'#f59e0b'],[0.6,'#f59e0b'],[0.6,'#ef4444'],[1,'#ef4444']],
                line=dict(width=0.5, color='white')
            ),
            hovertemplate="Prob: %{x:.1f}%<br>Count: %{y}<extra></extra>"
        ))
        fig_hist.add_vline(x=30, line_dash="dash", line_color="#f59e0b", line_width=1.5,
                           annotation_text="30%", annotation_font_size=10)
        fig_hist.add_vline(x=60, line_dash="dash", line_color="#ef4444", line_width=1.5,
                           annotation_text="60%", annotation_font_size=10)
        fig_hist.update_layout(
            height=280, margin=dict(t=10,b=30,l=40,r=10),
            xaxis_title="Attrition Probability (%)",
            yaxis_title="Employee Count",
            paper_bgcolor='white', plot_bgcolor='#fafafa',
            xaxis=dict(gridcolor='#f1f5f9'), yaxis=dict(gridcolor='#f1f5f9'),
            font=dict(family='DM Sans', size=11)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Row 3: Risk by Dept + OverTime ──
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<div class='section-title'>Avg Risk by Department & Role</div>", unsafe_allow_html=True)
        dept_avg = df.groupby('Department')['Attrition_Prob'].agg(['mean','count']).reset_index()
        dept_avg.columns = ['Department', 'Avg_Risk', 'Count']
        dept_avg = dept_avg.sort_values('Avg_Risk', ascending=True)
        colors_dept = ['#ef4444' if v > 35 else '#f59e0b' if v > 20 else '#22c55e' for v in dept_avg['Avg_Risk']]
        fig_dept = go.Figure(go.Bar(
            x=dept_avg['Avg_Risk'], y=dept_avg['Department'],
            orientation='h',
            marker=dict(color=colors_dept, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in dept_avg['Avg_Risk']],
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>Avg Risk: %{x:.1f}%<extra></extra>"
        ))
        fig_dept.update_layout(
            height=250, margin=dict(t=5,b=20,l=20,r=60),
            xaxis_title="Avg Attrition Probability (%)", yaxis_title="",
            paper_bgcolor='white', plot_bgcolor='#fafafa',
            xaxis=dict(gridcolor='#f1f5f9', range=[0, dept_avg['Avg_Risk'].max()+10]),
            font=dict(family='DM Sans', size=11)
        )
        st.plotly_chart(fig_dept, use_container_width=True)

    with col4:
        st.markdown("<div class='section-title'>Risk by Overtime & Satisfaction</div>", unsafe_allow_html=True)
        ot_risk = df.groupby('OverTime')['Attrition_Prob'].mean().reset_index()
        fig_ot = go.Figure(go.Bar(
            x=ot_risk['OverTime'], y=ot_risk['Attrition_Prob'],
            marker=dict(color=['#22c55e','#ef4444'], line=dict(width=0)),
            text=[f"{v:.1f}%" for v in ot_risk['Attrition_Prob']],
            textposition='outside',
            width=0.4
        ))
        fig_ot.update_layout(
            height=250, margin=dict(t=5,b=20,l=40,r=10),
            xaxis_title="Overtime Status", yaxis_title="Avg Attrition Prob (%)",
            paper_bgcolor='white', plot_bgcolor='#fafafa',
            yaxis=dict(gridcolor='#f1f5f9', range=[0, ot_risk['Attrition_Prob'].max()+10]),
            font=dict(family='DM Sans', size=11)
        )
        st.plotly_chart(fig_ot, use_container_width=True)

    # ── High Risk Table ──
    st.markdown("<div class='section-title'>🔴 High-Risk Employees (Immediate Attention Required)</div>", unsafe_allow_html=True)
    show_cols = ['Employee_ID','Department','JobRole','MonthlyIncome','OverTime','JobSatisfaction',
                 'YearsSinceLastPromotion','Attrition_Prob','Risk_Category']
    available = [c for c in show_cols if c in high_risk_df.columns]
    display_df = high_risk_df[available].sort_values('Attrition_Prob', ascending=False).head(20)
    st.dataframe(display_df.style.background_gradient(
        subset=['Attrition_Prob'], cmap='RdYlGn_r'
    ).format({'Attrition_Prob': '{:.1f}%', 'MonthlyIncome': '${:,.0f}'}),
    use_container_width=True, height=320)

# ════════════════════════════════════════════════════════
# PAGE 2 — EMPLOYEE PROFILE
# ════════════════════════════════════════════════════════
elif page == "👤  Employee Profile":

    st.markdown("""
    <div class="pan-header">
        <div class="pan-badge">Individual Intelligence</div>
        <h1>Employee Risk Profile</h1>
        <p>Deep-dive attrition risk analysis for individual employees</p>
    </div>
    """, unsafe_allow_html=True)

    if not data_loaded:
        st.error("Could not load data files."); st.stop()

    col_search, col_rand = st.columns([3, 1])
    with col_search:
        emp_ids = df['Employee_ID'].tolist() if 'Employee_ID' in df.columns else [f"EMP_{str(i+1001).zfill(4)}" for i in range(len(df))]
        selected_id = st.selectbox("🔍 Select Employee ID", emp_ids, key="emp_selector")
    with col_rand:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        if st.button("🎲 Random High-Risk", use_container_width=True):
            selected_id = high_risk_df['Employee_ID'].sample(1).iloc[0]

    emp_row = df[df['Employee_ID'] == selected_id].iloc[0]
    prob  = emp_row['Attrition_Prob']
    risk_label, risk_class, risk_emoji = get_risk_label(prob)

    # ── Profile Header ──
    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns([1.5, 1.5, 1])
    with pc1:
        st.markdown(f"""
        <div class="profile-card">
            <div style="font-family:Syne,sans-serif; font-size:1.4rem; font-weight:800; color:#0f172a;">{selected_id}</div>
            <div style="color:#64748b; font-size:0.88rem; margin:2px 0 12px;">{emp_row.get('JobRole','N/A')} · {emp_row.get('Department','N/A')}</div>
            <div style="display:flex; gap:8px; flex-wrap:wrap;">
                <span class="{risk_class}">{risk_emoji} {risk_label} Risk</span>
            </div>
            <div style="margin-top:14px;">
                <div style="font-size:0.72rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em;">Attrition Probability</div>
                <div style="font-family:Syne,sans-serif; font-size:2.5rem; font-weight:800; color:{'#ef4444' if prob>=60 else '#f59e0b' if prob>=30 else '#22c55e'};">{prob:.1f}%</div>
                <div class="profile-risk-bar">
                    <div style="height:100%; width:{min(prob,100)}%; background:{'#ef4444' if prob>=60 else '#f59e0b' if prob>=30 else '#22c55e'}; border-radius:5px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with pc2:
        st.markdown(f"""
        <div class="profile-card">
            <div style="font-size:0.72rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.75rem;">Employee Details</div>
            <table style="width:100%; font-size:0.85rem; border-collapse:collapse;">
                <tr><td style="color:#64748b; padding:3px 0;">Age</td><td style="font-weight:600; text-align:right;">{int(emp_row.get('Age', 0))} yrs</td></tr>
                <tr><td style="color:#64748b; padding:3px 0;">Monthly Income</td><td style="font-weight:600; text-align:right;">${int(emp_row.get('MonthlyIncome', 0)):,}</td></tr>
                <tr><td style="color:#64748b; padding:3px 0;">Years at Company</td><td style="font-weight:600; text-align:right;">{int(emp_row.get('YearsAtCompany', 0))} yrs</td></tr>
                <tr><td style="color:#64748b; padding:3px 0;">Since Promotion</td><td style="font-weight:600; text-align:right;">{int(emp_row.get('YearsSinceLastPromotion', 0))} yrs</td></tr>
                <tr><td style="color:#64748b; padding:3px 0;">Overtime</td><td style="font-weight:600; text-align:right;">{emp_row.get('OverTime','N/A')}</td></tr>
                <tr><td style="color:#64748b; padding:3px 0;">Marital Status</td><td style="font-weight:600; text-align:right;">{emp_row.get('MaritalStatus','N/A')}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with pc3:
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={'suffix': '%', 'font': {'size': 28, 'family': 'Syne', 'color': '#0f172a'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#94a3b8', 'tickfont': {'size': 9}},
                'bar': {'color': '#ef4444' if prob >= 60 else '#f59e0b' if prob >= 30 else '#22c55e', 'thickness': 0.25},
                'bgcolor': 'white',
                'borderwidth': 0,
                'steps': [
                    {'range': [0,30],   'color': '#dcfce7'},
                    {'range': [30,60],  'color': '#fef3c7'},
                    {'range': [60,100], 'color': '#fee2e2'}
                ],
                'threshold': {'line': {'color': '#0f172a', 'width': 2}, 'value': prob}
            }
        ))
        fig_gauge.update_layout(
            height=210, margin=dict(t=20,b=0,l=20,r=20),
            paper_bgcolor='white', font=dict(family='DM Sans')
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Satisfaction Radar ──
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    rc1, rc2 = st.columns(2)

    with rc1:
        st.markdown("<div class='section-title'>Satisfaction Scores</div>", unsafe_allow_html=True)
        sat_fields = ['JobSatisfaction','EnvironmentSatisfaction','RelationshipSatisfaction','WorkLifeBalance','JobInvolvement']
        sat_labels = ['Job Sat.','Env Sat.','Rel Sat.','Work-Life','Job Inv.']
        sat_vals_emp = [emp_row.get(f, 2) for f in sat_fields]
        sat_vals_avg = [df[f].mean() for f in sat_fields]
        sat_vals_emp_c = sat_vals_emp + [sat_vals_emp[0]]
        sat_vals_avg_c = sat_vals_avg + [sat_vals_avg[0]]
        sat_labels_c   = sat_labels + [sat_labels[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=sat_vals_avg_c, theta=sat_labels_c, fill='toself',
            name='Company Avg', line=dict(color='#3b82f6', width=2),
            fillcolor='rgba(59,130,246,0.1)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=sat_vals_emp_c, theta=sat_labels_c, fill='toself',
            name=selected_id, line=dict(color='#ef4444', width=2),
            fillcolor='rgba(239,68,68,0.15)'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,4],
                                       tickfont=dict(size=9), gridcolor='#e2e8f0'),
                       angularaxis=dict(tickfont=dict(size=10))),
            showlegend=True, height=280,
            margin=dict(t=20,b=20,l=40,r=40),
            legend=dict(orientation='h', y=-0.15, font=dict(size=10)),
            paper_bgcolor='white'
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with rc2:
        st.markdown("<div class='section-title'>⚠️ Risk Factor Analysis</div>", unsafe_allow_html=True)
        reasons = []
        if emp_row.get('OverTime') == 'Yes':                        reasons.append(('❗ Working Overtime', 'high'))
        if emp_row.get('JobSatisfaction', 4) <= 2:                  reasons.append(('😞 Low Job Satisfaction', 'high'))
        if emp_row.get('EnvironmentSatisfaction', 4) <= 2:          reasons.append(('🏢 Poor Work Environment', 'high'))
        if emp_row.get('WorkLifeBalance', 4) <= 2:                  reasons.append(('⚖️ Poor Work-Life Balance', 'medium'))
        if emp_row.get('YearsSinceLastPromotion', 0) >= 4:          reasons.append((f'📉 No promotion in {int(emp_row.get("YearsSinceLastPromotion",0))} yrs', 'medium'))
        if emp_row.get('MonthlyIncome', 9999) < 3000:               reasons.append(('💰 Below-average Salary', 'medium'))
        if emp_row.get('NumCompaniesWorked', 0) >= 5:               reasons.append(('🔄 High Job Hopper', 'medium'))
        if emp_row.get('BusinessTravel') == 'Travel_Frequently':    reasons.append(('✈️ Frequent Business Travel', 'low'))
        if emp_row.get('StockOptionLevel', 1) == 0:                 reasons.append(('📊 No Stock Options', 'low'))
        if emp_row.get('MaritalStatus') == 'Single':                reasons.append(('👤 Single Status', 'low'))

        if reasons:
            for reason, level in reasons[:6]:
                color = '#fee2e2' if level=='high' else '#fef3c7' if level=='medium' else '#f0fdf4'
                border = '#ef4444' if level=='high' else '#f59e0b' if level=='medium' else '#22c55e'
                st.markdown(f"""
                <div style="background:{color}; border-left:3px solid {border}; border-radius:0 8px 8px 0;
                            padding:0.5rem 0.8rem; margin:4px 0; font-size:0.85rem; font-weight:500;">
                    {reason}
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#dcfce7; border-left:3px solid #22c55e; border-radius:0 8px 8px 0;
                        padding:0.75rem 1rem; font-size:0.88rem; font-weight:500; color:#166534;">
                ✅ No significant risk factors detected for this employee.
            </div>""", unsafe_allow_html=True)

        if 'Attrition' in emp_row.index:
            actual = "Left ❌" if emp_row['Attrition'] == 1 else "Stayed ✅"
            st.markdown(f"""
            <div class="info-box" style="margin-top:0.75rem;">
                📌 <b>Actual Outcome:</b> {actual}
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# PAGE 3 — DEPARTMENT VIEW
# ════════════════════════════════════════════════════════
elif page == "🏢  Department View":

    st.markdown("""
    <div class="pan-header">
        <div class="pan-badge">Department Analytics</div>
        <h1>Department-Level Risk View</h1>
        <p>Aggregated attrition risk intelligence across teams and roles</p>
    </div>
    """, unsafe_allow_html=True)

    if not data_loaded:
        st.error("Could not load data files."); st.stop()

    dept_sel = st.selectbox("Select Department", sorted(df['Department'].unique()))
    dept_data = df[df['Department'] == dept_sel]

    # Department KPIs
    d1, d2, d3, d4 = st.columns(4)
    d_total   = len(dept_data)
    d_high    = len(dept_data[dept_data['Risk_Category'] == '🔴 High Risk'])
    d_avg_prob = dept_data['Attrition_Prob'].mean()
    d_ot_pct  = (dept_data['OverTime'] == 'Yes').mean() * 100 if 'OverTime' in dept_data.columns else 0

    with d1:
        st.markdown(f"""<div class="metric-card metric-blue">
            <div class="metric-label">Dept Employees</div>
            <div class="metric-value">{d_total}</div>
            <div class="metric-sub">{dept_sel}</div>
        </div>""", unsafe_allow_html=True)
    with d2:
        st.markdown(f"""<div class="metric-card metric-red">
            <div class="metric-label">High Risk Count</div>
            <div class="metric-value">{d_high}</div>
            <div class="metric-sub">{d_high/d_total*100:.1f}% of dept</div>
        </div>""", unsafe_allow_html=True)
    with d3:
        st.markdown(f"""<div class="metric-card metric-amber">
            <div class="metric-label">Avg Attrition Risk</div>
            <div class="metric-value">{d_avg_prob:.1f}%</div>
            <div class="metric-sub">Department average</div>
        </div>""", unsafe_allow_html=True)
    with d4:
        st.markdown(f"""<div class="metric-card {'metric-red' if d_ot_pct>40 else 'metric-green'}">
            <div class="metric-label">Overtime Rate</div>
            <div class="metric-value">{d_ot_pct:.1f}%</div>
            <div class="metric-sub">Working overtime</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Risk by Job Role
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='section-title'>Risk by Job Role</div>", unsafe_allow_html=True)
        role_stats = dept_data.groupby('JobRole').agg(
            Avg_Risk=('Attrition_Prob','mean'),
            Count=('Attrition_Prob','count'),
            High_Risk=('Risk_Category', lambda x: (x=='🔴 High Risk').sum())
        ).reset_index().sort_values('Avg_Risk', ascending=True)

        fig_role = go.Figure(go.Bar(
            x=role_stats['Avg_Risk'], y=role_stats['JobRole'],
            orientation='h',
            marker=dict(color=['#ef4444' if v > 35 else '#f59e0b' if v > 20 else '#22c55e'
                               for v in role_stats['Avg_Risk']]),
            text=[f"{v:.1f}% ({c} emp)" for v, c in zip(role_stats['Avg_Risk'], role_stats['Count'])],
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>Avg Risk: %{x:.1f}%<extra></extra>"
        ))
        fig_role.update_layout(
            height=300, margin=dict(t=5,b=20,l=20,r=80),
            xaxis_title="Avg Attrition Probability (%)", yaxis_title="",
            paper_bgcolor='white', plot_bgcolor='#fafafa',
            xaxis=dict(gridcolor='#f1f5f9', range=[0, role_stats['Avg_Risk'].max()+15]),
            font=dict(family='DM Sans', size=11)
        )
        st.plotly_chart(fig_role, use_container_width=True)

    with col_b:
        st.markdown("<div class='section-title'>Risk Category Breakdown</div>", unsafe_allow_html=True)
        risk_counts = dept_data['Risk_Category'].value_counts().reset_index()
        risk_counts.columns = ['Risk','Count']
        color_map = {'🔴 High Risk': '#ef4444', '🟡 Medium Risk': '#f59e0b', '🟢 Low Risk': '#22c55e'}
        fig_bar = go.Figure(go.Bar(
            x=risk_counts['Risk'], y=risk_counts['Count'],
            marker=dict(color=[color_map.get(r,'#94a3b8') for r in risk_counts['Risk']],
                        line=dict(width=0)),
            text=risk_counts['Count'], textposition='outside',
            width=0.5
        ))
        fig_bar.update_layout(
            height=300, margin=dict(t=5,b=20,l=40,r=10),
            xaxis_title="Risk Category", yaxis_title="Number of Employees",
            paper_bgcolor='white', plot_bgcolor='#fafafa',
            yaxis=dict(gridcolor='#f1f5f9'),
            font=dict(family='DM Sans', size=11)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Scatter — Income vs Risk
    st.markdown("<div class='section-title'>Income vs Attrition Probability</div>", unsafe_allow_html=True)
    fig_scatter = px.scatter(
        dept_data, x='MonthlyIncome', y='Attrition_Prob',
        color='Risk_Category',
        color_discrete_map={'🔴 High Risk':'#ef4444','🟡 Medium Risk':'#f59e0b','🟢 Low Risk':'#22c55e'},
        size_max=12, opacity=0.7,
        hover_data=['JobRole','OverTime','YearsAtCompany'] if 'OverTime' in dept_data.columns else ['JobRole'],
        labels={'MonthlyIncome':'Monthly Income ($)','Attrition_Prob':'Attrition Probability (%)'}
    )
    fig_scatter.update_layout(
        height=350, paper_bgcolor='white', plot_bgcolor='#fafafa',
        xaxis=dict(gridcolor='#f1f5f9'), yaxis=dict(gridcolor='#f1f5f9'),
        font=dict(family='DM Sans', size=11),
        margin=dict(t=10,b=40,l=40,r=10)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Full dept table
    st.markdown("<div class='section-title'>All Employees in Department</div>", unsafe_allow_html=True)
    show_cols = ['Employee_ID','JobRole','MonthlyIncome','OverTime','JobSatisfaction',
                 'YearsAtCompany','YearsSinceLastPromotion','Attrition_Prob','Risk_Category']
    available = [c for c in show_cols if c in dept_data.columns]
    st.dataframe(
        dept_data[available].sort_values('Attrition_Prob', ascending=False)
        .style.background_gradient(subset=['Attrition_Prob'], cmap='RdYlGn_r')
        .format({'Attrition_Prob': '{:.1f}%', 'MonthlyIncome': '${:,.0f}'}),
        use_container_width=True, height=320
    )

# ════════════════════════════════════════════════════════
# PAGE 4 — WHAT-IF EXPLORER
# ════════════════════════════════════════════════════════
elif page == "🔬  What-If Explorer":

    st.markdown("""
    <div class="pan-header">
        <div class="pan-badge">Scenario Simulation</div>
        <h1>What-If Scenario Explorer</h1>
        <p>Simulate HR interventions and see how they impact attrition probability</p>
    </div>
    """, unsafe_allow_html=True)

    if not data_loaded:
        st.error("Could not load data files."); st.stop()

    st.markdown("""
    <div class="info-box">
        💡 <b>How it works:</b> Select an employee, adjust their attributes using the sliders below, and instantly see how HR interventions (salary raise, reducing overtime, etc.) would change their attrition probability.
    </div>
    """, unsafe_allow_html=True)

    selected_what_if = st.selectbox("Select Employee to Simulate", df['Employee_ID'].tolist(), key="whatif_sel")
    base_row = df[df['Employee_ID'] == selected_what_if].iloc[0]
    base_prob = base_row['Attrition_Prob']

    st.markdown("<div class='section-title'>🎛️ Adjust Employee Attributes</div>", unsafe_allow_html=True)

    w1, w2, w3 = st.columns(3)
    with w1:
        new_income = st.slider("Monthly Income ($)", 1000, 20000,
                               int(base_row.get('MonthlyIncome', 5000)), step=500)
        new_overtime = st.selectbox("Overtime", ["No", "Yes"],
                                    index=0 if base_row.get('OverTime') == 'No' else 1)
    with w2:
        new_job_sat = st.slider("Job Satisfaction (1–4)", 1, 4,
                                int(base_row.get('JobSatisfaction', 2)))
        new_env_sat = st.slider("Environment Satisfaction (1–4)", 1, 4,
                                int(base_row.get('EnvironmentSatisfaction', 2)))
    with w3:
        new_wlb = st.slider("Work-Life Balance (1–4)", 1, 4,
                             int(base_row.get('WorkLifeBalance', 2)))
        new_promo = st.slider("Years Since Last Promotion", 0, 15,
                              int(base_row.get('YearsSinceLastPromotion', 2)))

    # Build modified feature vector
    try:
        sim_df = df.drop(columns=['Employee_ID','Attrition_Prob','Risk_Category'], errors='ignore')
        if 'Attrition' in sim_df.columns:
            sim_df = sim_df.drop(columns=['Attrition'])

        # Encode categories (same logic as notebook)
        cat_enc = sim_df.copy()
        binary_map = {'Yes': 1, 'No': 0}
        travel_map  = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
        gender_map  = {'Male': 1, 'Female': 0}
        marital_map = {'Single': 0, 'Married': 1, 'Divorced': 2}

        if 'OverTime' in cat_enc.columns:      cat_enc['OverTime']       = cat_enc['OverTime'].map(binary_map)
        if 'BusinessTravel' in cat_enc.columns: cat_enc['BusinessTravel'] = cat_enc['BusinessTravel'].map(travel_map)
        if 'Gender' in cat_enc.columns:         cat_enc['Gender']         = cat_enc['Gender'].map(gender_map)
        if 'MaritalStatus' in cat_enc.columns:  cat_enc['MaritalStatus']  = cat_enc['MaritalStatus'].map(marital_map)
        ohe_cols = [c for c in ['Department','EducationField','JobRole'] if c in cat_enc.columns]
        cat_enc = pd.get_dummies(cat_enc, columns=ohe_cols, drop_first=True)
        bool_cols_enc = cat_enc.select_dtypes(include='bool').columns
        cat_enc[bool_cols_enc] = cat_enc[bool_cols_enc].astype(int)

        # Align with trained features
        for col in features:
            if col not in cat_enc.columns:
                cat_enc[col] = 0
        cat_enc = cat_enc[features]

        # Get original row index
        idx = df[df['Employee_ID'] == selected_what_if].index[0]
        sim_row = cat_enc.iloc[[idx]].copy()

        # Apply changes
        col_map = {
            'MonthlyIncome': new_income,
            'OverTime': 1 if new_overtime == 'Yes' else 0,
            'JobSatisfaction': new_job_sat,
            'EnvironmentSatisfaction': new_env_sat,
            'WorkLifeBalance': new_wlb,
            'YearsSinceLastPromotion': new_promo
        }
        for col, val in col_map.items():
            if col in sim_row.columns:
                sim_row[col] = val

        sim_scaled = scaler.transform(sim_row)
        new_prob = model.predict_proba(sim_scaled)[0][1] * 100
        delta    = new_prob - base_prob
        new_risk, new_risk_class, new_risk_emoji = get_risk_label(new_prob)

        # ── Results ──
        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown(f"""
            <div class="metric-card metric-blue">
                <div class="metric-label">Baseline Risk</div>
                <div class="metric-value" style="color:#64748b;">{base_prob:.1f}%</div>
                <div class="metric-sub">Original probability</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="metric-card {'metric-red' if new_prob>=60 else 'metric-amber' if new_prob>=30 else 'metric-green'}">
                <div class="metric-label">Simulated Risk</div>
                <div class="metric-value">{new_prob:.1f}%</div>
                <div class="metric-sub">{new_risk_emoji} {new_risk} Risk</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            delta_color = 'metric-green' if delta < 0 else 'metric-red'
            delta_icon  = '📉' if delta < 0 else '📈'
            st.markdown(f"""
            <div class="metric-card {delta_color}">
                <div class="metric-label">Risk Change</div>
                <div class="metric-value">{delta:+.1f}%</div>
                <div class="metric-sub">{delta_icon} {'Improvement' if delta < 0 else 'Deterioration'}</div>
            </div>""", unsafe_allow_html=True)

        # Before/After bar chart
        st.markdown("<div class='section-title'>Before vs After Intervention</div>", unsafe_allow_html=True)
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            name='Baseline', x=['Attrition Probability'], y=[base_prob],
            marker=dict(color='#94a3b8'), width=0.25,
            text=[f"{base_prob:.1f}%"], textposition='outside'
        ))
        fig_compare.add_trace(go.Bar(
            name='After Intervention', x=['Attrition Probability'], y=[new_prob],
            marker=dict(color='#22c55e' if new_prob < base_prob else '#ef4444'), width=0.25,
            text=[f"{new_prob:.1f}%"], textposition='outside'
        ))
        fig_compare.add_hline(y=30, line_dash="dot", line_color="#f59e0b",
                              annotation_text="Medium Risk Threshold (30%)")
        fig_compare.add_hline(y=60, line_dash="dot", line_color="#ef4444",
                              annotation_text="High Risk Threshold (60%)")
        fig_compare.update_layout(
            height=320, barmode='group',
            yaxis=dict(range=[0, 105], title="Attrition Probability (%)", gridcolor='#f1f5f9'),
            xaxis=dict(title=""),
            paper_bgcolor='white', plot_bgcolor='#fafafa',
            legend=dict(orientation='h', y=-0.15),
            font=dict(family='DM Sans', size=12),
            margin=dict(t=20,b=40,l=60,r=20)
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        if delta < -5:
            st.success(f"✅ Intervention would reduce attrition risk by **{abs(delta):.1f} percentage points**. This employee's risk would change from **{get_risk_label(base_prob)[0]} Risk** → **{new_risk} Risk**.")
        elif delta > 5:
            st.error(f"⚠️ These changes would **increase** attrition risk by {abs(delta):.1f} percentage points.")
        else:
            st.info("ℹ️ Minimal change in risk probability. Consider stronger interventions.")

    except Exception as e:
        st.warning(f"Simulation error: {e}. Make sure the model features match the dataset columns.")
