import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="User Activity Pattern Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .block-container { padding: 1.5rem 2rem; }
  .metric-card {
    background: #1565C0;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    color: white;
    margin-bottom: 0.4rem;
  }
  .metric-value { font-size: 1.9rem; font-weight: 700; color: #00B0FF; }
  .metric-label { font-size: 0.75rem; color: #90CAF9; text-transform: uppercase; letter-spacing: 1px; }
  .section-title {
    font-size: 1rem; font-weight: 600; color: #1565C0;
    border-left: 4px solid #1565C0;
    padding-left: 0.6rem; margin: 1.2rem 0 0.8rem 0;
  }
  .risk-high { background:#fff0f0; border:1px solid #E53935; border-radius:8px; padding:0.7rem 1rem; margin:0.3rem 0; }
  .risk-med  { background:#fff8f0; border:1px solid #FF6F00; border-radius:8px; padding:0.7rem 1rem; margin:0.3rem 0; }
  .risk-ok   { background:#f0fff4; border:1px solid #00897B; border-radius:8px; padding:0.7rem 1rem; margin:0.3rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Load data ────────────────────────────────────────────────
@st.cache_data
def load():
    df = pd.read_csv("ecom_data_rfm.csv")
    df = df.drop(columns=[df.columns[0]])
    df = df[df["Customer_Segment"] != "NULL"].dropna(subset=["Customer_Segment"])
    df["Frequency"] = pd.to_numeric(df["Frequency"], errors="coerce")
    df["Recency"]   = pd.to_numeric(df["Recency"],   errors="coerce")
    df["Monetary"]  = pd.to_numeric(df["Monetary"],  errors="coerce")
    df = df.dropna(subset=["Frequency","Recency","Monetary"])

    df["activity_window"] = pd.cut(
        df["Recency"],
        bins=[0,30,90,180,270,999],
        labels=["Very Recent (0-30d)","Recent (31-90d)",
                "Moderate (91-180d)","Inactive (181-270d)","Dormant (270d+)"]
    )
    df["value_tier"] = pd.cut(
        df["Monetary"],
        bins=[-1,200,1000,5000,999999],
        labels=["Low Value","Mid Value","High Value","Premium"]
    )
    df["loyalty_pattern"] = df["Customer_Segment"].map(lambda s:
        "Loyal"    if s == "Loyal Customers" else
        "Growing"  if s in ["Potential Loyalist","New Customers","Promising"] else
        "At Risk"  if s in ["At Risk","About To Sleep","Need Attention"] else
        "Lost"
    )

    mean_f = df["Frequency"].mean(); std_f = df["Frequency"].std()
    mean_m = df["Monetary"].mean();  std_m = df["Monetary"].std()
    ft = mean_f + 2*std_f
    lt = mean_m - std_m

    def flag(r):
        if r["Recency"] > 300 and r["Frequency"] <= 2:
            return "HIGH RISK - Churned"
        if r["Frequency"] > ft and r["Monetary"] < lt:
            return "SUSPICIOUS - High Freq Low Spend"
        if r["Recency"] > 180 and r["Monetary"] > 5000:
            return "ALERT - High Value Going Inactive"
        if r["Monetary"] == 0:
            return "ALERT - Zero Spend"
        return "NORMAL"

    df["anomaly_flag"] = df.apply(flag, axis=1)
    return df

df = load()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Spark Dashboard")
    st.markdown("**User Activity Pattern Detection**")
    st.markdown("---")
    st.markdown("### Filters")

    seg_opts = sorted(df["Customer_Segment"].unique().tolist())
    sel_seg  = st.multiselect("Segment", seg_opts, default=seg_opts)

    cty_opts = sorted(df["Country"].unique().tolist())
    sel_cty  = st.multiselect("Country", cty_opts, default=cty_opts)

    tier_opts = ["Low Value","Mid Value","High Value","Premium"]
    sel_tier  = st.multiselect("Value Tier", tier_opts, default=tier_opts)

    st.markdown("---")
    st.markdown("### Raw Preview")
    fdf = df[
        df["Customer_Segment"].isin(sel_seg) &
        df["Country"].isin(sel_cty) &
        df["value_tier"].isin(sel_tier)
    ]
    st.dataframe(
        fdf[["CustomerID","Frequency","Recency","Monetary","Customer_Segment"]].head(10),
        use_container_width=True
    )

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<h1 style='color:#1565C0;font-size:1.7rem;margin-bottom:0'>
⚡ User Activity Pattern Detection
</h1>
<p style='color:#607D8B;margin-top:0.2rem;font-size:0.9rem'>
Apache Spark · PySpark · RFM Customer Behavior Analytics
</p>
""", unsafe_allow_html=True)
st.markdown("---")

# ── KPI Cards ────────────────────────────────────────────────
total     = len(fdf)
countries = fdf["Country"].nunique()
loyal     = len(fdf[fdf["Customer_Segment"] == "Loyal Customers"])
at_risk   = len(fdf[fdf["Customer_Segment"].isin(["At Risk","About To Sleep"])])
anomalies = len(fdf[fdf["anomaly_flag"] != "NORMAL"])
avg_spend = round(fdf["Monetary"].mean(), 2)

kpi_items = [
    (str(total),      "Total Customers"),
    (str(countries),  "Countries"),
    (f"£{avg_spend}", "Avg Spend"),
    (str(loyal),      "Loyal Customers"),
    (str(at_risk),    "At-Risk"),
    (f"{round(anomalies/total*100)}%", "Anomaly Rate"),
]
cols = st.columns(6)
for i, col in enumerate(cols):
    val, label = kpi_items[i][0], kpi_items[i][1]
    with col:
        st.markdown(f'''
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{val}</div>
        </div>''', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Analysis", "🔍 Pattern Detection", "🚨 Anomaly Detection", "📋 Raw Data"
])

COLORS = {
    "Loyal Customers":      "#00897B",
    "Potential Loyalist":   "#1565C0",
    "At Risk":              "#E53935",
    "About To Sleep":       "#FF6F00",
    "Lost Lowest":          "#B71C1C",
    "Need Attention":       "#F57C00",
    "New Customers":        "#7B1FA2",
    "Promising":            "#0288D1",
    "Recent High Spender":  "#2E7D32",
    "Champion":             "#00695C",
}

# ══ TAB 1 — ANALYSIS ══════════════════════════════════════════
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Customer Segment Distribution</div>', unsafe_allow_html=True)
        seg_df = fdf["Customer_Segment"].value_counts().reset_index()
        seg_df.columns = ["segment","count"]
        fig = px.bar(seg_df, x="count", y="segment", orientation="h",
                     color="segment", text="count",
                     color_discrete_map=COLORS)
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(t=10,b=10),
                          xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Segment Share (Pie)</div>', unsafe_allow_html=True)
        fig2 = px.pie(seg_df, names="segment", values="count",
                      color="segment", color_discrete_map=COLORS, hole=0.45)
        fig2.update_layout(margin=dict(t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-title">Activity Window Distribution</div>', unsafe_allow_html=True)
        act_df = fdf["activity_window"].value_counts().reset_index()
        act_df.columns = ["window","count"]
        order = ["Very Recent (0-30d)","Recent (31-90d)","Moderate (91-180d)",
                 "Inactive (181-270d)","Dormant (270d+)"]
        act_df["window"] = pd.Categorical(act_df["window"], categories=order, ordered=True)
        act_df = act_df.sort_values("window")
        fig3 = px.bar(act_df, x="window", y="count",
                      color="count", color_continuous_scale=["#1565C0","#E53935"],
                      text="count")
        fig3.update_traces(textposition="outside")
        fig3.update_layout(showlegend=False, margin=dict(t=10,b=10),
                           xaxis=dict(showgrid=False,title=""),
                           yaxis=dict(showgrid=False),
                           coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown('<div class="section-title">Top 10 Countries by Customers</div>', unsafe_allow_html=True)
        cty_df = fdf["Country"].value_counts().head(10).reset_index()
        cty_df.columns = ["country","count"]
        fig4 = px.bar(cty_df, x="count", y="country", orientation="h",
                      color="count", color_continuous_scale=["#B3E5FC","#0D47A1"],
                      text="count")
        fig4.update_traces(textposition="outside")
        fig4.update_layout(showlegend=False, margin=dict(t=10,b=10),
                           xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
                           coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    c5, c6 = st.columns(2)

    with c5:
        st.markdown('<div class="section-title">Monetary Distribution</div>', unsafe_allow_html=True)
        fig5 = px.histogram(fdf[fdf["Monetary"] < 5000], x="Monetary", nbins=40,
                            color_discrete_sequence=["#1565C0"])
        fig5.update_layout(margin=dict(t=10,b=10),
                           xaxis=dict(showgrid=False,title="Spend (£)"),
                           yaxis=dict(showgrid=False,title="Customers"))
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        st.markdown('<div class="section-title">Frequency Distribution</div>', unsafe_allow_html=True)
        fig6 = px.histogram(fdf[fdf["Frequency"] < 500], x="Frequency", nbins=40,
                            color_discrete_sequence=["#00897B"])
        fig6.update_layout(margin=dict(t=10,b=10),
                           xaxis=dict(showgrid=False,title="Purchase Frequency"),
                           yaxis=dict(showgrid=False,title="Customers"))
        st.plotly_chart(fig6, use_container_width=True)


# ══ TAB 2 — PATTERN DETECTION ═════════════════════════════════
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Loyalty Pattern Funnel</div>', unsafe_allow_html=True)
        loy_df = fdf["loyalty_pattern"].value_counts().reset_index()
        loy_df.columns = ["stage","count"]
        loy_order = ["Loyal","Growing","At Risk","Lost"]
        loy_df["stage"] = pd.Categorical(loy_df["stage"], categories=loy_order, ordered=True)
        loy_df = loy_df.sort_values("stage")
        fig_f = go.Figure(go.Funnel(
            y=loy_df["stage"], x=loy_df["count"],
            textinfo="value+percent initial",
            marker=dict(color=["#00897B","#1565C0","#FF6F00","#E53935"])
        ))
        fig_f.update_layout(margin=dict(t=10,b=10))
        st.plotly_chart(fig_f, use_container_width=True)

        m1, m2 = st.columns(2)
        loyal_n   = len(fdf[fdf["loyalty_pattern"] == "Loyal"])
        growing_n = len(fdf[fdf["loyalty_pattern"] == "Growing"])
        m1.metric("Loyal Rate",   f"{round(loyal_n/total*100)}%")
        m2.metric("Growing Rate", f"{round(growing_n/total*100)}%")

    with c2:
        st.markdown('<div class="section-title">Value Tier Breakdown</div>', unsafe_allow_html=True)
        tier_df = fdf["value_tier"].value_counts().reset_index()
        tier_df.columns = ["tier","count"]
        fig_t = px.pie(tier_df, names="tier", values="count", hole=0.45,
                       color="tier",
                       color_discrete_map={
                           "Premium":"#E53935","High Value":"#FF6F00",
                           "Mid Value":"#1565C0","Low Value":"#90CAF9"
                       })
        fig_t.update_layout(margin=dict(t=10,b=10))
        st.plotly_chart(fig_t, use_container_width=True)

    st.markdown('<div class="section-title">Recency vs Monetary (Bubble = Frequency)</div>', unsafe_allow_html=True)
    sample = fdf.sample(min(500, len(fdf)), random_state=42)
    fig_s = px.scatter(
        sample, x="Recency", y="Monetary",
        size="Frequency", color="Customer_Segment",
        color_discrete_map=COLORS,
        hover_data=["CustomerID","Country"],
        size_max=30,
        labels={"Recency":"Days Since Last Purchase","Monetary":"Total Spend (£)"}
    )
    fig_s.update_layout(margin=dict(t=10,b=10),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=False))
    st.plotly_chart(fig_s, use_container_width=True)

    st.markdown('<div class="section-title">Avg Spend per Segment</div>', unsafe_allow_html=True)
    spend_seg = fdf.groupby("Customer_Segment")["Monetary"].mean().round(2).reset_index()
    spend_seg.columns = ["segment","avg_spend"]
    spend_seg = spend_seg.sort_values("avg_spend", ascending=False)
    fig_sp = px.bar(spend_seg, x="segment", y="avg_spend",
                    color="segment", text="avg_spend",
                    color_discrete_map=COLORS)
    fig_sp.update_traces(textposition="outside", texttemplate="£%{text}")
    fig_sp.update_layout(showlegend=False, margin=dict(t=10,b=10),
                         xaxis=dict(showgrid=False,title=""),
                         yaxis=dict(showgrid=False,title="Avg Spend (£)"))
    st.plotly_chart(fig_sp, use_container_width=True)


# ══ TAB 3 — ANOMALY DETECTION ═════════════════════════════════
with tab3:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Anomaly Flag Distribution</div>', unsafe_allow_html=True)
        anm_df = fdf["anomaly_flag"].value_counts().reset_index()
        anm_df.columns = ["flag","count"]
        fig_a = px.pie(anm_df, names="flag", values="count", hole=0.45,
                       color="flag",
                       color_discrete_map={
                           "NORMAL":"#00897B",
                           "HIGH RISK - Churned":"#E53935",
                           "SUSPICIOUS - High Freq Low Spend":"#FF6F00",
                           "ALERT - High Value Going Inactive":"#7B1FA2",
                           "ALERT - Zero Spend":"#B71C1C"
                       })
        fig_a.update_layout(margin=dict(t=10,b=10))
        st.plotly_chart(fig_a, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Recency vs Frequency (Anomaly View)</div>', unsafe_allow_html=True)
        fig_rv = px.scatter(
            fdf.sample(min(600, len(fdf)), random_state=1),
            x="Recency", y="Frequency",
            color="anomaly_flag",
            color_discrete_map={
                "NORMAL":"#00897B",
                "HIGH RISK - Churned":"#E53935",
                "SUSPICIOUS - High Freq Low Spend":"#FF6F00",
                "ALERT - High Value Going Inactive":"#7B1FA2",
                "ALERT - Zero Spend":"#B71C1C"
            },
            hover_data=["CustomerID","Monetary"],
            labels={"Recency":"Days Since Purchase","Frequency":"Purchase Count"}
        )
        fig_rv.update_layout(margin=dict(t=10,b=10),
                             xaxis=dict(showgrid=False),
                             yaxis=dict(showgrid=False))
        st.plotly_chart(fig_rv, use_container_width=True)

    st.markdown('<div class="section-title">Anomaly Report — All Flagged Customers</div>', unsafe_allow_html=True)

    flagged_df = fdf[fdf["anomaly_flag"] != "NORMAL"].sort_values("anomaly_flag")

    for _, row in flagged_df.head(30).iterrows():
        flag = row["anomaly_flag"]
        if "HIGH RISK" in flag:
            card, icon, col = "risk-high", "🔴", "#E53935"
        elif "SUSPICIOUS" in flag or "ALERT" in flag:
            card, icon, col = "risk-med", "🟡", "#FF6F00"
        else:
            card, icon, col = "risk-ok", "🟢", "#00897B"

        st.markdown(f"""
        <div class="{card}">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-weight:600;font-size:0.9rem">{icon} Customer {int(row['CustomerID'])} — {row['Country']}</span>
            <span style="background:{col};color:white;padding:2px 10px;border-radius:20px;font-size:0.72rem;font-weight:600">{flag}</span>
          </div>
          <div style="display:flex;gap:1.5rem;margin-top:0.4rem;font-size:0.8rem;color:#555">
            <span>Frequency: <b>{int(row['Frequency'])}</b></span>
            <span>Recency: <b>{int(row['Recency'])} days</b></span>
            <span>Spend: <b>£{round(row['Monetary'],2)}</b></span>
            <span>Segment: <b>{row['Customer_Segment']}</b></span>
          </div>
        </div>""", unsafe_allow_html=True)

    if len(flagged_df) > 30:
        st.info(f"Showing 30 of {len(flagged_df)} flagged customers. Download full list below.")

    st.download_button(
        "⬇ Download All Flagged Customers CSV",
        data=flagged_df.to_csv(index=False),
        file_name="anomalous_customers.csv",
        mime="text/csv"
    )


# ══ TAB 4 — RAW DATA ══════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Full Dataset</div>', unsafe_allow_html=True)

    search = st.text_input("Search Customer ID", "")
    disp   = fdf.copy()
    if search:
        disp = disp[disp["CustomerID"].astype(str).str.contains(search)]

    st.dataframe(disp, use_container_width=True, height=400)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇ Download Filtered Data",
                           data=disp.to_csv(index=False),
                           file_name="filtered_rfm.csv", mime="text/csv")
    with c2:
        st.markdown(f"**Showing:** {len(disp)} of {len(df)} customers")

    st.markdown('<div class="section-title">Dataset Statistics</div>', unsafe_allow_html=True)
    st.dataframe(fdf[["Frequency","Recency","Monetary"]].describe().round(2),
                 use_container_width=True)