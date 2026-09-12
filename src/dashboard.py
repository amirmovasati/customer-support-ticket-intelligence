# region: Setup and Imports
import sys
from pathlib import Path
import streamlit as st
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from pipeline import load_models, process_ticket
from database import SessionLocal, Ticket, init_db
from reports import ticket_count_by_category
# endregion


# region: Page Config and Custom Styling
# Dark, professional theme with a deliberate palette — not Streamlit's default.

st.set_page_config(page_title="Support Ticket Intelligence", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #0F172A; }
h1 { color: #F1F5F9 !important; font-weight: 700 !important; margin-bottom: 0.2rem !important; }
h3 { color: #F1F5F9 !important; }
p, span, label { color: #CBD5E1; }
.subtitle { color: #94A3B8; font-size: 1rem; margin-bottom: 1.5rem; }

.metric-card {
    background-color: #1E293B; border-radius: 10px; padding: 1rem 1.2rem;
    border: 1px solid #334155;
}
.metric-value { font-size: 1.7rem; font-weight: 700; color: #818CF8; line-height: 1.2; }
.metric-label { font-size: 0.8rem; color: #E2E8F0; font-weight: 600; margin-top: 0.3rem; }
.metric-note { font-size: 0.72rem; color: #64748B; margin-top: 0.15rem; }

.result-card {
    background-color: #1E293B; border-radius: 10px; padding: 1.3rem;
    border: 1px solid #334155;
}
.result-label { color: #94A3B8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; }
.result-value { color: #F1F5F9; font-size: 1rem; margin-top: 0.1rem; margin-bottom: 0.9rem; }
.priority-badge {
    display: inline-block; padding: 0.25rem 0.8rem; border-radius: 20px;
    font-weight: 600; font-size: 0.8rem; color: #0F172A;
}
.section-note { color: #64748B; font-size: 0.85rem; margin-top: -0.6rem; margin-bottom: 0.8rem; }

div.stButton > button {
    background-color: #4F46E5; color: white; border-radius: 8px;
    border: none; padding: 0.55rem 1.2rem; font-weight: 600; width: 100%;
}
div.stButton > button:hover { background-color: #4338CA; }

.stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
    background-color: #1E293B; color: #F1F5F9; border-color: #334155;
}
</style>
""", unsafe_allow_html=True)

PRIORITY_COLORS = {"High": "#F87171", "Medium": "#FBBF24", "Low": "#34D399"}
# endregion


# region: Load Models (cached once per app session)
@st.cache_resource
def get_models():
    return load_models()

models = get_models()
init_db()
# endregion


# region: Header
st.markdown('<div style="height:4px; width:60px; background-color:#F97316; border-radius:2px; margin-bottom:0.8rem;"></div>', unsafe_allow_html=True)
st.title("Customer Support Ticket Intelligence")
st.markdown('<div class="subtitle">AI-assisted classification, prioritization, and response drafting for support tickets.</div>', unsafe_allow_html=True)
# endregion


# region: Top Metrics
session = SessionLocal()
try:
    total_tickets = session.query(Ticket).count()
    high_count = session.query(Ticket).filter(Ticket.priority == "High").count()
    medium_count = session.query(Ticket).filter(Ticket.priority == "Medium").count()
finally:
    session.close()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'''<div class="metric-card">
        <div class="metric-value">{total_tickets:,}</div>
        <div class="metric-label">Total Tickets</div>
        <div class="metric-note">Live count from the database</div>
    </div>''', unsafe_allow_html=True)
with col2:
    st.markdown(f'''<div class="metric-card">
        <div class="metric-value">{high_count:,}</div>
        <div class="metric-label">High Priority</div>
        <div class="metric-note">Live count from the database</div>
    </div>''', unsafe_allow_html=True)
with col3:
    st.markdown(f'''<div class="metric-card">
        <div class="metric-value">{medium_count:,}</div>
        <div class="metric-label">Medium Priority</div>
        <div class="metric-note">Live count from the database</div>
    </div>''', unsafe_allow_html=True)

st.write("")
# endregion


# region: Ticket Submission and Result
left_col, right_col = st.columns([1, 1.2])

with left_col:
    st.subheader("Submit a Ticket")
    st.markdown('<div class="section-note">Enter a customer message as it would arrive from a support form or email.</div>', unsafe_allow_html=True)
    instruction = st.text_area("Customer message", placeholder="e.g. I want to cancel my order, it's taking too long", height=120, label_visibility="collapsed")
    category = st.selectbox("Category (optional)", ["UNKNOWN", "ACCOUNT", "ORDER", "REFUND", "PAYMENT", "SHIPPING", "DELIVERY", "INVOICE", "CONTACT", "FEEDBACK", "SUBSCRIPTION", "CANCEL"])
    submitted = st.button("Analyze Ticket")

with right_col:
    st.subheader("Result")
    if submitted and instruction.strip():
        with st.spinner("Analyzing ticket..."):
            result = process_ticket(instruction, category, models)
            ticket_session = SessionLocal()
            try:
                ticket = Ticket(
                    instruction=instruction, category=category,
                    intent=result["intent"], priority=result["priority"],
                    draft_response=result["draft_response"],
                )
                ticket_session.add(ticket)
                ticket_session.commit()
            finally:
                ticket_session.close()

        priority_color = PRIORITY_COLORS[result["priority"]]
        st.markdown(f'''<div class="result-card">
            <div class="result-label">Intent</div>
            <div class="result-value">{result['intent']}</div>
            <div class="result-label">Priority</div>
            <div class="result-value"><span class="priority-badge" style="background-color:{priority_color}">{result['priority']}</span></div>
            <div class="result-label">Suggested Response</div>
            <div class="result-value">{result['draft_response']}</div>
        </div>''', unsafe_allow_html=True)
    elif submitted:
        st.warning("Please enter a customer message first.")
    else:
        st.markdown('<div class="section-note">Submit a ticket on the left to see the AI-generated analysis here.</div>', unsafe_allow_html=True)
# endregion


# region: Category Distribution Chart
st.write("")
st.subheader("Ticket Volume by Category")
st.markdown('<div class="section-note">Total tickets processed so far, grouped by category — including the historical dataset and any tickets submitted through this dashboard.</div>', unsafe_allow_html=True)

category_data = ticket_count_by_category()
categories = [row.category for row in category_data]
counts = [row.ticket_count for row in category_data]

fig = px.bar(x=categories, y=counts)
fig.update_traces(marker_color="#818CF8", marker_line_width=0, opacity=0.9)
fig.update_layout(
    plot_bgcolor="#1E293B",
    paper_bgcolor="#1E293B",
    font_color="#CBD5E1",
    xaxis=dict(title=None, showgrid=False, linecolor="#334155"),
    yaxis=dict(title=None, showgrid=True, gridcolor="#334155", zeroline=False),
    margin=dict(l=10, r=10, t=10, b=10),
    height=320,
)
st.markdown('<div class="metric-card" style="padding:1rem;">', unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)
# endregion