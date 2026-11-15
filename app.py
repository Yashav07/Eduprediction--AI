import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="EduPredict AI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.stApp { background: #f5f7fb; }
.block-container { padding-top: 2rem; }
[data-testid="stMetric"] {
    background: white; padding: 16px; border-radius: 12px;
    border: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_demo_model():
    # Synthetic demonstration data only. Replace with real, anonymized
    # historical student records before treating predictions as meaningful.
    rng = np.random.default_rng(42)
    n = 1800
    df = pd.DataFrame({
        "Attendance": rng.integers(35, 101, n),
        "Assignments": rng.integers(25, 101, n),
        "Study Hours": rng.uniform(0, 10, n),
        "Previous Marks": rng.integers(20, 101, n),
        "Internal Marks": rng.integers(15, 101, n),
    })
    noise = rng.normal(0, 6, n)
    df["Final Marks"] = np.clip(
        0.20 * df["Attendance"]
        + 0.20 * df["Assignments"]
        + 0.12 * (df["Study Hours"] / 10 * 100)
        + 0.28 * df["Previous Marks"]
        + 0.20 * df["Internal Marks"]
        + noise,
        0, 100
    )
    X = df.drop(columns=["Final Marks"])
    y = df["Final Marks"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=150, max_depth=10, random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, mean_absolute_error(y_test, pred), r2_score(y_test, pred)

model, model_mae, model_r2 = train_demo_model()

def category(score):
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Very Good"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Average"
    return "Needs Improvement"

st.title("🎓 EduPredict AI")
st.caption("AI-Based Student Performance Analysis and Prediction System")

with st.sidebar:
    st.title("EduPredict AI")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Performance Calculator", "ML Prediction", "Student Report"]
    )
    st.divider()
    st.caption("Educational demonstration project")

st.subheader("Student Information")
c1, c2, c3 = st.columns(3)
with c1:
    student_name = st.text_input("Student Name", "Demo Student")
with c2:
    student_id = st.text_input("Student ID / UID", "STU001")
with c3:
    subject = st.selectbox(
        "Subject",
        ["Mathematics", "Physics", "Chemistry", "Computer Science", "English"]
    )

st.subheader("Academic Performance Details")
c1, c2, c3 = st.columns(3)
with c1:
    attendance = st.slider("Attendance (%)", 0, 100, 85)
    assignments = st.slider("Assignment Performance (%)", 0, 100, 80)
with c2:
    study_hours = st.slider("Daily Study Hours", 0.0, 12.0, 3.0, 0.5)
    previous_marks = st.slider("Previous Examination Marks (%)", 0, 100, 75)
with c3:
    internal_marks = st.slider("Internal Examination Marks (%)", 0, 100, 80)

# Transparent, fixed weighted calculator formula.
calculator_score = round(
    0.30 * internal_marks
    + 0.25 * assignments
    + 0.20 * attendance
    + 0.25 * previous_marks, 2
)

features = pd.DataFrame([{
    "Attendance": attendance,
    "Assignments": assignments,
    "Study Hours": study_hours,
    "Previous Marks": previous_marks,
    "Internal Marks": internal_marks
}])
ml_score = round(float(np.clip(model.predict(features)[0], 0, 100)), 2)

if page == "Dashboard":
    st.header(f"Welcome, {student_name}!")
    a, b, c = st.columns(3)
    a.metric("Calculator Score", f"{calculator_score:.2f}%")
    b.metric("ML Estimated Marks", f"{ml_score:.2f}%")
    c.metric("Attendance", f"{attendance}%")

    chart_df = pd.DataFrame({
        "Parameter": ["Attendance", "Assignments", "Previous Marks",
                      "Internal Marks", "Calculator Score", "ML Estimate"],
        "Score": [attendance, assignments, previous_marks, internal_marks,
                  calculator_score, ml_score]
    })
    fig = px.bar(
        chart_df, x="Parameter", y="Score", text="Score",
        range_y=[0, 100], title="Academic Performance Overview"
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "The ML result is an educational estimate from a model trained on "
        "synthetic demonstration data. It is not a guaranteed exam result."
    )

elif page == "Performance Calculator":
    st.header("📊 Performance Calculator")
    st.metric("Weighted Performance Score", f"{calculator_score:.2f}%")
    st.metric("Performance Category", category(calculator_score))
    st.markdown("**Formula**")
    st.code(
        "Score = (Internal × 0.30) + (Assignments × 0.25) "
        "+ (Attendance × 0.20) + (Previous Marks × 0.25)"
    )
    st.write("The weights total 100%. This is a project-defined score, "
             "not an official university grading formula.")

elif page == "ML Prediction":
    st.header("🤖 Machine Learning Prediction")
    st.metric("Estimated Final Examination Marks", f"{ml_score:.2f}/100")
    st.metric("Estimated Category", category(ml_score))
    st.progress(ml_score / 100)
    a, b = st.columns(2)
    a.metric("Demo test MAE", f"{model_mae:.2f}")
    b.metric("Demo test R²", f"{model_r2:.3f}")
    st.caption(
        "These evaluation metrics are calculated on a held-out portion of "
        "synthetic data. They do not demonstrate real-world accuracy."
    )
    st.write("Model inputs")
    st.dataframe(features, use_container_width=True, hide_index=True)

elif page == "Student Report":
    st.header("📄 Student Performance Report")
    st.write(f"**Name:** {student_name}")
    st.write(f"**Student ID:** {student_id}")
    st.write(f"**Subject:** {subject}")
    report = pd.DataFrame({
        "Parameter": [
            "Attendance", "Assignment Performance", "Daily Study Hours",
            "Previous Marks", "Internal Marks",
            "Calculator Score", "ML Estimated Marks"
        ],
        "Result": [
            f"{attendance}%", f"{assignments}%", f"{study_hours} hours",
            f"{previous_marks}%", f"{internal_marks}%",
            f"{calculator_score}%", f"{ml_score}%"
        ]
    })
    st.dataframe(report, use_container_width=True, hide_index=True)
    st.write(f"**Calculator category:** {category(calculator_score)}")
    st.write(f"**ML estimate category:** {category(ml_score)}")
    st.download_button(
        "Download Student Report (CSV)",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name=f"{student_id}_performance_report.csv",
        mime="text/csv"
    )

st.divider()
st.caption(
    "EduPredict AI | Educational project. Use real, anonymized and "
    "consented historical data before relying on predictions."
)
