import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="Linear Regression Visualizer", layout="wide")
st.title("Linear Regression: How Does a Model Learn?")

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    dataset_type = st.selectbox("Dataset", ["Clean", "Noisy", "With Outliers"])
    noise_level  = st.slider("Noise Level", 0.0, 5.0, 1.0, step=0.1)
    m_manual     = st.slider("Slope (m)",     -5.0, 5.0, 1.0, step=0.1)
    b_manual     = st.slider("Intercept (b)", -10.0, 10.0, 0.0, step=0.1)
    lr           = st.slider("Learning Rate (α)", 0.001, 0.5, 0.01, step=0.001, format="%.3f")
    n_iters      = st.slider("GD Iterations", 10, 500, 100)
    add_outliers = st.checkbox("Add Outliers", value=False)

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data & Line Fit",
    "📉 Error Visualization",
    "🗺️ Loss Landscape",
    "🚀 Gradient Descent",
    "⚡ Learning Rate",
    "🔀 Noise & Outliers"
])

@st.cache_data
def generate_data(n=50, noise=1.0, outliers=False, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, n)
    y = 2.5 * X + 3 + rng.normal(0, noise, n)
    if outliers:
        idx = rng.choice(n, 5, replace=False)
        y[idx] += rng.choice([-1, 1], 5) * 15
    return X, y

X, y = generate_data(noise=noise_level, outliers=add_outliers)

with tab1:
    st.subheader("Can you eyeball the best fit?")
    st.write("Move the **slope** and **intercept** sliders in the sidebar to fit the line to the data.")

    y_pred = m_manual * X + b_manual
    mse    = np.mean((y - y_pred) ** 2)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(X, y, alpha=0.6, label="Data points")
    x_line = np.linspace(0, 10, 200)
    ax.plot(x_line, m_manual * x_line + b_manual, color="red", label=f"y = {m_manual}x + {b_manual}")
    ax.set_xlabel("X"); ax.set_ylabel("y")
    ax.legend()
    st.pyplot(fig)

    st.metric("Current MSE", f"{mse:.3f}")
    st.info("**What to observe:** MSE drops as your line gets closer to the data cloud. "
            "The true line has slope ≈ 2.5 and intercept ≈ 3.")

with tab2:
    st.subheader("Residuals: The Gaps the Model Tries to Close")
    st.write("Each vertical line is an **error** — the distance between predicted and actual y.")

    y_pred = m_manual * X + b_manual
    residuals = y - y_pred

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(X, y, zorder=5, label="Actual")
    x_line = np.linspace(0, 10, 200)
    ax.plot(x_line, m_manual * x_line + b_manual, color="red", label="Prediction")

    for xi, yi, yp in zip(X, y, y_pred):          # draw residual lines
        ax.plot([xi, xi], [yi, yp], color="gray", linewidth=0.8, alpha=0.7)

    ax.legend()
    st.pyplot(fig)

    col1, col2, col3 = st.columns(3)
    col1.metric("MSE",  f"{np.mean(residuals**2):.3f}")
    col2.metric("MAE",  f"{np.mean(np.abs(residuals)):.3f}")
    col3.metric("Max Error", f"{np.max(np.abs(residuals)):.3f}")

    st.info("**What to observe:** MSE squares each error, so large residuals hurt disproportionately. "
            "Outliers dominate the loss.")

with tab3:
    st.subheader("Loss Surface: J(m, b)")
    st.write("Every point on this surface = MSE for that (slope, intercept) combo. "
             "The model searches for the **lowest point** (the bowl's bottom).")

    # Build grid
    m_vals = np.linspace(-1, 6, 80)
    b_vals = np.linspace(-5, 12, 80)
    M, B   = np.meshgrid(m_vals, b_vals)
    Z      = np.array([[np.mean((y - (mi * X + bi))**2) for mi in m_vals]
                        for bi in b_vals])

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Contour Plot (top-down view)**")
        fig2d = go.Figure(go.Contour(
            x=m_vals, y=b_vals, z=Z,
            colorscale="Viridis",
            contours=dict(showlabels=True)
        ))
        fig2d.add_trace(go.Scatter(
            x=[m_manual], y=[b_manual],
            mode="markers",
            marker=dict(color="red", size=12, symbol="x"),
            name="Current position"
        ))
        fig2d.update_layout(xaxis_title="slope (m)", yaxis_title="intercept (b)", height=400)
        st.plotly_chart(fig2d, use_container_width=True)

    with col2:
        st.write("**3D Surface**")
        fig3d = go.Figure(go.Surface(x=M, y=B, z=Z, colorscale="Viridis", opacity=0.85))
        fig3d.add_trace(go.Scatter3d(
            x=[m_manual], y=[b_manual],
            z=[np.mean((y - (m_manual * X + b_manual))**2)],
            mode="markers",
            marker=dict(color="red", size=6)
        ))
        fig3d.update_layout(height=400,
            scene=dict(xaxis_title="m", yaxis_title="b", zaxis_title="MSE"))
        st.plotly_chart(fig3d, use_container_width=True)

    st.info("**What to observe:** The red marker shows where YOUR (m,b) sits on the surface. "
            "Move the sliders — watch it move toward the minimum.")

def run_gd(X, y, lr, n_iters, m_init=0.0, b_init=0.0):
    m, b = m_init, b_init
    n = len(X)
    history = [(m, b, np.mean((y - (m*X+b))**2))]
    for _ in range(n_iters):
        y_pred = m * X + b
        dm = (-2/n) * np.sum(X * (y - y_pred))
        db = (-2/n) * np.sum(y - y_pred)
        m -= lr * dm
        b -= lr * db
        history.append((m, b, np.mean((y - (m*X+b))**2)))
    return history

with tab4:
    st.subheader("Gradient Descent: The Model Walking Downhill")
    st.write("The model takes small steps in the direction that **reduces MSE** most quickly.")

    history = run_gd(X, y, lr, n_iters)
    ms, bs, losses = zip(*history)

    # Plot path on contour
    m_vals = np.linspace(-1, 6, 60)
    b_vals = np.linspace(-5, 12, 60)
    Z = np.array([[np.mean((y - (mi*X+bi))**2) for mi in m_vals] for bi in b_vals])

    fig = go.Figure(go.Contour(x=m_vals, y=b_vals, z=Z, colorscale="Viridis",
                               contours=dict(showlabels=False), opacity=0.8))
    fig.add_trace(go.Scatter(x=list(ms), y=list(bs), mode="lines+markers",
                             line=dict(color="red", width=2),
                             marker=dict(size=4),
                             name="GD path"))
    fig.add_trace(go.Scatter(x=[ms[0]], y=[bs[0]], mode="markers",
                             marker=dict(color="green", size=12, symbol="star"),
                             name="Start"))
    fig.add_trace(go.Scatter(x=[ms[-1]], y=[bs[-1]], mode="markers",
                             marker=dict(color="red", size=12, symbol="x"),
                             name="End"))
    fig.update_layout(xaxis_title="m", yaxis_title="b", height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Loss curve
    st.write("**Loss vs Iterations**")
    fig2, ax = plt.subplots(figsize=(8, 3))
    ax.plot(losses)
    ax.set_xlabel("Iteration"); ax.set_ylabel("MSE")
    ax.set_title("Convergence Curve")
    st.pyplot(fig2)

    col1, col2 = st.columns(2)
    col1.metric("Final m", f"{ms[-1]:.3f}", delta=f"true ≈ 2.5")
    col2.metric("Final b", f"{bs[-1]:.3f}", delta=f"true ≈ 3.0")

    st.info("**What to observe:** The path spirals/walks toward the minimum. "
            "Try different learning rates in the sidebar — too large = diverges, too small = very slow.")

with tab5:
    st.subheader("Learning Rate: Too Fast, Too Slow, Just Right")

    lrs_to_compare = [0.001, 0.01, 0.1, 0.3]
    fig, ax = plt.subplots(figsize=(8, 4))

    for test_lr in lrs_to_compare:
        hist = run_gd(X, y, test_lr, 200)
        _, _, test_losses = zip(*hist)
        ax.plot(test_losses, label=f"α = {test_lr}")

    ax.set_xlabel("Iteration"); ax.set_ylabel("MSE (log scale)")
    ax.set_yscale("log")
    ax.legend(); ax.set_title("Convergence for Different Learning Rates")
    st.pyplot(fig)

    st.info("**What to observe:**\n"
            "- **Too small (0.001):** Converges but slowly\n"
            "- **Good (0.01):** Smooth, fast convergence\n"
            "- **Too large (0.3):** Oscillates or diverges")


with tab6:
    st.subheader("How Robust is Linear Regression?")

    X_clean, y_clean = generate_data(noise=0.5, outliers=False)
    X_noisy, y_noisy = generate_data(noise=noise_level, outliers=True)

    hist_clean = run_gd(X_clean, y_clean, 0.01, 200)
    hist_noisy = run_gd(X_noisy, y_noisy, 0.01, 200)

    m_c, b_c = hist_clean[-1][0], hist_clean[-1][1]
    m_n, b_n = hist_noisy[-1][0], hist_noisy[-1][1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(X_clean, y_clean, alpha=0.4, color="blue", label="Clean data")
    ax.scatter(X_noisy, y_noisy, alpha=0.4, color="orange", label="Noisy + outliers")
    x_line = np.linspace(0, 10, 200)
    ax.plot(x_line, m_c * x_line + b_c, color="blue",   label=f"Clean fit: y={m_c:.2f}x+{b_c:.2f}")
    ax.plot(x_line, m_n * x_line + b_n, color="orange", label=f"Noisy fit: y={m_n:.2f}x+{b_n:.2f}")
    ax.legend(); ax.set_xlabel("X"); ax.set_ylabel("y")
    st.pyplot(fig)

    st.info("**What to observe:** Outliers **pull the line** away from the true relationship. "
            "MSE squares errors, so a single extreme point can dominate the entire loss.")
