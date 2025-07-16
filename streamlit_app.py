import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="White Background", layout="wide")

# Inject custom CSS to set background color
st.markdown(
    """
    <style>
    body {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.set_page_config(layout="wide")
st.title("Blochkugel mit Gate-Anwendung")

# --- Hilfsfunktionen für Zustände und Gates ---
def ket_0():
    return np.array([1, 0], dtype=complex)

def ket_1():
    return np.array([0, 1], dtype=complex)

def normalize(state):
    return state / np.linalg.norm(state)

def bloch_vector(state):
    alpha, beta = state
    x = 2 * (alpha.conjugate() * beta).real
    y = 2 * (alpha.conjugate() * beta).imag
    z = abs(alpha)**2 - abs(beta)**2
    return np.array([x, y, z])

def apply_gate(state, gate):
    return gate @ state

# --- Standard-Gates ---
pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
hadamard = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

# --- Rotationen (R_x, R_y, R_z) ---
def rotation_x(theta):
    return np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)

def rotation_y(theta):
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)

def rotation_z(theta):
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ], dtype=complex)

# --- Initialisierung ---
if "state" not in st.session_state:
    st.session_state.state = ket_0()
    st.session_state.traj = [bloch_vector(st.session_state.state)]

def update_state(gate):
    new_state = normalize(apply_gate(st.session_state.state, gate))
    st.session_state.state = new_state
    st.session_state.traj.append(bloch_vector(new_state))

def reset():
    st.session_state.state = ket_0()
    st.session_state.traj = [bloch_vector(st.session_state.state)]

# --- UI für Gates und Rotation ---
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("Pauli-X"):
        update_state(pauli_x)
with col2:
    if st.button("Pauli-Y"):
        update_state(pauli_y)
with col3:
    if st.button("Pauli-Z"):
        update_state(pauli_z)
with col4:
    if st.button("Hadamard"):
        update_state(hadamard)
with col5:
    if st.button("Reset"):
        reset()

# --- Rotation UI ---
st.markdown("### Rotation anwenden")
angle_deg = st.slider("Rotationswinkel (in Grad)", min_value=1, max_value=360, value=45, step=1)
angle_rad = np.deg2rad(angle_deg)

colx, coly, colz = st.columns(3)
with colx:
    if st.button("Rotate X"):
        update_state(rotation_x(angle_rad))
with coly:
    if st.button("Rotate Y"):
        update_state(rotation_y(angle_rad))
with colz:
    if st.button("Rotate Z"):
        update_state(rotation_z(angle_rad))

# --- SLERP ---
def slerp(v0, v1, num_points=100):
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    theta = np.arccos(dot)
    if theta < 1e-6:
        return np.tile(v0, (num_points, 1))
    if np.isclose(dot, -1.0):
        orthogonal = np.array([1, 0, 0])
        if np.allclose(v0, orthogonal) or np.allclose(v0, -orthogonal):
            orthogonal = np.array([0, 0, 1])
        axis = np.cross(v0, orthogonal)
        axis /= np.linalg.norm(axis)
        t = np.linspace(0, 1, num_points)
        return np.array([
            (v0 * np.cos(ti*np.pi) +
             np.cross(axis, v0) * np.sin(ti*np.pi) +
             axis * np.dot(axis, v0) * (1 - np.cos(ti*np.pi)))
            for ti in t
        ])
    sin_theta = np.sin(theta)
    t = np.linspace(0, 1, num_points)
    return (np.sin((1 - t) * theta)[:, None] * v0 +
            np.sin(t * theta)[:, None] * v1) / sin_theta

# --- Plot Funktion ---
def plot_bloch(traj):
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    fig = go.Figure()
    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.3, colorscale='Blues', showscale=False))

    # Achsen
    for axis, color, label_pos in zip(
        ['X', 'Y', 'Z'],
        ['black', 'black', 'black'],
        [([-1.2, 1.2], [0, 0], [0, 0]),
         ([0, 0], [-1.2, 1.2], [0, 0]),
         ([0, 0], [0, 0], [-1.2, 1.2])]
    ):
        fig.add_trace(go.Scatter3d(
            x=label_pos[0], y=label_pos[1], z=label_pos[2],
            mode='lines+text', text=[f"-{axis}", f"+{axis}"], textposition='top center',
            line=dict(color=color, width=5)
        ))

    # Basiszustände
    labels = [
        (0, 0, 1.6, "|0⟩"),
        (0, 0, -1.6, "|1⟩"),
        (1.5, 0, 0, "|+⟩"),
        (-1.5, 0, 0, "|−⟩"),
        (0, 1.5, 0, "|i⟩"),
        (0, -1.5, 0, "|−i⟩"),
    ]
    for x, y, z, label in labels:
        fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z],
                                   mode='text', text=[label],
                                   textposition='middle center',
                                   textfont=dict(size=18, color='black')))

    # Slerp Linie
    traj = np.array(traj)
    if len(traj) >= 2:
        arc_points = slerp(traj[-2], traj[-1])
        fig.add_trace(go.Scatter3d(
            x=arc_points[:, 0], y=arc_points[:, 1], z=arc_points[:, 2],
            mode='lines', line=dict(color='blue', width=6)
        ))

    # Vektor
    v = traj[-1]
    fig.add_trace(go.Scatter3d(
        x=[0, v[0]], y=[0, v[1]], z=[0, v[2]],
        mode='lines+markers+text',
        line=dict(color='red', width=10),
        marker=dict(size=5, color='red'),
        text=["", "ψ"], textposition='top center'
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[-2, 2], zeroline=False),
            yaxis=dict(title='Y', range=[-2, 2], zeroline=False),
            zaxis=dict(title='Z', range=[-2, 2], zeroline=False),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False
    )
    return fig

# --- Zustand anzeigen ---
state = st.session_state.state
alpha, beta = state
alpha_str = f"{alpha.real:.3f}" + (f" + {alpha.imag:.3f}i" if abs(alpha.imag) > 1e-6 else "")
beta_str = f"{beta.real:.3f}" + (f" + {beta.imag:.3f}i" if abs(beta.imag) > 1e-6 else "")
st.markdown(f"**Aktueller Zustand |ψ⟩:**  \n"
            f"|ψ⟩ = ({alpha_str}) |0⟩ + ({beta_str}) |1⟩")

# --- Plot anzeigen ---
fig = plot_bloch(st.session_state.traj)
st.plotly_chart(fig, use_container_width=True, height=600)



