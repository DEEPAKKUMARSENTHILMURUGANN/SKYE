"""
SKYE Real-Time Aircraft Health Monitoring Dashboard
=================================================
Premium aviation-grade cockpit diagnostics and health monitoring suite.
Supports manual override mode to control flight phases and operational faults.
Includes a highly-annotated explanatory anomaly score graph.
"""

import os
import sys
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import threading
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sensor_simulator import simulator as sensor_sim
from edge_processing import processing as edge_proc
from models.anomaly_detection.evaluate import AnomalyEvaluator
from models.rul_prediction.model import RULEnsemble
from utils.recommendation import RecommendationEngine
_telemetry_lock = threading.Lock()
_telemetry_history = []
_active_mode = "NORMAL"
_active_faults = []
_flight_phase = "PRE-FLIGHT"
_scenario_control = "AUTO" 
edge_processor = edge_proc.EdgeAIProcessor(window_size=30)
anomaly_evaluator = AnomalyEvaluator()
rul_ensemble = RULEnsemble()
sim = sensor_sim.AircraftSensorSimulator(update_interval_seconds=0.5)

def _run_sensor():
    global _active_mode, _active_faults, _flight_phase, _scenario_control, sim
    
    step = 0
    while True:
        if _scenario_control == "AUTO":
            if step == 50:
                sim.set_mode("DEGRADATION")
                sim.inject_fault("bearing_wear")
            elif step == 110:
                sim.set_mode("FAULT_INJECTION")
                sim.inject_fault("hydraulic_leak")
            elif step == 160:
                sim.set_mode("SUDDEN_ANOMALY")
                sim.inject_fault("compressor_surge")
                sim.inject_fault("wing_structural_damage")
            elif step >= 220:
                sim.set_mode("NORMAL")
                sim.reset()
                step = 0
        else:
            pass
            
        sample = sim.get_next_telemetry()
        _active_mode = sim.mode
        _active_faults = list(sim.active_faults)
        _flight_phase = sample["flight_phase"]
        processed = edge_processor.process_telemetry(sample)
        flat_data = {
            "timestamp": sample["timestamp"],
            "step": sample["step"],
            "flight_cycle": sample["flight_cycle"],
            "flight_phase": sample["flight_phase"],
            "altitude": sample["altitude"],
            "speed": sample["speed"],
            **processed["raw_flat"],
            "overall_health": processed["heuristic_health"]["overall"],
            "engine_health": processed["heuristic_health"]["engine"],
            "hydraulic_health": processed["heuristic_health"]["hydraulic"],
            "structural_health": processed["heuristic_health"]["structural"],
            "environmental_health": processed["heuristic_health"]["environmental"],
        }
        
        with _telemetry_lock:
            _telemetry_history.append(flat_data)
            if len(_telemetry_history) > 100:
                _telemetry_history.pop(0)
                
        if _scenario_control == "AUTO":
            step += 1
        time.sleep(0.5)
_thread = threading.Thread(target=_run_sensor, daemon=True)
_thread.start()

def get_latest_snapshot():
    with _telemetry_lock:
        return list(_telemetry_history)

def get_rul_input_window(history_list):
    window_size = 30
    if len(history_list) < window_size:
        pad_len = window_size - len(history_list)
        pad = [history_list[0] if history_list else {}] * pad_len
        full_list = pad + history_list
    else:
        full_list = history_list[-window_size:]
        
    window_data = np.zeros((window_size, 24))
    s_base = {
        "s1": 518.67, "s2": 642.3, "s3": 1589.7, "s4": 1400.6,
        "s5": 14.62, "s6": 21.61, "s7": 554.3, "s8": 2388.06,
        "s9": 9044.03, "s10": 1.3, "s11": 47.47, "s12": 521.66,
        "s13": 2388.02, "s14": 8138.62, "s15": 8.41, "s16": 0.03,
        "s17": 392.0, "s18": 2388.0, "s19": 100.0, "s20": 38.85, "s21": 23.31
    }
    
    for i, row in enumerate(full_list):
        window_data[i, 0] = 0.001
        window_data[i, 1] = 0.0002
        window_data[i, 2] = 100.0
        for s_idx in range(1, 22):
            window_data[i, s_idx + 2] = s_base[f"s{s_idx}"]
            
        sim_egt = row.get("egt", 610.0)
        window_data[i, 3] = 642.3 + (sim_egt - 610.0) * 0.1
        sim_vibe = row.get("engine_vibration", 2.0)
        window_data[i, 4] = 1589.7 + (sim_vibe - 2.0) * 10.0
        sim_n1 = row.get("n1_speed", 90.0)
        window_data[i, 5] = 1400.6 + (sim_n1 - 90.0) * 2.0
        sim_oil = row.get("oil_pressure", 55.0)
        window_data[i, 23] = 23.31 + (sim_oil - 55.0) * 0.1
        
    return window_data

def make_sparkline(y_data, hex_color="#3080e6"):
    """Generates a small, clean background sparkline graph."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=list(y_data),
        mode="lines",
        line=dict(color=hex_color, width=1.5),
        hoverinfo="none"
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=32,
        showlegend=False
    )
    return fig


def make_advisory_styles(anom_status):
    """Returns (dot_style, text_style, label) for the advisory header indicator."""
    if anom_status == "ANOMALY":
        color = "#ff3f34"
        label = "ANOMALY DETECTED"
    elif anom_status == "WARNING":
        color = "#ff9f1c"
        label = "ADVISORY"
    else:
        color = "#05c46b"
        label = "NOMINAL"
    dot_style = {
        "width": "8px", "height": "8px", "borderRadius": "50%",
        "backgroundColor": color, "display": "inline-block"
    }
    text_style = {
        "fontFamily": "'Share Tech Mono', monospace", "fontSize": "14px",
        "color": color, "letterSpacing": "1.5px", "fontWeight": "bold"
    }
    return dot_style, text_style, label


def create_timeline_fig(df_hist, threshold, latest_score):
    """Creates the annotated anomaly score timeline figure."""
    fig = go.Figure()

    if df_hist is not None and len(df_hist) > 0 and "anomaly_score_cache" in df_hist.columns:
        scores = df_hist["anomaly_score_cache"].fillna(0).tolist()
        x_vals = list(range(len(scores)))
        phases = df_hist.get("flight_phase", pd.Series(["UNKNOWN"] * len(scores)))
        score_colors = []
        for s in scores:
            if s > threshold * 1.5:
                score_colors.append("#ff3f34")
            elif s > threshold:
                score_colors.append("#ff9f1c")
            else:
                score_colors.append("#3080e6")

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=scores,
            mode="lines",
            line=dict(color="#3080e6", width=2),
            name="Anomaly Score",
            fill="tozeroy",
            fillcolor="rgba(48,128,230,0.08)",
            hovertemplate="Step %{x}<br>Score: %{y:.4f}<extra></extra>"
        ))
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#ff9f1c",
            line_width=1.5,
            annotation_text=f"THRESHOLD ({threshold:.3f})",
            annotation_position="top right",
            annotation_font=dict(color="#ff9f1c", size=9)
        )
        if len(x_vals) > 0:
            last_x = x_vals[-1]
            marker_color = "#ff3f34" if latest_score > threshold else "#05c46b"
            fig.add_trace(go.Scatter(
                x=[last_x],
                y=[latest_score],
                mode="markers",
                marker=dict(color=marker_color, size=8, symbol="circle"),
                showlegend=False,
                hovertemplate=f"Latest: {latest_score:.4f}<extra></extra>"
            ))
    else:
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="lines", line=dict(color="#3080e6", width=2)))
        if threshold:
            fig.add_hline(y=threshold, line_dash="dash", line_color="#ff9f1c", line_width=1.5)

    fig.update_layout(
        paper_bgcolor="#090d16",
        plot_bgcolor="#090d16",
        margin=dict(l=40, r=20, t=10, b=30),
        height=220,
        showlegend=False,
        xaxis=dict(
            color="#4a5b78",
            gridcolor="#141c2c",
            showgrid=True,
            zeroline=True,
            zerolinecolor="#1e2c42"
        ),
        yaxis=dict(
            color="#4a5b78",
            gridcolor="#141c2c",
            showgrid=True,
            zeroline=True,
            zerolinecolor="#1e2c42"
        ),
        font=dict(family="'Share Tech Mono', monospace", color="#8a9bb4", size=10)
    )
    return fig
app = dash.Dash(__name__, external_stylesheets=[
    "https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;500;600;700&display=swap"
])
app.title = "SKYE Interactive Cockpit Health Monitor"

app.layout = html.Div(
    style={
        "backgroundColor": "#05070c",
        "color": "#8a9bb4",
        "fontFamily": "'Inter', sans-serif",
        "padding": "16px 20px",
        "minHeight": "100vh",
        "boxSizing": "border-box"
    },
    children=[
        dcc.Interval(id="dashboard-interval", interval=1000, n_intervals=0),
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "marginBottom": "16px",
                "paddingBottom": "8px",
                "borderBottom": "1px solid
            },
            children=[
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "10px"},
                    children=[
                        html.Div(
                            style={
                                "width": "24px",
                                "height": "24px",
                                "border": "2px solid
                                "borderRadius": "50%",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                                "boxShadow": "0 0 10px rgba(48,128,230,0.4)"
                            },
                            children=[
                                html.Div(style={"width": "10px", "height": "10px", "backgroundColor": "#3080e6", "borderRadius": "50%"})
                            ]
                        ),
                        html.Div([
                            html.Div("EDGE AI · COCKPIT MONITORING SYSTEM", style={"fontFamily": "'Share Tech Mono', monospace", "fontSize": "16px", "color": "#cdd6e4", "letterSpacing": "1.5px", "fontWeight": "bold"}),
                            html.Div("V2.5.0", style={"fontSize": "10px", "color": "#4a5b78", "marginTop": "1px", "fontWeight": "600"})
                        ])
                    ]
                ),
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "8px"},
                    children=[
                        html.Span(
                            id="advisory-dot",
                            style={
                                "width": "8px",
                                "height": "8px",
                                "borderRadius": "50%",
                                "backgroundColor": "#d27e1f",
                                "display": "inline-block"
                            }
                        ),
                        html.Span("ADVISORY", id="advisory-text", style={"fontFamily": "'Share Tech Mono', monospace", "fontSize": "14px", "color": "#d27e1f", "letterSpacing": "1.5px", "fontWeight": "bold"})
                    ]
                ),
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "20px", "fontFamily": "'Share Tech Mono', monospace", "fontSize": "13px", "color": "#8a9bb4"},
                    children=[
                        html.Div([
                            html.Span("ELAPSED ", style={"color": "#4a5b78"}),
                            html.Span("02:16:25", id="header-elapsed", style={"color": "#3080e6"})
                        ]),
                        html.Div([
                            html.Span("ETA BOM ", style={"color": "#4a5b78"}),
                            html.Span("00:47", id="header-eta", style={"color": "#3080e6"})
                        ]),
                        html.Button(
                            "PAUSE",
                            style={
                                "backgroundColor": "transparent",
                                "border": "1px solid
                                "color": "#cdd6e4",
                                "padding": "4px 16px",
                                "borderRadius": "4px",
                                "cursor": "pointer",
                                "fontSize": "12px",
                                "fontFamily": "'Share Tech Mono', monospace",
                                "fontWeight": "bold",
                                "letterSpacing": "1px"
                            }
                        )
                    ]
                )
            ]
        ),
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(5, 1fr)",
                "gap": "12px",
                "marginBottom": "16px"
            },
            children=[
                html.Div(
                    id="card-vibration",
                    style={
                        "backgroundColor": "#090d16",
                        "border": "1px solid
                        "borderRadius": "6px",
                        "padding": "12px",
                        "position": "relative",
                        "overflow": "hidden"
                    },
                    children=[
                        html.Div("ENGINE VIBRATION", style={"fontSize": "10px", "fontWeight": "700", "letterSpacing": "1px", "color": "#4a5b78", "marginBottom": "4px"}),
                        html.Div([
                            html.Span("0.21", id="val-vibration", style={"fontSize": "26px", "fontWeight": "bold", "color": "#ffffff", "fontFamily": "'Share Tech Mono', monospace"}),
                            html.Span(" g RMS", style={"fontSize": "11px", "color": "#4a5b78", "fontWeight": "600", "marginLeft": "4px"})
                        ]),
                        html.Div(id="chg-vibration", style={"fontSize": "10px", "fontWeight": "bold", "marginTop": "2px"}),
                        html.Div(dcc.Graph(id="spark-vibration", config={"displayModeBar": False}), style={"position": "absolute", "bottom": "0", "left": "0", "right": "0", "height": "32px"})
                    ]
                ),
                html.Div(
                    id="card-egt",
                    style={
                        "backgroundColor": "#090d16",
                        "border": "1px solid
                        "borderRadius": "6px",
                        "padding": "12px",
                        "position": "relative",
                        "overflow": "hidden"
                    },
                    children=[
                        html.Div("EXHAUST GAS TEMP", style={"fontSize": "10px", "fontWeight": "700", "letterSpacing": "1px", "color": "#4a5b78", "marginBottom": "4px"}),
                        html.Div([
                            html.Span("633", id="val-egt", style={"fontSize": "26px", "fontWeight": "bold", "color": "#ffffff", "fontFamily": "'Share Tech Mono', monospace"}),
                            html.Span(" °C", style={"fontSize": "12px", "color": "#4a5b78", "fontWeight": "600", "marginLeft": "2px"})
                        ]),
                        html.Div(id="chg-egt", style={"fontSize": "10px", "fontWeight": "bold", "marginTop": "2px"}),
                        html.Div(dcc.Graph(id="spark-egt", config={"displayModeBar": False}), style={"position": "absolute", "bottom": "0", "left": "0", "right": "0", "height": "32px"})
                    ]
                ),
                html.Div(
                    id="card-oil",
                    style={
                        "backgroundColor": "#090d16",
                        "border": "1px solid
                        "borderRadius": "6px",
                        "padding": "12px",
                        "position": "relative",
                        "overflow": "hidden"
                    },
                    children=[
                        html.Div("OIL PRESSURE", style={"fontSize": "10px", "fontWeight": "700", "letterSpacing": "1px", "color": "#4a5b78", "marginBottom": "4px"}),
                        html.Div([
                            html.Span("40", id="val-oil", style={"fontSize": "26px", "fontWeight": "bold", "color": "#ffffff", "fontFamily": "'Share Tech Mono', monospace"}),
                            html.Span(" PSI", style={"fontSize": "11px", "color": "#4a5b78", "fontWeight": "600", "marginLeft": "4px"})
                        ]),
                        html.Div(id="chg-oil", style={"fontSize": "10px", "fontWeight": "bold", "marginTop": "2px"}),
                        html.Div(dcc.Graph(id="spark-oil", config={"displayModeBar": False}), style={"position": "absolute", "bottom": "0", "left": "0", "right": "0", "height": "32px"})
                    ]
                ),
                html.Div(
                    id="card-hydraulic",
                    style={
                        "backgroundColor": "#090d16",
                        "border": "1px solid
                        "borderRadius": "6px",
                        "padding": "12px",
                        "position": "relative",
                        "overflow": "hidden"
                    },
                    children=[
                        html.Div("HYDRAULIC PRESSURE", style={"fontSize": "10px", "fontWeight": "700", "letterSpacing": "1px", "color": "#4a5b78", "marginBottom": "4px"}),
                        html.Div([
                            html.Span("2992", id="val-hydraulic", style={"fontSize": "26px", "fontWeight": "bold", "color": "#ffffff", "fontFamily": "'Share Tech Mono', monospace"}),
                            html.Span(" PSI", style={"fontSize": "11px", "color": "#4a5b78", "fontWeight": "600", "marginLeft": "4px"})
                        ]),
                        html.Div(id="chg-hydraulic", style={"fontSize": "10px", "fontWeight": "bold", "marginTop": "2px"}),
                        html.Div(dcc.Graph(id="spark-hydraulic", config={"displayModeBar": False}), style={"position": "absolute", "bottom": "0", "left": "0", "right": "0", "height": "32px"})
                    ]
                ),
                html.Div(
                    id="card-stress",
                    style={
                        "backgroundColor": "#090d16",
                        "border": "1px solid
                        "borderRadius": "6px",
                        "padding": "12px",
                        "position": "relative",
                        "overflow": "hidden"
                    },
                    children=[
                        html.Div("AIRFRAME STRESS", style={"fontSize": "10px", "fontWeight": "700", "letterSpacing": "1px", "color": "#4a5b78", "marginBottom": "4px"}),
                        html.Div([
                            html.Span("90", id="val-stress", style={"fontSize": "26px", "fontWeight": "bold", "color": "#ffffff", "fontFamily": "'Share Tech Mono', monospace"}),
                            html.Span(" MPa", style={"fontSize": "11px", "color": "#4a5b78", "fontWeight": "600", "marginLeft": "4px"})
                        ]),
                        html.Div(id="chg-stress", style={"fontSize": "10px", "fontWeight": "bold", "marginTop": "2px"}),
                        html.Div(dcc.Graph(id="spark-stress", config={"displayModeBar": False}), style={"position": "absolute", "bottom": "0", "left": "0", "right": "0", "height": "32px"})
                    ]
                )
            ]
        ),
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "7fr 5fr",
                "gap": "16px"
            },
            children=[
                html.Div(
                    children=[
                        html.Div(
                            style={
                                "backgroundColor": "#090d16",
                                "border": "1px solid
                                "borderRadius": "6px",
                                "padding": "16px",
                                "marginBottom": "16px"
                            },
                            children=[
                                html.Div(
                                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "16px"},
                                    children=[
                                        html.Div("SUBSYSTEM HEALTH SCORES", style={"fontSize": "12px", "fontWeight": "700", "letterSpacing": "1px", "color": "#ffffff"}),
                                        html.Div(
                                            style={"display": "flex", "alignItems": "center", "gap": "12px"},
                                            children=[
                                                html.Div(
                                                    "TFT ANOMALY MODEL",
                                                    style={
                                                        "backgroundColor": "#0e1a2f",
                                                        "color": "#3080e6",
                                                        "fontSize": "10px",
                                                        "fontWeight": "bold",
                                                        "padding": "3px 8px",
                                                        "borderRadius": "3px",
                                                        "fontFamily": "'Share Tech Mono', monospace",
                                                        "border": "1px solid
                                                    }
                                                ),
                                                html.Div(
                                                    "11:10:43 PM",
                                                    id="current-time-clock",
                                                    style={"fontFamily": "'Share Tech Mono', monospace", "fontSize": "11px", "color": "#4a5b78"}
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                html.Div(id="subsystem-bars-container", style={"display": "flex", "flexDirection": "column", "gap": "10px"})
                            ]
                        ),
                        html.Div(
                            style={
                                "backgroundColor": "#090d16",
                                "border": "1px solid
                                "borderRadius": "6px",
                                "padding": "16px"
                            },
                            children=[
                                html.Div(
                                    "ANOMALY SCORE TIMELINE - SEQUENCE RECONSTRUCTION LOSS (LSTM AUTOENCODER)",
                                    style={"fontSize": "11px", "fontWeight": "700", "letterSpacing": "1px", "color": "#ffffff", "marginBottom": "12px"}
                                ),
                                dcc.Graph(id="anomaly-timeline-graph", config={"displayModeBar": False})
                            ]
                        )
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(
                            style={
                                "backgroundColor": "#090d16",
                                "border": "1px solid
                                "borderRadius": "6px",
                                "padding": "16px",
                                "marginBottom": "16px"
                            },
                            children=[
                                html.Div(
                                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "16px"},
                                    children=[
                                        html.Div("RUL ESTIMATES", style={"fontSize": "12px", "fontWeight": "700", "letterSpacing": "1px", "color": "#ffffff"}),
                                        html.Div(
                                            "LSTM+XGB",
                                            style={
                                                "backgroundColor": "#0e1a2f",
                                                "color": "#3080e6",
                                                "fontSize": "10px",
                                                "fontWeight": "bold",
                                                "padding": "3px 8px",
                                                "borderRadius": "3px",
                                                "fontFamily": "'Share Tech Mono', monospace",
                                                "border": "1px solid
                                            }
                                        )
                                    ]
                                ),
                                html.Div(id="rul-estimates-container", style={"display": "flex", "flexDirection": "column", "gap": "10px"})
                            ]
                        ),
                        html.Div(
                            style={
                                "backgroundColor": "#090d16",
                                "border": "1px solid
                                "borderRadius": "6px",
                                "padding": "16px",
                                "marginBottom": "16px"
                            },
                            children=[
                                html.Div("SIMULATION CONTROL PANEL (COCKPIT CONFIG)", style={"fontFamily": "'Share Tech Mono', monospace", "fontSize": "12px", "fontWeight": "700", "letterSpacing": "1px", "color": "#3080e6", "marginBottom": "14px"}),
                                html.Div(
                                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "14px"},
                                    children=[
                                        html.Div("SIMULATION MODE:", style={"fontSize": "11px", "fontWeight": "700", "color": "#8a9bb4"}),
                                        dcc.RadioItems(
                                            id="ctrl-mode-select",
                                            options=[
                                                {"label": "AUTO SCENARIO", "value": "AUTO"},
                                                {"label": "MANUAL OVERRIDE", "value": "MANUAL"}
                                            ],
                                            value="AUTO",
                                            labelStyle={"display": "inline-block", "marginRight": "12px", "fontSize": "10px", "fontWeight": "bold", "color": "#cdd6e4"},
                                            inputStyle={"marginRight": "4px"}
                                        )
                                    ]
                                ),
                                html.Div(
                                    id="phase-selector-container",
                                    style={"marginBottom": "14px"},
                                    children=[
                                        html.Div("MANUAL FLIGHT PHASE:", style={"fontSize": "11px", "fontWeight": "700", "color": "#8a9bb4", "marginBottom": "8px"}),
                                        dcc.RadioItems(
                                            id="phase-radio",
                                            options=[
                                                {"label": "PRE-FLIGHT", "value": "PRE-FLIGHT"},
                                                {"label": "TAKEOFF", "value": "TAKEOFF"},
                                                {"label": "CRUISE", "value": "CRUISE"},
                                                {"label": "LANDING", "value": "LANDING"},
                                                {"label": "TAXI", "value": "TAXI"}
                                            ],
                                            value="PRE-FLIGHT",
                                            labelStyle={"display": "inline-block", "marginRight": "10px", "fontSize": "9px", "fontFamily": "'Share Tech Mono', monospace", "fontWeight": "bold", "color": "#cdd6e4"},
                                            inputStyle={"marginRight": "4px"}
                                        )
                                    ]
                                ),
                                html.Div(
                                    id="mode-selector-container",
                                    style={"marginBottom": "14px"},
                                    children=[
                                        html.Div("MANUAL OPERATIONAL PROFILE:", style={"fontSize": "11px", "fontWeight": "700", "color": "#8a9bb4", "marginBottom": "8px"}),
                                        dcc.RadioItems(
                                            id="mode-radio",
                                            options=[
                                                {"label": "NORMAL PROFILE", "value": "NORMAL"},
                                                {"label": "GRADUAL WEAR", "value": "DEGRADATION"},
                                                {"label": "FAULT INJECTION", "value": "FAULT_INJECTION"},
                                                {"label": "SUDDEN ANOMALY", "value": "SUDDEN_ANOMALY"}
                                            ],
                                            value="NORMAL",
                                            labelStyle={"display": "inline-block", "marginRight": "10px", "fontSize": "9px", "fontFamily": "'Share Tech Mono', monospace", "fontWeight": "bold", "color": "#cdd6e4"},
                                            inputStyle={"marginRight": "4px"}
                                        )
                                    ]
                                ),
                                html.Div(
                                    id="fault-injector-container",
                                    children=[
                                        html.Div("ACTIVE SYSTEM FAULT SIGNATURES:", style={"fontSize": "11px", "fontWeight": "700", "color": "#8a9bb4", "marginBottom": "8px"}),
                                        dcc.Checklist(
                                            id="faults-checklist",
                                            options=[
                                                {"label": "Engine Main Bearing Wear", "value": "bearing_wear"},
                                                {"label": "Hydraulic Primary Actuator Leak", "value": "hydraulic_leak"},
                                                {"label": "Engine Compressor Surge Oscillation", "value": "compressor_surge"},
                                                {"label": "Wing Main root structural damage", "value": "wing_structural_damage"},
                                                {"label": "Bleed temperature valve fail open", "value": "ecs_bleed_valve_fail"}
                                            ],
                                            value=[],
                                            labelStyle={"display": "inline-block", "marginRight": "14px", "fontSize": "9px", "fontFamily": "'Share Tech Mono', monospace", "color": "#8a9bb4", "marginBottom": "6px"},
                                            inputStyle={"marginRight": "4px"}
                                        )
                                    ]
                                )
                            ]
                        ),
                        html.Div(
                            id="cockpit-alerts-panel",
                            style={
                                "backgroundColor": "#090d16",
                                "border": "1px solid
                                "borderRadius": "6px",
                                "padding": "16px"
                            }
                        )
                    ]
                )
            ]
        )
    ]
)

@app.callback(
    Output("phase-selector-container", "style"),
    Output("mode-selector-container", "style"),
    Output("fault-injector-container", "style"),
    Input("ctrl-mode-select", "value"),
    Input("phase-radio", "value"),
    Input("mode-radio", "value"),
    Input("faults-checklist", "value")
)
def handle_cockpit_controls(ctrl_mode, selected_phase, selected_mode, selected_faults):
    global _scenario_control, sim
    
    _scenario_control = ctrl_mode
    
    if ctrl_mode == "MANUAL":
        sim.manual_phase = selected_phase
        sim.set_mode(selected_mode)
        sim.active_faults = set(selected_faults)
        
        visible_style = {"display": "block", "marginBottom": "14px"}
        return visible_style, visible_style, {"display": "block"}
    else:
        sim.manual_phase = None
        dimmed_style = {"opacity": "0.3", "pointerEvents": "none", "marginBottom": "14px"}
        return dimmed_style, dimmed_style, {"opacity": "0.3", "pointerEvents": "none"}

@app.callback(
    Output("advisory-dot", "style"),
    Output("advisory-text", "children"),
    Output("advisory-text", "style"),
    Output("current-time-clock", "children"),
    Output("val-vibration", "children"),
    Output("chg-vibration", "children"),
    Output("chg-vibration", "style"),
    Output("spark-vibration", "figure"),
    Output("val-egt", "children"),
    Output("chg-egt", "children"),
    Output("chg-egt", "style"),
    Output("spark-egt", "figure"),
    Output("val-oil", "children"),
    Output("chg-oil", "children"),
    Output("chg-oil", "style"),
    Output("card-oil", "style"),
    Output("spark-oil", "figure"),
    Output("val-hydraulic", "children"),
    Output("chg-hydraulic", "children"),
    Output("chg-hydraulic", "style"),
    Output("spark-hydraulic", "figure"),
    Output("val-stress", "children"),
    Output("chg-stress", "children"),
    Output("chg-stress", "style"),
    Output("spark-stress", "figure"),
    Output("subsystem-bars-container", "children"),
    Output("anomaly-timeline-graph", "figure"),
    Output("rul-estimates-container", "children"),
    Output("cockpit-alerts-panel", "children"),
    Input("dashboard-interval", "n_intervals")
)
def update_dashboard_ui(_):
    """Update all UI components based on latest telemetry."""
    history = get_latest_snapshot()
    current_time_str = time.strftime("%I:%M:%S %p")

    if not history:
        empty_fig = make_sparkline([0], "#3080e6")
        empty_timeline = create_timeline_fig(None, 0.5, 0.0)
        dot_s, txt_s, _ = make_advisory_styles("NORMAL")
        return (
            dot_s, "NOMINAL", txt_s, current_time_str,
            "0.00", "--", {"color": "#4a5b78"}, empty_fig,
            "000", "--", {"color": "#4a5b78"}, empty_fig,
            "00", "--", {"color": "#4a5b78"}, {
                "backgroundColor": "#090d16", "border": "1px solid
                "borderRadius": "6px", "padding": "12px",
                "position": "relative", "overflow": "hidden"
            }, empty_fig,
            "0000", "--", {"color": "#4a5b78"}, empty_fig,
            "00", "--", {"color": "#4a5b78"}, empty_fig,
            [], empty_timeline, [], []
        )

    latest_row = history[-1]
    df_hist = pd.DataFrame(history)
    def calc_delta(col, decimal_places=0):
        if len(history) < 2:
            return 0.0
        return round(history[-1].get(col, 0.0) - history[-2].get(col, 0.0), decimal_places)
    vibe = latest_row.get("engine_vibration", 0.0)
    vibe_chg = calc_delta("engine_vibration", 2)
    egt = latest_row.get("egt", 0.0)
    egt_chg = calc_delta("egt", 0)
    oil = latest_row.get("oil_pressure", 0.0)
    oil_chg = calc_delta("oil_pressure", 0)
    hyd = latest_row.get("hydraulic_pressure", 0.0)
    hyd_chg = calc_delta("hydraulic_pressure", 0)
    stress = latest_row.get("airframe_stress", 0.0)
    stress_chg = calc_delta("airframe_stress", 0)
    def fmt_chg(val, sign=True):
        if val > 0:
            return f"▲ +{val}" if sign else f"▲ {val}"
        elif val < 0:
            return f"▼ {abs(val)}"
        else:
            return "▲ 0"

    vibe_chg_str = fmt_chg(vibe_chg)
    vibe_chg_style = {"color": "#ff3f34" if vibe > 3.0 else "#05c46b"}
    egt_chg_str = fmt_chg(egt_chg)
    egt_chg_style = {"color": "#ff3f34" if egt > 720.0 else "#05c46b"}
    oil_chg_str = fmt_chg(oil_chg)
    oil_chg_style = {"color": "#ff3f34" if oil < 45.0 else "#05c46b"}
    card_oil_style = {
        "backgroundColor": "#090d16",
        "border": "1px solid
        "borderRadius": "6px",
        "padding": "12px",
        "position": "relative",
        "overflow": "hidden"
    }
    hyd_chg_str = fmt_chg(hyd_chg, sign=False)
    hyd_chg_style = {"color": "#ff3f34" if hyd < 2800.0 else "#05c46b"}
    stress_chg_str = fmt_chg(stress_chg, sign=False)
    stress_chg_style = {"color": "#ff3f34" if stress > 250.0 else "#05c46b"}
    vibe_color = "#ff3f34" if vibe > 3.0 else "#3080e6"
    egt_color = "#ff3f34" if egt > 720.0 else "#3080e6"
    oil_color = "#ff9f1c" if oil < 45.0 else "#3080e6"
    hyd_color = "#ff3f34" if hyd < 2800.0 else "#3080e6"
    stress_color = "#ff3f34" if stress > 250.0 else "#3080e6"

    fig_vibe = make_sparkline(df_hist["engine_vibration"].tail(15).values, vibe_color)
    fig_egt = make_sparkline(df_hist["egt"].tail(15).values, egt_color)
    fig_oil = make_sparkline(df_hist["oil_pressure"].tail(15).values, oil_color)
    fig_hyd = make_sparkline(df_hist["hydraulic_pressure"].tail(15).values, hyd_color)
    fig_stress = make_sparkline(df_hist["airframe_stress"].tail(15).values, stress_color)
    window_arr = edge_processor.get_sliding_window_array()
    anomaly_res = anomaly_evaluator.detect_anomaly(window_arr)
    anom_score = anomaly_res["anomaly_score"]
    anom_status = anomaly_res["status"]
    advisory_dot_style, advisory_text_style, adv_txt = make_advisory_styles(anom_status)
    df_hist_copy = df_hist.copy()
    df_hist_copy["anomaly_score_cache"] = [h.get("anomaly_score_cache", 0.0) for h in history]
    timeline_fig = create_timeline_fig(df_hist_copy, anomaly_evaluator.threshold, anom_score)
    eng_h = latest_row.get("engine_health", 100.0)
    hyd_h = latest_row.get("hydraulic_health", 100.0)
    struc_h = latest_row.get("structural_health", 100.0)
    env_h = latest_row.get("environmental_health", 100.0)

    subsystems_data = [
        ("Engine
        ("Engine
        ("Hydraulics", hyd_h),
        ("Airframe", struc_h),
        ("Avionics", 99.0),
        ("Fuel System", max(50.0, env_h - 10.0)),
        ("APU", 32.0 if _active_mode == "SUDDEN_ANOMALY" or "ecs_bleed_valve_fail" in _active_faults else 88.0)
    ]
    bars_list = []
    for name, score in subsystems_data:
        bar_color = "#05c46b" if score > 80 else "#ff9f1c" if score > 50 else "#ff3f34"
        bars_list.append(
            html.Div(
                style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "fontSize": "13px", "fontWeight": "600"},
                children=[
                    html.Div(name, style={"width": "110px", "color": "#8a9bb4", "fontFamily": "'Share Tech Mono', monospace"}),
                    html.Div(
                        style={"flexGrow": "1", "height": "6px", "backgroundColor": "#121926", "borderRadius": "3px", "margin": "0 15px", "position": "relative", "overflow": "hidden"},
                        children=[html.Div(style={"width": f"{score}%", "height": "100%", "backgroundColor": bar_color, "borderRadius": "3px", "transition": "width 0.4s ease-in-out"})]
                    ),
                    html.Div(f"{int(score)}", style={"width": "24px", "textAlign": "right", "color": bar_color, "fontFamily": "'Share Tech Mono', monospace", "fontWeight": "bold"})
                ]
            )
        )
    with _telemetry_lock:
        if _telemetry_history:
            _telemetry_history[-1]["anomaly_score_cache"] = anom_score
    rul_input = get_rul_input_window(history)
    rul_res = rul_ensemble.predict(rul_input)
    rul_val = rul_res["ensemble_rul"]
    rul_items = [
        ("ENGINE
        ("HYD PUMP · UNIT A", max(50, int(hyd_h * 3.5)), "WARNING" if hyd_h < 85 else None),
        ("APU STARTER MOTOR", 48 if _active_mode == "SUDDEN_ANOMALY" or "ecs_bleed_valve_fail" in _active_faults else 420, "REPLACE" if _active_mode == "SUDDEN_ANOMALY" or "ecs_bleed_valve_fail" in _active_faults else None),
        ("FUEL CONTROL VALVE", max(80, int(env_h * 2.2)), "WARNING" if env_h < 80 else None),
        ("AIRFRAME · FRAME 22", int(struc_h * 49), None)
    ]
    rul_cards = []
    for component, hours, status in rul_items:
        card_border = "#141c2c"
        hours_color = "#ffffff"
        status_badge = None
        if status == "REPLACE":
            card_border = "#ff3f34"
            hours_color = "#ff3f34"
            status_badge = html.Span("REPLACE", style={"backgroundColor": "rgba(255, 63, 52, 0.1)", "color": "#ff3f34", "fontSize": "9px", "fontWeight": "bold", "padding": "2px 6px", "borderRadius": "2px", "border": "1px solid
        elif status == "WARNING":
            card_border = "#ff9f1c"
            hours_color = "#ff9f1c"
            status_badge = html.Span("[!]", style={"color": "#ff9f1c", "fontSize": "11px", "fontFamily": "'Share Tech Mono', monospace", "fontWeight": "bold"})
        rul_cards.append(
            html.Div(
                style={"backgroundColor": "#0c1220", "border": f"1px solid {card_border}", "borderRadius": "4px", "padding": "10px 12px", "display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                children=[
                    html.Div([
                        html.Div(component, style={"fontSize": "10px", "fontWeight": "700", "color": "#4a5b78", "letterSpacing": "0.5px"}),
                        html.Div([
                            html.Span(f"{hours}", style={"fontSize": "18px", "fontWeight": "bold", "color": hours_color, "fontFamily": "'Share Tech Mono', monospace"}),
                            html.Span(" flight hours remaining", style={"fontSize": "10px", "color": "#4a5b78", "marginLeft": "4px"})
                        ])
                    ]),
                    html.Div(status_badge) if status_badge else html.Div()
                ]
            )
        )
    recs = RecommendationEngine.generate_recommendations(
        health_scores={"engine": eng_h, "hydraulic": hyd_h, "structural": struc_h, "environmental": env_h, "fatigue_cycles": int(latest_row.get("fatigue_cycles", 1250))},
        anomaly_status=anom_status,
        anomaly_score=anom_score,
        rul_info=rul_res,
        active_faults=_active_faults
    )
    alert_elements = []
    if recs:
        alert_elements.append(html.Div("COCKPIT ADVISORIES & FAULT REMEDIALS:", style={"fontSize": "10px", "fontWeight": "bold", "color": "#4a9bb4", "marginBottom": "8px", "letterSpacing": "1px"}))
        for r in recs[:2]:
            alert_elements.append(
                html.Div(
                    style={"fontSize": "11px", "color": "#ff7b72" if r["urgency"] == "CRITICAL" else "#ffa657", "marginBottom": "6px", "fontFamily": "monospace"},
                    children=[
                        html.Span(f"[{r['urgency']}] ", style={"fontWeight": "bold"}),
                        html.Span(f"{r['subsystem']}: {r['action']}")
                    ]
                )
            )

    return (
        advisory_dot_style,
        adv_txt,
        advisory_text_style,
        current_time_str,
        f"{vibe:.2f}",
        vibe_chg_str,
        vibe_chg_style,
        fig_vibe,
        f"{int(egt)}",
        egt_chg_str,
        egt_chg_style,
        fig_egt,
        f"{int(oil)}",
        oil_chg_str,
        oil_chg_style,
        card_oil_style,
        fig_oil,
        f"{int(hyd)}",
        hyd_chg_str,
        hyd_chg_style,
        fig_hyd,
        f"{int(stress)}",
        stress_chg_str,
        stress_chg_style,
        fig_stress,
        bars_list,
        timeline_fig,
        rul_cards,
        alert_elements
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
