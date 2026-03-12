"""
Agent-Based Disease Simulation Module
======================================
Two modes:
  1. Curve-Guided ABM  – agents' aggregate new-infection count tracks a
     target daily-case curve (from the forecast / validation ensemble).
     A proportional controller adjusts transmission probability each day.
  2. Custom (Emergent) ABM – pure agent simulation driven only by R₀,
     incubation, and infectious period sliders.  No external target.

Algorithm
---------
Each agent lives on a 100×100 continuous 2-D space and follows SEIR rules:
  • S → E  when a susceptible agent is within `infection_radius` of an
    infectious agent and a Bernoulli draw with probability
    `transmission_prob × proximity_factor` succeeds.
  • E → I  after `incubation_period` days (deterministic timer).
  • I → R  after `infectious_period` days (deterministic timer).

Movement: random walk with wall-bounce and 10 % per-step random direction
change, producing natural-looking Brownian-style motion.

Curve guidance (Mode 1):
  On each day d the controller computes:
      ratio = target_new_infections[d] / (actual_new_infections + 1)
  and scales `transmission_prob` by `ratio^α` where α = 0.6 (damping
  factor to avoid oscillation).  This makes the ABM's aggregate curve
  follow the forecast while keeping individual-level stochastic dynamics.
"""

import numpy as np
import plotly.graph_objects as go


# ── Colours & labels ────────────────────────────────────────────────
STATE_COLORS = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c', 3: '#3498db'}
STATE_NAMES  = {0: 'Susceptible', 1: 'Exposed', 2: 'Infectious', 3: 'Recovered'}


# ====================================================================
#  Core ABM engine
# ====================================================================
def run_agent_simulation(
    n_agents: int,
    sim_days: int,
    sim_R0: float,
    sim_incubation: int = 3,
    sim_infectious: int = 7,
    ifr: float = 0.01,
    target_curve: np.ndarray | None = None,
    seed: int = 42,
) -> list[dict]:
    """Run the spatial SEIR agent-based simulation.

    Parameters
    ----------
    n_agents : int
        Number of agents in the virtual space.
    sim_days : int
        Duration of the simulation in days.
    sim_R0 : float
        Basic reproduction number (used to compute base transmission
        probability).
    sim_incubation : int
        Latent period in days (E → I timer).
    sim_infectious : int
        Infectious period in days (I → R timer).
    ifr : float
        Infection Fatality Rate (0-1).  When an agent leaves state I,
        it dies with probability ``ifr`` and recovers otherwise.
    target_curve : np.ndarray or None
        If provided, a 1-D array of length ``sim_days`` giving the
        *fraction* of agents that should become newly infected each day.
        The controller adjusts ``transmission_prob`` to track this curve.
        When ``None`` the simulation runs in pure emergent mode.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    frames : list[dict]
        One dict per simulated day with keys ``x``, ``y``, ``colors``,
        ``names``, ``day``, ``S``, ``E``, ``I``, ``R``, ``D``,
        ``new_infections``.
    """
    rng = np.random.RandomState(seed)

    # -- initialise agents -------------------------------------------
    pos_x = rng.uniform(5, 95, n_agents)
    pos_y = rng.uniform(5, 95, n_agents)
    vel_x = rng.uniform(-2, 2, n_agents)
    vel_y = rng.uniform(-2, 2, n_agents)

    states          = np.zeros(n_agents, dtype=int)   # 0=S
    exposure_timer  = np.zeros(n_agents)
    infection_timer = np.zeros(n_agents)

    # seed a few infectious agents
    n_initial = max(1, int(n_agents * 0.02))
    idx_inf   = rng.choice(n_agents, n_initial, replace=False)
    states[idx_inf]          = 2           # Infectious
    infection_timer[idx_inf] = sim_infectious

    # base transmission probability from R₀
    contacts_per_day = 5
    base_tp = min(0.5, sim_R0 / (contacts_per_day * sim_infectious))
    transmission_prob = base_tp

    infection_radius = 8.0
    damping_alpha    = 0.6        # controller damping for curve-guided mode

    frames: list[dict] = []

    for day in range(sim_days):
        # ---- record frame ------------------------------------------
        frame_colors = [STATE_COLORS[s] for s in states]
        frame_names  = [STATE_NAMES[s]  for s in states]

        frames.append({
            'x': pos_x.copy(),
            'y': pos_y.copy(),
            'colors': frame_colors,
            'names':  frame_names,
            'day': day,
            'S': int(np.sum(states == 0)),
            'E': int(np.sum(states == 1)),
            'I': int(np.sum(states == 2)),
            'R': int(np.sum(states == 3)),
            'D': int(np.sum(states == 4)),
            'new_infections': 0,       # filled below after transmission
        })

        # ---- movement (dead agents don't move) ----------------------
        alive = states != 4
        pos_x[alive] += vel_x[alive]
        pos_y[alive] += vel_y[alive]

        # bounce off walls
        oob_x = (pos_x < 0) | (pos_x > 100)
        oob_y = (pos_y < 0) | (pos_y > 100)
        vel_x[oob_x] *= -1
        vel_y[oob_y] *= -1
        pos_x = np.clip(pos_x, 0, 100)
        pos_y = np.clip(pos_y, 0, 100)

        # random direction jitter (10 % chance each step)
        jitter = rng.random(n_agents) < 0.1
        vel_x[jitter] = rng.uniform(-2, 2, int(jitter.sum()))
        vel_y[jitter] = rng.uniform(-2, 2, int(jitter.sum()))

        # ---- disease transmission ----------------------------------
        inf_idx = np.where(states == 2)[0]
        sus_idx = np.where(states == 0)[0]

        new_infections_today = 0

        for i in inf_idx:
            ix, iy = pos_x[i], pos_y[i]
            for s in sus_idx:
                dx = ix - pos_x[s]
                dy = iy - pos_y[s]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < infection_radius:
                    prox = 1.0 - dist / infection_radius
                    if rng.random() < transmission_prob * prox:
                        states[s] = 1   # S → E
                        exposure_timer[s] = sim_incubation
                        new_infections_today += 1

        # update this day's count in the frame
        frames[-1]['new_infections'] = new_infections_today

        # ---- curve-guided controller --------------------------------
        if target_curve is not None and day < len(target_curve):
            target_new = target_curve[day] * n_agents   # target as count
            ratio = (target_new + 1) / (new_infections_today + 1)
            # clamp ratio so transmission_prob stays sensible
            ratio = np.clip(ratio, 0.2, 5.0)
            transmission_prob = np.clip(
                base_tp * (ratio ** damping_alpha), 0.001, 0.6
            )

        # refresh susceptible index (some just became exposed)
        sus_idx = np.where(states == 0)[0]

        # ---- state transitions E → I --------------------------------
        exp_idx = np.where(states == 1)[0]
        exposure_timer[exp_idx] -= 1
        newly_infectious = exp_idx[exposure_timer[exp_idx] <= 0]
        states[newly_infectious] = 2
        infection_timer[newly_infectious] = sim_infectious

        # ---- state transitions I → R or I → D ----------------------
        infection_timer[inf_idx] -= 1
        leaving_I = inf_idx[infection_timer[inf_idx] <= 0]
        for agent in leaving_I:
            if rng.random() < ifr:
                states[agent] = 4   # Dead
                vel_x[agent] = 0
                vel_y[agent] = 0
            else:
                states[agent] = 3   # Recovered

    return frames


# ====================================================================
#  Build the target-curve fraction array from forecast / validation
# ====================================================================

def build_target_curve_from_forecast(forecast_results: dict,
                                     n_agents: int) -> np.ndarray:
    """Convert a forecast's ``new_cases`` ensemble into a daily fraction
    target for the ABM controller.

    Parameters
    ----------
    forecast_results : dict
        Must contain ``'new_cases'`` (2-D array, sims × days) and
        ``'population'`` (int).
    n_agents : int
        Number of agents in the simulation (used for scaling).

    Returns
    -------
    target_fractions : np.ndarray  (length = forecast days)
        Fraction of agents that should become newly infected each day.
    """
    new_cases = np.array(forecast_results['new_cases'])
    population = forecast_results.get('population', 1_000_000)

    # mean daily new cases across all ensemble simulations
    if new_cases.ndim == 2:
        mean_daily = np.mean(new_cases, axis=0)
    else:
        mean_daily = new_cases

    # convert absolute count → fraction of population
    daily_fraction = mean_daily / max(population, 1)

    # scale so the peak fraction maps to something reachable by n_agents
    # e.g. if peak fraction is 0.001 of 330M pop, re-normalise so peak
    # is ~30-40 % of agents (visually dramatic but not 100 %)
    peak_frac = np.max(daily_fraction)
    if peak_frac > 0:
        desired_peak = 0.35   # 35 % of agents at peak
        scale = desired_peak / peak_frac
        daily_fraction = daily_fraction * scale

    # clamp
    daily_fraction = np.clip(daily_fraction, 0.0, 0.6)
    return daily_fraction


def build_target_curve_from_validation(validation_data,
                                        n_agents: int) -> np.ndarray:
    """Convert validation real-case data into a daily fraction target.

    Parameters
    ----------
    validation_data : pd.DataFrame
        Must contain ``'new_cases'`` column.
    n_agents : int
        Number of agents in the simulation.

    Returns
    -------
    target_fractions : np.ndarray
    """
    cases = validation_data['new_cases'].values.astype(float)
    cases = np.nan_to_num(cases, nan=0.0)
    cases = np.clip(cases, 0, None)

    peak = np.max(cases)
    if peak > 0:
        daily_fraction = cases / peak * 0.35   # peak → 35 % of agents
    else:
        daily_fraction = np.zeros(len(cases))

    daily_fraction = np.clip(daily_fraction, 0.0, 0.6)
    return daily_fraction


# ====================================================================
#  Plotly figure builders
# ====================================================================

def build_animation_figure(frames_data: list[dict],
                           animation_speed: int = 150) -> go.Figure:
    """Build a Plotly animated scatter figure from simulation frames.

    Parameters
    ----------
    frames_data : list[dict]
        Output of :func:`run_agent_simulation`.
    animation_speed : int
        Milliseconds per frame in the animation player.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    initial = frames_data[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=initial['x'], y=initial['y'],
        mode='markers',
        marker=dict(size=12, color=initial['colors'],
                    line=dict(width=1, color='white')),
        text=initial['names'],
        hovertemplate='%{text}<extra></extra>',
    ))

    anim_frames = []
    for f in frames_data:
        anim_frames.append(go.Frame(
            data=[go.Scatter(
                x=f['x'], y=f['y'],
                mode='markers',
                marker=dict(size=12, color=f['colors'],
                            line=dict(width=1, color='white')),
                text=f['names'],
                hovertemplate='%{text}<extra></extra>',
            )],
            name=str(f['day']),
            layout=go.Layout(
                title=f"Day {f['day']+1} | "
                      f"🟢S:{f['S']} 🟡E:{f['E']} "
                      f"🔴I:{f['I']} 🔵R:{f['R']} "
                      f"⚫D:{f['D']}"
            ),
        ))
    fig.frames = anim_frames

    fig.update_layout(
        title=(f"Day 1 | 🟢S:{initial['S']} 🟡E:{initial['E']} "
               f"🔴I:{initial['I']} 🔵R:{initial['R']} "
               f"⚫D:{initial['D']}"),
        xaxis=dict(range=[0, 100], showgrid=False,
                   zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, 100], showgrid=False,
                   zeroline=False, showticklabels=False,
                   scaleanchor='x'),
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#16213e',
        font=dict(color='white'),
        height=600,
        updatemenus=[dict(
            type='buttons', showactive=False,
            y=1.15, x=0.5, xanchor='center',
            buttons=[
                dict(label='▶ Play', method='animate',
                     args=[None, {
                         'frame': {'duration': animation_speed,
                                   'redraw': True},
                         'fromcurrent': True,
                         'transition': {'duration': 50},
                     }]),
                dict(label='⏸ Pause', method='animate',
                     args=[[None], {
                         'frame': {'duration': 0, 'redraw': False},
                         'mode': 'immediate',
                         'transition': {'duration': 0},
                     }]),
            ],
        )],
        sliders=[{
            'active': 0,
            'yanchor': 'top', 'xanchor': 'left',
            'currentvalue': {'prefix': 'Day: ', 'visible': True,
                             'xanchor': 'center'},
            'transition': {'duration': 50},
            'pad': {'b': 10, 't': 50},
            'len': 0.9, 'x': 0.05, 'y': 0,
            'steps': [
                {'args': [[str(f['day'])],
                          {'frame': {'duration': 50, 'redraw': True},
                           'mode': 'immediate'}],
                 'label': str(f['day'] + 1),
                 'method': 'animate'}
                for f in frames_data
            ],
        }],
    )
    return fig


def build_seir_curves_figure(frames_data: list[dict]) -> go.Figure:
    """Build a line chart of SEIR compartment counts over time."""
    days = [f['day'] + 1 for f in frames_data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=[f['S'] for f in frames_data],
                             name='Susceptible',
                             line=dict(color='#2ecc71', width=2)))
    fig.add_trace(go.Scatter(x=days, y=[f['E'] for f in frames_data],
                             name='Exposed',
                             line=dict(color='#f39c12', width=2)))
    fig.add_trace(go.Scatter(x=days, y=[f['I'] for f in frames_data],
                             name='Infectious',
                             line=dict(color='#e74c3c', width=2)))
    fig.add_trace(go.Scatter(x=days, y=[f['R'] for f in frames_data],
                             name='Recovered',
                             line=dict(color='#3498db', width=2)))
    fig.add_trace(go.Scatter(x=days, y=[f['D'] for f in frames_data],
                             name='Dead',
                             line=dict(color='#7f8c8d', width=2,
                                       dash='dot')))
    fig.update_layout(
        title='SEIRD Compartment Counts from Agent Simulation',
        xaxis_title='Day',
        yaxis_title='Number of Agents',
        template='plotly_dark',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return fig


def compute_statistics(frames_data: list[dict], n_agents: int) -> dict:
    """Derive summary statistics from simulation frames."""
    I_counts = [f['I'] for f in frames_data]
    R_counts = [f['R'] for f in frames_data]
    E_counts = [f['E'] for f in frames_data]
    D_counts = [f['D'] for f in frames_data]

    peak_infected = max(I_counts)
    peak_day      = I_counts.index(peak_infected) + 1
    total_infected = R_counts[-1] + I_counts[-1] + E_counts[-1] + D_counts[-1]
    total_deaths   = D_counts[-1]
    mortality_rate = (total_deaths / max(total_infected, 1)) * 100
    attack_rate    = (total_infected / n_agents) * 100

    return {
        'peak_infected':  peak_infected,
        'peak_day':       peak_day,
        'total_infected': total_infected,
        'total_deaths':   total_deaths,
        'mortality_rate': mortality_rate,
        'attack_rate':    attack_rate,
    }
