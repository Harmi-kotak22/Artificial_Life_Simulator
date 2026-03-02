# Disease Outbreak Forecaster - Simple Project Overview

## 🎯 What is This Project?

This is a **Disease Outbreak Forecasting System** - a tool that predicts how diseases spread through populations. Think of it like a weather forecast, but for disease outbreaks!

---

## 🧬 How Does This Relate to Artificial Life Simulation?

### The Connection

**Artificial Life (ALife)** studies life-like behaviors using computer simulations. Our project simulates how **living organisms (humans) interact** during a disease outbreak.

| ALife Concept | Our Implementation |
|---------------|-------------------|
| **Agents** | Individual people in a population |
| **Interactions** | Disease transmission between people |
| **Emergent Behavior** | Epidemic curves (growth → peak → decline) |
| **Environment** | Population size, density, interventions |
| **Evolution Over Time** | Day-by-day spread of infection |

### What We Simulate

We simulate a **virtual population** where:
- People get **exposed** to the disease
- They become **infected** and can spread it
- Eventually they **recover** (or not)
- The disease spreads like a living entity through the population

This is **artificial life** because we're modeling the **emergent behavior** of a disease spreading through a living population - the epidemic "lives", grows, peaks, and dies based on simple rules.

---

## 🔬 The Core Algorithm: SEIR Model

Our main simulation uses the **SEIR Compartmental Model** - a classic epidemiological model that divides the population into 4 groups:

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ SUSCEPTIBLE │ ───► │   EXPOSED   │ ───► │  INFECTIOUS │ ───► │  RECOVERED  │
│     (S)     │      │     (E)     │      │     (I)     │      │     (R)     │
│             │      │             │      │             │      │             │
│ Can catch   │      │ Infected    │      │ Can spread  │      │ Immune now  │
│ the disease │      │ but not yet │      │ the disease │      │             │
│             │      │ contagious  │      │ to others   │      │             │
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
```

### The Math (Simplified)

Every day, the simulation calculates:

1. **New Exposures** = β × S × I / N
   - β = transmission rate (how easily disease spreads)
   - S = susceptible people
   - I = infectious people  
   - N = total population

2. **New Infections** = σ × E
   - σ = 1/(incubation period) ≈ 1/3 days

3. **New Recoveries** = γ × I
   - γ = 1/(infectious period) ≈ 1/7 days

### R₀ - The Magic Number

**R₀ (R-naught)** = Average number of people one infected person spreads to

- R₀ > 1 → Outbreak **grows** exponentially 📈
- R₀ = 1 → Outbreak **stable** (endemic)
- R₀ < 1 → Outbreak **dies out** 📉

**Formula:** `R₀ = β / γ`

---

## 📊 Important Algorithms Used

### 1. Monte Carlo Simulation (Ensemble Forecasting)

**What:** Run the simulation 500 times with slightly different parameters

**Why:** Creates a range of possible outcomes instead of just one prediction

```
Run 1:  R₀ = 2.3  →  Peak = 45,000 cases
Run 2:  R₀ = 2.5  →  Peak = 52,000 cases
Run 3:  R₀ = 2.1  →  Peak = 38,000 cases
...
Run 500: R₀ = 2.4 →  Peak = 48,000 cases

Result: "Peak will likely be 40,000-55,000 cases (80% confidence)"
```

### 2. Maximum Likelihood Estimation (MLE) - Auto-Fit

**What:** Automatically finds the best R₀ by analyzing real case data

**How:**
1. Take first 14-21 days of data
2. Fit exponential curve: `cases = a × e^(r×t)`
3. Calculate: `R₀ = 1 + (r / γ)`

**Example:**
```
Real data shows 10% daily growth
r = 0.10
γ = 1/7 = 0.143
R₀ = 1 + (0.10 / 0.143) = 1.7
```

### 3. Time-Varying R₀

**What:** R₀ decreases over time as people change behavior

**Formula:** `R₀(t) = R₀_initial × e^(-decay_rate × t)`

**Why:** Models real-world effects like:
- People wearing masks
- Social distancing
- Lockdowns
- Natural immunity building up

### 4. Log-Normal Parameter Uncertainty

**What:** Add realistic randomness to parameters

**Why:** In real life, we don't know exact transmission rates

```python
# Instead of: R₀ = 2.5 (exact)
# We use:     R₀ = 2.5 × random_lognormal(0, 0.35)
# Gives:      R₀ ranges from ~1.5 to ~4.0
```

### 5. Stochastic (Random) Transitions

**What:** Use Poisson random numbers for daily transitions

```python
# Instead of: new_infections = exactly σ × E
# We use:     new_infections = Poisson(σ × E)
```

**Why:** Real outbreaks have randomness - some days more spread, some less

---

## ✅ What We Have Completed

### Core Features (100% Done)

| Feature | Status | Description |
|---------|--------|-------------|
| SEIR Model | ✅ | Basic disease spread simulation |
| Monte Carlo Ensemble | ✅ | 500 simulations for uncertainty |
| Parameter Uncertainty | ✅ | Log-normal distributions (35% CV) |
| Stochastic Transitions | ✅ | Poisson random transmission |
| Intervention Modeling | ✅ | Vaccination, social distancing, lockdowns |
| Time-Varying R₀ | ✅ | Exponential decay of spread rate |

### Data Integration (100% Done)

| Feature | Status | Description |
|---------|--------|-------------|
| Johns Hopkins Data | ✅ | Real COVID-19 case data |
| Our World in Data | ✅ | International COVID data |
| CDC FluView | ✅ | Influenza data (simulated) |
| Custom Data Upload | ✅ | User's own CSV files |

### Validation System (100% Done)

| Feature | Status | Description |
|---------|--------|-------------|
| MAE (Mean Absolute Error) | ✅ | Average prediction error |
| RMSE | ✅ | Root mean squared error |
| Correlation | ✅ | How well forecast matches trend |
| Coverage | ✅ | % of real data within confidence interval |
| Bias | ✅ | Over/under prediction tendency |
| MAPE | ✅ | Mean absolute percentage error |
| Forecast Quality Score | ✅ | 0-100 overall score |

### Advanced Features (100% Done)

| Feature | Status | Description |
|---------|--------|-------------|
| MLE Auto-Fitting | ✅ | Automatic R₀ estimation from data |
| Detection Rate | ✅ | Account for unreported cases |
| Dynamic Widget Keys | ✅ | UI updates when data changes |

### User Interface (100% Done)

| Feature | Status | Description |
|---------|--------|-------------|
| Streamlit Dashboard | ✅ | Interactive web app |
| Disease Forecasting Page | ✅ | Create outbreak predictions |
| Intervention Page | ✅ | Model policy effects |
| Validation Page | ✅ | Compare with real data |
| Learning Center | ✅ | Educational content |

---

## 📈 Validation Metrics Explained

When comparing our forecast to real data, we measure:

### 1. Coverage (Target: 80%+)
"What % of real data points fall within our prediction interval?"

```
Good:  Real=45,000  Predicted=[40,000 - 55,000]  ✅ Within
Bad:   Real=70,000  Predicted=[40,000 - 55,000]  ❌ Outside
```

### 2. Correlation (Target: 0.7+)
"Does our forecast follow the same trend as reality?"

- +1.0 = Perfect match
- 0.0 = No relationship
- -1.0 = Opposite trend

### 3. MAE - Mean Absolute Error
"On average, how many cases off are we?"

```
MAE = Average of |Predicted - Actual|
Example: MAE = 5,000 means "off by ~5,000 cases per day"
```

### 4. Bias
"Do we consistently over-predict or under-predict?"

- Positive bias = We predict too high
- Negative bias = We predict too low

---

## 🎮 How to Use

### Basic Workflow

1. **Forecast Page** → Set disease, population, R₀, duration
2. **Generate Forecast** → See predicted cases with confidence bands
3. **Validation Page** → Fetch real data (Johns Hopkins)
4. **Compare** → See how well your forecast matches reality

### Tips for Good Results

1. **Use MLE Auto-Fit** - Let the algorithm find the best R₀
2. **Try Time-Varying R₀** - For data that peaks and declines
3. **Select growing phases** - SEIR works best on clear growth periods
4. **Check fit quality (R²)** - Values > 0.7 mean good exponential fit

---

## 🔮 Future Improvements (Not Yet Done)

- [ ] Spatial spread (city-to-city transmission)
- [ ] Age-structured populations
- [ ] Multiple disease strains
- [ ] Neural network hybrid models
- [ ] Other disaster types (floods, earthquakes)

---

## 📚 Key Formulas Summary

```
SEIR Transitions:
  S → E:  β × S × I / N  (exposure)
  E → I:  σ × E          (incubation complete)
  I → R:  γ × I          (recovery)

Basic Reproduction Number:
  R₀ = β / γ

Time-Varying R₀:
  R₀(t) = R₀_initial × exp(-decay × t)

MLE Estimation:
  r = slope of log(cases) vs time
  R₀ = 1 + (r / γ)

Coverage:
  % of actual values within [10th, 90th] percentile of predictions
```

---

*Last Updated: February 2026*
