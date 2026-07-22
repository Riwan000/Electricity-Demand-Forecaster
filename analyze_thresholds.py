"""
Analyze actual data to derive anomaly detection thresholds.

Run this once to understand your real data distribution and set evidence-based thresholds.
"""

import pandas as pd
import numpy as np

# Load historical data
df = pd.read_csv('data/final_merged_dataset.csv')

print("=" * 80)
print("THRESHOLD ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. FORECAST DEMAND THRESHOLDS
# ============================================================================
print("\n1. FORECAST DEMAND (MU)")
print("-" * 80)

demand_col = 'energy met (mu)'  # This is what the model predicts
energy_data = df[demand_col].dropna()

# Overall statistics
mean = energy_data.mean()
std_dev = energy_data.std()
min_val = energy_data.min()
max_val = energy_data.max()
q01 = energy_data.quantile(0.01)
q99 = energy_data.quantile(0.99)

print(f"Overall Demand (all states combined):")
print(f"  Mean:              {mean:.2f} MU")
print(f"  Std Dev:           {std_dev:.2f} MU")
print(f"  Min:               {min_val:.2f} MU")
print(f"  Max:               {max_val:.2f} MU")
print(f"  1st percentile:    {q01:.2f} MU")
print(f"  99th percentile:   {q99:.2f} MU")
print(f"\nRecommended thresholds:")
print(f"  ✅ Hard min:       {q01:.2f} MU (1st percentile)")
print(f"  ✅ Hard max:       {q99:.2f} MU (99th percentile)")
print(f"  ✅ Z-score:        2.0 (statistically >2σ = top 2.3% of distribution)")

# Per-state statistics
print(f"\nPer-state statistics:")
for state in df['states'].unique()[:5]:  # Show first 5 states
    state_data = df[df['states'] == state][demand_col].dropna()
    if len(state_data) > 10:
        print(f"  {state:20} mean={state_data.mean():.1f}±{state_data.std():.1f}, "
              f"range=[{state_data.min():.1f}, {state_data.max():.1f}]")

# ============================================================================
# 2. TEMPERATURE THRESHOLDS
# ============================================================================
print("\n\n2. WEATHER TEMPERATURE (Kelvin)")
print("-" * 80)

temp_col = '2m_temperature_max'  # Max temperature in Kelvin
temp_data = df[temp_col].dropna()

# Convert to Celsius for readability
temp_celsius = temp_data - 273.15
mean_c = temp_celsius.mean()
std_dev_c = temp_celsius.std()
min_c = temp_celsius.min()
max_c = temp_celsius.max()
q01_c = temp_celsius.quantile(0.01)
q99_c = temp_celsius.quantile(0.99)

print(f"Temperature (Kelvin → Celsius):")
print(f"  Mean:              {temp_data.mean():.2f}K ({mean_c:.2f}°C)")
print(f"  Std Dev:           {std_dev_c:.2f}°C")
print(f"  Min:               {temp_data.min():.2f}K ({min_c:.2f}°C)")
print(f"  Max:               {temp_data.max():.2f}K ({max_c:.2f}°C)")
print(f"  1st percentile:    {q01_c:.2f}°C")
print(f"  99th percentile:   {q99_c:.2f}°C")
print(f"\nRecommended thresholds:")
print(f"  ✅ Hard min:       {q01_c:.2f}°C ({q01_c + 273.15:.2f}K)")
print(f"  ✅ Hard max:       {q99_c:.2f}°C ({q99_c + 273.15:.2f}K)")
print(f"  ✅ Z-score:        2.5 (less strict than forecast due to natural variation)")

# ============================================================================
# 3. LATENCY THRESHOLDS (from Phase 5 if available)
# ============================================================================
print("\n\n3. LATENCY THRESHOLDS")
print("-" * 80)

try:
    metrics = pd.read_csv('logs/metrics.csv')
    latency = metrics[metrics['metric_name'] == 'forecast_latency_seconds']['value']
    if len(latency) > 0:
        print(f"Latency (from Phase 5 logs):")
        print(f"  Mean:              {latency.mean():.3f}s")
        print(f"  Std Dev:           {latency.std():.3f}s")
        print(f"  Min:               {latency.min():.3f}s")
        print(f"  Max:               {latency.max():.3f}s")
        print(f"  99th percentile:   {latency.quantile(0.99):.3f}s")
        print(f"\nRecommended thresholds:")
        print(f"  ✅ Hard max:       {latency.quantile(0.99):.3f}s (99th percentile)")
        print(f"  ✅ Z-score:        2.0")
    else:
        print("No latency metrics yet. Will use defaults: hard max=5.0s, Z-score=2.0")
except FileNotFoundError:
    print("logs/metrics.csv not found yet. Use defaults: hard max=5.0s, Z-score=2.0")

# ============================================================================
# 4. ERROR RATE THRESHOLDS (from Phase 5 if available)
# ============================================================================
print("\n\n4. ERROR RATE THRESHOLDS")
print("-" * 80)

try:
    errors = pd.read_csv('logs/errors.csv')
    metrics = pd.read_csv('logs/metrics.csv')

    total_attempts = len(metrics)
    error_count = len(errors)
    error_rate = error_count / total_attempts if total_attempts > 0 else 0

    print(f"Error rate (from Phase 5 logs):")
    print(f"  Total attempts:    {total_attempts}")
    print(f"  Total errors:      {error_count}")
    print(f"  Error rate:        {error_rate:.1%}")
    print(f"\nRecommended threshold:")
    print(f"  ✅ Hard max:       20% (>20% = definitely broken)")
    print(f"     Note: Your current baseline is {error_rate:.1%}, aim for <5%")
except FileNotFoundError:
    print("logs/ not found yet. Use default: hard max=20% error rate")

# ============================================================================
# 5. SUMMARY TABLE
# ============================================================================
print("\n\n" + "=" * 80)
print("RECOMMENDED THRESHOLDS (EVIDENCE-BASED)")
print("=" * 80)

print("""
┌──────────────────────┬─────────────────────────┬──────────────────────┐
│ Anomaly Type         │ Hard Threshold          │ Z-Score Threshold    │
├──────────────────────┼─────────────────────────┼──────────────────────┤
│ Forecast Demand      │ <""" + f"{q01:.1f}" + """ or >{q99:.1f} MU │ Z > 2.0              │
│ Weather Temperature  │ <""" + f"{q01_c:.1f}" + """°C or >{q99_c:.1f}°C │ Z > 2.5              │
│ Latency              │ > 5.0 seconds           │ Z > 2.0              │
│ Error Rate           │ > 20% in 10-min window  │ N/A                  │
└──────────────────────┴─────────────────────────┴──────────────────────┘
""")

# ============================================================================
# 6. INTERPRETATION GUIDE
# ============================================================================
print("\nINTERPRETATION:")
print("-" * 80)
print("""
WHY THESE VALUES?

Hard Thresholds (99th percentile):
  - Set at the extreme of historical data
  - Flags values that NEVER occurred before
  - Example: If max temp ever was 45°C, flag 50°C
  - Very conservative (only extreme outliers)

Z-Score (2.0 standard deviations):
  - Flags values that are statistically unusual
  - 2.0σ = top 2.3% of a normal distribution
  - Less conservative than hard thresholds
  - Adapts to each state's characteristics

Why Z > 2.5 for weather vs Z > 2.0 for demand?
  - Temperature varies naturally (summer vs winter)
  - More variation is normal
  - Demand is more predictable
  - So we need bigger deviations to flag weather as anomalous

Error Rate 20%?
  - If 1 in 5 predictions fails = something is broken
  - Below 5% is healthy
  - 5-20% is degraded but still running
  - >20% = system is down
""")

print("\n" + "=" * 80)
