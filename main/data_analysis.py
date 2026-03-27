"""
Master Thesis: Rocket Launch Failure Prediction Using Machine Learning
Chapter 4: Data Analysis - Python Script Template
This script generates all visualizations and tables for Chapter 4.
"""
# Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

print("=" * 80)
print("CHAPTER 4: DATA ANALYSIS - GENERATING VISUALIZATIONS AND TABLES")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/11] Loading dataset...")

# Get project root dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))

# If script is inside /main, go up one level to project root
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# Create output directories
figures_dir = os.path.join(project_root, "figures")
tables_dir = os.path.join(project_root, "tables")

os.makedirs(figures_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

# Build path to data file
file_path = os.path.join(
    project_root,
    "data",
    "Rocket_Launch_Industry_Dataset_Clean.xlsx"
)

# Try file path
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"Dataset not found at {file_path}. Please place it in the 'data' folder."
    )

# Check for required excel sheets
required_sheets = ['Launches', 'Configs', 'Families', 'Companies', 'Locations']
xls = pd.ExcelFile(file_path)

for sheet in required_sheets:
    if sheet not in xls.sheet_names:
        raise ValueError(f"Missing sheet: {sheet}")

# Load Excel sheets
launches_df = pd.read_excel(file_path, sheet_name='Launches')
configs_df = pd.read_excel(file_path, sheet_name='Configs')
families_df = pd.read_excel(file_path, sheet_name='Families')
companies_df = pd.read_excel(file_path, sheet_name='Companies')
locations_df = pd.read_excel(file_path, sheet_name='Locations')

print(f"✓ Loaded {len(launches_df)} launches")
print(f"✓ Date range: {launches_df['Launch Time'].min()} to {launches_df['Launch Time'].max()}")

# ============================================================================
# TABLE 4.1: DATASET STRUCTURE SUMMARY
# ============================================================================
print("\n[2/11] Creating Table 4.1: Dataset Structure...")

dataset_structure = pd.DataFrame({
    'Sheet Name': ['Launches', 'Configs', 'Families', 'Companies', 'Missions', 'Locations', 'Source'],
    'Rows': [6168, 480, 205, 59, 7449, 145, 0],
    'Columns': [16, 13, 8, 3, 4, 14, 1],
    'Description': [
        'Individual launch records with outcomes',
        'Rocket configuration specifications',
        'Rocket family historical performance',
        'Launch organization information',
        'Mission payload details',
        'Launch site geographic data',
        'Dataset source metadata'
    ]
})

dataset_structure.to_csv(os.path.join(tables_dir, "table_4_1_dataset_structure.csv"), index=False)
print("Table 4.1 saved")

# ============================================================================
# TABLE 4.2: TARGET VARIABLE DISTRIBUTION
# ============================================================================
print("\n[3/11] Creating Table 4.2: Target Variable Distribution...")

status_counts = launches_df['Launch Status'].value_counts()
status_pct = (status_counts / len(launches_df) * 100).round(2)

target_distribution = pd.DataFrame({
    'Launch Status': status_counts.index,
    'Count': status_counts.values,
    'Percentage': status_pct.values
})

# Add binary classification row
binary_success = status_counts.get('Success', 0)
binary_failure = status_counts.get('Failure', 0) + status_counts.get('Partial Failure', 0) + status_counts.get('Prelaunch Failure', 0)

binary_df = pd.DataFrame({
    'Launch Status': ['Success (Binary)', 'Failure (Binary)'],
    'Count': [binary_success, binary_failure],
    'Percentage': [(binary_success/len(launches_df)*100), (binary_failure/len(launches_df)*100)]
})

target_distribution = pd.concat([target_distribution, binary_df], ignore_index=True)
target_distribution.to_csv(os.path.join(tables_dir, "table_4_2_target_distribution.csv"), index=False)
print("Table 4.2 saved")
print(f"  Imbalance Ratio: {binary_success/binary_failure:.2f}:1")

# ============================================================================
# FIGURE 4.1: TARGET VARIABLE DISTRIBUTION
# ============================================================================
print("\n[4/11] Creating Figure 4.1: Target Variable Distribution...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 4-class distribution
colors_4class = ['#2ecc71', '#e74c3c', '#e67e22', '#c0392b']
ax1.bar(target_distribution['Launch Status'][:4], target_distribution['Count'][:4], color=colors_4class)
ax1.set_ylabel('Number of Launches')
ax1.set_title('(a) Four-Class Distribution')
ax1.tick_params(axis='x', rotation=45)
for i, (count, pct) in enumerate(zip(target_distribution['Count'][:4], target_distribution['Percentage'][:4])):
    ax1.text(i, count + 100, f'{count}\n({pct:.2f}%)', ha='center', va='bottom', fontsize=9)

# Binary distribution
colors_binary = ['#2ecc71', '#e74c3c']
ax2.bar(target_distribution['Launch Status'][4:], target_distribution['Count'][4:], color=colors_binary)
ax2.set_ylabel('Number of Launches')
ax2.set_title('(b) Binary Classification')
ax2.tick_params(axis='x', rotation=0)
for i, (count, pct) in enumerate(zip(target_distribution['Count'][4:], target_distribution['Percentage'][4:])):
    ax2.text(i, count + 100, f'{count}\n({pct:.2f}%)', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "my_figure.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 4.1 saved")

# ============================================================================
# FIGURE 4.2: LAUNCHES OVER TIME
# ============================================================================
print("\n[5/11] Creating Figure 4.2: Launches Over Time...")

yearly_counts = launches_df['Launch Year'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(yearly_counts.index, yearly_counts.values, linewidth=2, color='#3498db', marker='o', markersize=3)
ax.fill_between(yearly_counts.index, yearly_counts.values, alpha=0.3, color='#3498db')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Launches')
ax.set_title('Temporal Distribution of Rocket Launches (1957-2021)')
ax.grid(True, alpha=0.3)

# Annotate key periods
ax.axvspan(1957, 1991, alpha=0.1, color='red', label='Cold War Era')
ax.axvspan(2010, 2021, alpha=0.1, color='green', label='Commercial Space Era')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "my_figure.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 4.2 saved")

# ============================================================================
# FIGURE 4.3: SUCCESS RATE BY DECADE
# ============================================================================
print("\n[6/11] Creating Figure 4.3: Success Rate by Decade...")

# Calculate success rate by decade
launches_df['Decade'] = (launches_df['Launch Year'] // 10) * 10
decade_success = launches_df.groupby('Decade').apply(
    lambda x: (x['Launch Status'] == 'Success').sum() / len(x) * 100
).reset_index(name='Success Rate')

decade_counts = launches_df['Decade'].value_counts().sort_index()

fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar chart for success rate
color = '#2ecc71'
ax1.bar(decade_success['Decade'], decade_success['Success Rate'], color=color, alpha=0.7, label='Success Rate')
ax1.set_xlabel('Decade')
ax1.set_ylabel('Success Rate (%)', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0, 100])

# Line chart for launch count
ax2 = ax1.twinx()
color = '#3498db'
ax2.plot(decade_counts.index, decade_counts.values, color=color, marker='o', linewidth=2, label='Launch Count')
ax2.set_ylabel('Number of Launches', color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax1.set_title('Success Rate and Launch Frequency by Decade')
ax1.grid(True, alpha=0.3, axis='y')

# Add values on bars
for decade, rate in zip(decade_success['Decade'], decade_success['Success Rate']):
    ax1.text(decade, rate + 2, f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "my_figure.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 4.3 saved")

# ============================================================================
# TABLE 4.5 & FIGURE 4.4: TOP ORGANIZATIONS
# ============================================================================
print("\n[7/11] Creating Table 4.5 & Figure 4.4: Top Organizations...")

org_analysis = launches_df.groupby('Rocket Organisation').agg({
    'Launch Id': 'count',
    'Launch Status': lambda x: (x == 'Success').sum()
}).reset_index()
org_analysis.columns = ['Organization', 'Total Launches', 'Successes']
org_analysis['Success Rate (%)'] = (org_analysis['Successes'] / org_analysis['Total Launches'] * 100).round(2)
org_analysis = org_analysis.sort_values('Total Launches', ascending=False).head(10)

# Save table
org_analysis.to_csv(os.path.join(tables_dir, "table_4_5_top_organizations.csv"), index=False)
print("Table 4.5 saved")

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(org_analysis['Organization'], org_analysis['Total Launches'], color='#3498db')

# Color bars by success rate
for bar, success_rate in zip(bars, org_analysis['Success Rate (%)']):
    if success_rate >= 90:
        bar.set_color('#2ecc71')
    elif success_rate >= 80:
        bar.set_color('#f39c12')
    else:
        bar.set_color('#e74c3c')

ax.set_xlabel('Number of Launches')
ax.set_title('Top 10 Organizations by Launch Count (Colored by Success Rate)')
ax.invert_yaxis()

# Add values
for i, (launches, rate) in enumerate(zip(org_analysis['Total Launches'], org_analysis['Success Rate (%)'])):
    ax.text(launches + 50, i, f'{launches} ({rate:.1f}%)', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "my_figure.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 4.4 saved")

# ============================================================================
# TABLE 4.3 & FIGURE 4.5: MISSING DATA
# ============================================================================
print("\n[8/11] Creating Table 4.3 & Figure 4.5: Missing Data Analysis...")

missing_summary = []
for col in launches_df.columns:
    missing_count = launches_df[col].isna().sum()
    missing_pct = (missing_count / len(launches_df)) * 100
    if missing_count > 0:
        missing_summary.append({
            'Variable': col,
            'Missing Count': missing_count,
            'Missing Percentage': round(missing_pct, 2),
            'Non-Missing Count': len(launches_df) - missing_count
        })

missing_df = pd.DataFrame(missing_summary).sort_values('Missing Percentage', ascending=False)
missing_df.to_csv(os.path.join(tables_dir, "table_4_3_missing_data.csv"), index=False)
print("Table 4.3 saved")

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 6))
missing_matrix = launches_df[missing_df['Variable']].isna().astype(int)
sns.heatmap(missing_matrix.T, cmap='RdYlGn_r', cbar_kws={'label': 'Missing (1) / Present (0)'}, 
            yticklabels=missing_df['Variable'], ax=ax, cbar=True)
ax.set_xlabel('Launch Index (Sample)')
ax.set_title('Missing Data Pattern Across Variables')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "my_figure.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 4.5 saved")

# ============================================================================
# TABLE 4.4: DESCRIPTIVE STATISTICS
# ============================================================================
print("\n[9/11] Creating Table 4.4: Descriptive Statistics...")

numeric_cols = ['Rocket Price', 'Rocket Payload to LEO', 'Launch Year']
desc_stats = launches_df[numeric_cols].describe().T
desc_stats['Missing (%)'] = missing_df[missing_df['Variable'].isin(numeric_cols)].set_index('Variable')['Missing Percentage']
desc_stats = desc_stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'Missing (%)']]
desc_stats.to_csv(os.path.join(tables_dir, "table_4_4_descriptive_statistics.csv"))
print("Table 4.4 saved")

# ============================================================================
# FIGURE 4.6: PAYLOAD DISTRIBUTION
# ============================================================================
print("\n[10/11] Creating Figure 4.6: Payload Distribution...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
payload_data = launches_df['Rocket Payload to LEO'].dropna()
ax1.hist(payload_data, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Payload to LEO (kg)')
ax1.set_ylabel('Frequency')
ax1.set_title('(a) Histogram of Payload Capacity')
ax1.axvline(payload_data.median(), color='red', linestyle='--', label=f'Median: {payload_data.median():.0f} kg')
ax1.legend()

# Box plot by status
success_payload = launches_df[launches_df['Launch Status'] == 'Success']['Rocket Payload to LEO'].dropna()
failure_payload = launches_df[launches_df['Launch Status'] != 'Success']['Rocket Payload to LEO'].dropna()

ax2.boxplot([success_payload, failure_payload], labels=['Success', 'Failure'], patch_artist=True,
            boxprops=dict(facecolor='#2ecc71', alpha=0.7),
            medianprops=dict(color='red', linewidth=2))
ax2.set_ylabel('Payload to LEO (kg)')
ax2.set_title('(b) Payload Distribution by Launch Status')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "my_figure.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 4.6 saved")

# ============================================================================
# FIGURE 4.7: IMBALANCE VISUALIZATION
# ============================================================================
print("\n[11/11] Creating Figure 4.7: Class Imbalance Visualization...")

fig, ax = plt.subplots(figsize=(10, 6))

# Create comparison with other imbalanced datasets
datasets = ['Your Dataset\n(Rocket Launches)', 'Credit Card Fraud\n(Typical)', 'Medical Diagnosis\n(Typical)', 'Spam Detection\n(Typical)']
imbalance_ratios = [10.38, 577, 100, 4]
colors_imbalance = ['#e74c3c', '#c0392b', '#e67e22', '#f39c12']

bars = ax.bar(datasets, imbalance_ratios, color=colors_imbalance, alpha=0.8)
ax.set_ylabel('Imbalance Ratio (Majority:Minority)')
ax.set_title('Class Imbalance Comparison Across Datasets')
ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Moderate Imbalance Threshold')
ax.legend()

# Add values on bars
for bar, ratio in zip(bars, imbalance_ratios):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 10,
            f'{ratio:.2f}:1', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "my_figure.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 4.7 saved")

# ============================================================================
# FIGURE 4.8: CORRELATION HEATMAP
# ============================================================================
print("\n[Extra] Creating Figure 4.8: Correlation Heatmap...")

# Select numeric features for correlation
numeric_features = ['Rocket Price', 'Rocket Payload to LEO', 'Launch Year']
corr_data = launches_df[numeric_features].dropna()

fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix = corr_data.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "my_figure.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 4.8 saved")

# ============================================================================
# FIGURE 4.9: SUCCESS RATE BY ROCKET FAMILY
# ============================================================================
print("\n[Extra] Creating Figure 4.9: Success Rate by Rocket Family...")

# Merge launches with families data
launches_with_family = launches_df.merge(
    families_df[['Family', 'Success Rate']], 
    left_on='Rocket Name', 
    right_on='Family', 
    how='left'
)

family_analysis = launches_df.groupby('Rocket Name').agg({
    'Launch Id': 'count',
    'Launch Status': lambda x: (x == 'Success').sum()
}).reset_index()
family_analysis.columns = ['Rocket Family', 'Total Launches', 'Successes']
family_analysis['Success Rate (%)'] = (family_analysis['Successes'] / family_analysis['Total Launches'] * 100).round(2)
family_analysis = family_analysis[family_analysis['Total Launches'] >= 50].sort_values('Success Rate (%)', ascending=True).head(15)

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(family_analysis['Rocket Family'], family_analysis['Success Rate (%)'], color='#3498db')

# Color by success rate
for bar, rate in zip(bars, family_analysis['Success Rate (%)']):
    if rate >= 95:
        bar.set_color('#2ecc71')
    elif rate >= 85:
        bar.set_color('#f39c12')
    else:
        bar.set_color('#e74c3c')

ax.set_xlabel('Success Rate (%)')
ax.set_title('Success Rate by Rocket Family (Min 50 Launches)')
ax.axvline(x=90, color='gray', linestyle='--', alpha=0.5, label='90% Threshold')
ax.legend()
ax.invert_yaxis()

# Add values
for i, (rate, launches) in enumerate(zip(family_analysis['Success Rate (%)'], family_analysis['Total Launches'])):
    ax.text(rate + 1, i, f'{rate:.1f}% (n={launches})', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "my_figure.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 4.9 saved")

# ============================================================================
# FIGURE 4.10: GEOGRAPHIC DISTRIBUTION
# ============================================================================
print("\n[Extra] Creating Figure 4.10: Geographic Distribution...")

location_counts = launches_df['Location'].value_counts().head(15)

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(range(len(location_counts)), location_counts.values, color='#3498db')
ax.set_yticks(range(len(location_counts)))
ax.set_yticklabels([loc[:50] + '...' if len(loc) > 50 else loc for loc in location_counts.index], fontsize=8)
ax.set_xlabel('Number of Launches')
ax.set_title('Top 15 Launch Locations by Launch Count')
ax.invert_yaxis()

# Add values
for i, count in enumerate(location_counts.values):
    ax.text(count + 10, i, f'{count}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "my_figure.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 4.10 saved")

# ============================================================================
# SUMMARY STATISTICS FILE
# ============================================================================
print("\n[Final] Creating summary statistics file...")
with open(os.path.join(current_dir, "chapter_4_summary_stats.txt"), 'w') as f:
    f.write("CHAPTER 4: DATA ANALYSIS - SUMMARY STATISTICS\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("DATASET OVERVIEW\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total Launches: {len(launches_df)}\n")
    f.write(f"Date Range: {launches_df['Launch Time'].min()} to {launches_df['Launch Time'].max()}\n")
    f.write(f"Total Years: {launches_df['Launch Year'].max() - launches_df['Launch Year'].min() + 1}\n")
    f.write(f"Unique Organizations: {launches_df['Rocket Organisation'].nunique()}\n")
    f.write(f"Unique Locations: {launches_df['Location'].nunique()}\n")
    f.write(f"Unique Rocket Configurations: {launches_df['Rocket Name'].nunique()}\n\n")
    
    f.write("TARGET VARIABLE DISTRIBUTION\n")
    f.write("-" * 80 + "\n")
    f.write(f"Success: {binary_success} ({binary_success/len(launches_df)*100:.2f}%)\n")
    f.write(f"Failure (All Types): {binary_failure} ({binary_failure/len(launches_df)*100:.2f}%)\n")
    f.write(f"Imbalance Ratio: {binary_success/binary_failure:.2f}:1\n")
    f.write(f"Baseline Accuracy (Predict All Success): {binary_success/len(launches_df)*100:.2f}%\n\n")
    
    f.write("MISSING DATA SUMMARY\n")
    f.write("-" * 80 + "\n")
    for _, row in missing_df.iterrows():
        f.write(f"{row['Variable']}: {row['Missing Count']} ({row['Missing Percentage']:.2f}%)\n")
    f.write("\n")
    
    f.write("KEY INSIGHTS\n")
    f.write("-" * 80 + "\n")
    f.write("1. Moderate-to-severe class imbalance (10.38:1) justifies SMOTE\n")
    f.write("2. High missingness (63%) in economic variables requires exclusion\n")
    f.write("3. Success rate improved from ~85% (1960s) to ~95% (2010s)\n")
    f.write("4. High cardinality in categorical features (409 rocket types, 137 locations)\n")
    f.write("5. Temporal patterns necessitate chronological train-test split\n")

print("✓ Summary statistics saved")

print("\n" + "=" * 80)
print("ALL VISUALIZATIONS AND TABLES GENERATED SUCCESSFULLY!")
print("=" * 80)