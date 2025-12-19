"""
Athlete Performance Analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Load data
print("Loading datasets...")
athletes = pd.read_csv('athletes.csv')
training = pd.read_csv('training_data.csv')
competition = pd.read_csv('competition_data.csv')
wellness = pd.read_csv('wellness_data.csv')

# Convert dates
training['date'] = pd.to_datetime(training['date'])
competition['date'] = pd.to_datetime(competition['date'])
wellness['date'] = pd.to_datetime(wellness['date'])

print(f"Data loaded successfully!")
print(f"  Athletes: {len(athletes)}")
print(f"  Training sessions: {len(training)}")
print(f"  Wellness responses: {len(wellness)}")
print(f"  Competition results: {len(competition)}")
print(f"  Study period: {training['date'].min().strftime('%B %d')} - {training['date'].max().strftime('%B %d, %Y')}")
print(f"  Compliance rate: {len(wellness)/(70*12)*100:.1f}%\n")

# EXPLORATORY DATA ANALYSIS

def create_eda_plots():
    """Generate comprehensive EDA visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Load and Performance Patterns', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    session_stats = training.groupby('session_type')['training_load'].agg(['mean', 'std', 'count'])
    session_stats[['mean', 'std']].plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'])
    ax.set_title('Training Load by Session Type')
    ax.set_xlabel('Session Type')
    ax.set_ylabel('Training Load (1-100 scale)')
    ax.legend(['Mean', 'Std Dev'])
    ax.grid(True, alpha=0.3)
    
    for i, count in enumerate(session_stats['count']):
        ax.text(i, session_stats['mean'].iloc[i] + 5, f'n={int(count)}', 
                ha='center', va='bottom', fontsize=9)
    
    ax = axes[0, 1]
    weekly_load = training.groupby('week_number')['training_load'].mean()
    ax.plot(weekly_load.index, weekly_load.values, marker='o', linewidth=2, markersize=8, color='#2ecc71')
    ax.set_title('Average Training Load Progression (10 Weeks)')
    ax.set_xlabel('Week Number')
    ax.set_ylabel('Average Training Load')
    ax.set_xticks(range(1, 11))
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    training_100m = training[training['100m_time'].notna()]['100m_time']
    ax.hist(training_100m, bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax.axvline(training_100m.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {training_100m.mean():.2f}s')
    ax.set_title(f'100m Training Times Distribution (n={len(training_100m)})')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    training_200m = training[training['200m_time'].notna()]['200m_time']
    ax.hist(training_200m, bins=20, color='#e67e22', alpha=0.7, edgecolor='black')
    ax.axvline(training_200m.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {training_200m.mean():.2f}s')
    ax.set_title(f'200m Training Times Distribution (n={len(training_200m)})')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure1_training_patterns.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 1: Training patterns saved")
    plt.close()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Wellness Metrics Time Series and Distributions', fontsize=16, fontweight='bold')
    
    wellness_metrics = ['sleep_hours', 'fatigue', 'soreness', 'mood', 'readiness', 'rpe']
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#1abc9c', '#9b59b6']
    
    for idx, (metric, color) in enumerate(zip(wellness_metrics, colors)):
        ax = axes[idx // 3, idx % 3]
        
        daily_avg = wellness.groupby('date')[metric].mean().reset_index()
        daily_avg = daily_avg.sort_values('date')
        
        ax.plot(daily_avg['date'], daily_avg[metric], color=color, linewidth=2, alpha=0.7)
        ax.fill_between(daily_avg['date'], daily_avg[metric], alpha=0.3, color=color)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('figure2_wellness_time_series.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 2: Wellness time series saved")
    plt.close()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    wellness_corr = wellness[wellness_metrics].corr()
    
    sns.heatmap(wellness_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Wellness Metrics Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('figure3_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 3: Correlation matrix saved")
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Historic Competition Performance (2023-2025)', fontsize=16, fontweight='bold')
    
    ax = axes[0]
    comp_100m = competition[competition['event'] == '100m'].copy()
    comp_100m = comp_100m.sort_values('date')
    
    if len(comp_100m) > 0:
        for athlete_id in comp_100m['athlete_id'].unique():
            athlete_data = comp_100m[comp_100m['athlete_id'] == athlete_id]
            if len(athlete_data) > 1:
                ax.plot(athlete_data['date'], athlete_data['time'], marker='o', alpha=0.4, linewidth=1)
        
        avg_100m = comp_100m.groupby('date')['time'].mean()
        ax.plot(avg_100m.index, avg_100m.values, marker='s', color='black', 
                linewidth=3, markersize=10, label='Average', zorder=10)
        
        ax.set_title(f'100m Performance Progression (n={len(comp_100m)} races)')
        ax.set_xlabel('Competition Date')
        ax.set_ylabel('Time (seconds)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    ax = axes[1]
    comp_200m = competition[competition['event'] == '200m'].copy()
    comp_200m = comp_200m.sort_values('date')
    
    if len(comp_200m) > 0:
        # Plot individual athlete progressions
        for athlete_id in comp_200m['athlete_id'].unique():
            athlete_data = comp_200m[comp_200m['athlete_id'] == athlete_id]
            if len(athlete_data) > 1:
                ax.plot(athlete_data['date'], athlete_data['time'], marker='o', alpha=0.4, linewidth=1)
        
        avg_200m = comp_200m.groupby('date')['time'].mean()
        ax.plot(avg_200m.index, avg_200m.values, marker='s', color='black', 
                linewidth=3, markersize=10, label='Average', zorder=10)
        
        ax.set_title(f'200m Performance Progression (n={len(comp_200m)} races)')
        ax.set_xlabel('Competition Date')
        ax.set_ylabel('Time (seconds)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('figure4_competition_progression.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 4: Competition progression saved")
    plt.close()

# HIERARCHICAL MIXED EFFECTS MODELING

def prepare_modeling_data():
    """Merge datasets and prepare for hierarchical modeling"""
    
    print("\n" + "=" * 80)
    print("=" * 80)
    
    training_wellness = pd.merge(
        training,
        wellness,
        on=['athlete_id', 'date'],
        how='left',
        suffixes=('_training', '_wellness')
    )
    
    if 'rpe_training' in training_wellness.columns:
        training_wellness['rpe'] = training_wellness['rpe_training']
        training_wellness = training_wellness.drop(['rpe_training', 'rpe_wellness'], axis=1, errors='ignore')
    
    training_wellness = pd.merge(
        training_wellness,
        athletes[['athlete_id', 'Athlete', '100m_PB', '200m_PB']],
        on='athlete_id',
        how='left'
    )
    
    print(f"\nMerged dataset: {len(training_wellness)} observations")
    print(f"Athletes: {training_wellness['athlete_id'].nunique()}")
    print(f"Date range: {training_wellness['date'].min().strftime('%Y-%m-%d')} to {training_wellness['date'].max().strftime('%Y-%m-%d')}")
    
    return training_wellness

def hierarchical_model_analysis(model_data):
    
    
    print("\n" + "=" * 80)
    print("HIERARCHICAL MIXED EFFECTS MODEL ANALYSIS")
    print("=" * 80)
    
    model_data = model_data[model_data['100m_time'].notna()].copy()
    
    print(f"\nObservations with 100m times: {len(model_data)}")
    print(f"Athletes represented: {model_data['athlete_id'].nunique()}")
    
    features = [
        'training_load',
        'readiness',
        'fatigue',
        'sleep_hours',
        'soreness',
        'mood',
        'stress',
        'resting_hr'
    ]
    
    missing_counts = model_data[features + ['100m_time']].isnull().sum()
    if missing_counts.sum() > 0:
        print("\nMissing values detected:")
        print(missing_counts[missing_counts > 0])
        print("\nFilling missing values with median...")
        for col in features:
            if model_data[col].isnull().sum() > 0:
                model_data[col].fillna(model_data[col].median(), inplace=True)
    
    X = model_data[features].copy()
    y = model_data['100m_time'].copy()
    athlete_ids = model_data['athlete_id'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=X.index)
    
    
    global_model = Ridge(alpha=1.0, random_state=42)
    global_model.fit(X_scaled, y)
    
    cv_scores = cross_val_score(global_model, X_scaled, y, cv=5, scoring='r2')
    
    global_pred = global_model.predict(X_scaled)
    global_r2 = global_model.score(X_scaled, y)
    
    print(f"\nGlobal Model R²: {global_r2:.4f}")
    print(f"Cross-Validation R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    print("\nFeature Coefficients (Standardized):")
    print("-" * 40)
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': global_model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    for _, row in coef_df.iterrows():
        print(f"  {row['Feature']:20s}: {row['Coefficient']:7.4f}")
    
    # ATHLETE-SPECIFIC EFFECTS 
    
    print("\n" + "-" * 80)
    print("ATHLETE-SPECIFIC EFFECTS (Random Intercepts)")
    print("-" * 80)
    
    athlete_effects = {}
    athlete_predictions = np.zeros(len(y))
    
    for athlete_id in np.unique(athlete_ids):
        athlete_mask = athlete_ids == athlete_id
        athlete_y = y[athlete_mask]
        athlete_X = X_scaled[athlete_mask]
        
        athlete_global_pred = global_model.predict(athlete_X)
        
        athlete_effect = (athlete_y - athlete_global_pred).mean()
        athlete_effects[athlete_id] = athlete_effect
        
        athlete_predictions[athlete_mask] = athlete_global_pred + athlete_effect
    
    mixed_r2 = 1 - (np.sum((y - athlete_predictions)**2) / np.sum((y - y.mean())**2))
    
    print(f"\nMixed Effects Model R²: {mixed_r2:.4f}")
    print(f"Improvement over fixed effects: {(mixed_r2 - global_r2):.4f}")
    
    print("\nAthlete-Specific Effects (Random Intercepts):")
    print("-" * 40)
    athlete_effects_df = pd.DataFrame([
        {'athlete_id': aid, 'effect': eff} 
        for aid, eff in athlete_effects.items()
    ]).sort_values('effect')
    
    athlete_effects_df = athlete_effects_df.merge(
        athletes[['athlete_id', 'Athlete']], 
        on='athlete_id'
    )
    
    for _, row in athlete_effects_df.iterrows():
        print(f"  {row['Athlete']:25s}: {row['effect']:+7.3f}s")
    
    # VARIANCE DECOMPOSITION
    
    print("\n" + "-" * 80)
    print("VARIANCE DECOMPOSITION ANALYSIS")
    print("-" * 80)
    
    total_var = np.var(y)
    
    between_var = np.var(list(athlete_effects.values()))
    
    # Within-athlete variance (residual variance after accounting for athlete effects)
    residuals = y - athlete_predictions
    within_var = np.var(residuals)
    
    print(f"\nTotal Variance: {total_var:.4f}")
    print(f"Between-Athlete Variance: {between_var:.4f} ({between_var/total_var*100:.1f}%)")
    print(f"Within-Athlete Variance: {within_var:.4f} ({within_var/total_var*100:.1f}%)")
    
    icc = between_var / (between_var + within_var)
    print(f"\nIntraclass Correlation Coefficient (ICC): {icc:.4f}")
    print(f"Interpretation: {icc*100:.1f}% of variance is between-athlete differences")
    print(f"                {(1-icc)*100:.1f}% of variance is within-athlete variation")
    
    fixed_var_explained = global_r2 * within_var
    print(f"\nWellness Variables Explain:")
    print(f"  {fixed_var_explained/within_var*100:.1f}% of within-athlete variance")
    print(f"  {fixed_var_explained/total_var*100:.1f}% of total variance")
    
    return {
        'model_data': model_data,
        'global_model': global_model,
        'athlete_effects': athlete_effects,
        'athlete_effects_df': athlete_effects_df,
        'features': features,
        'scaler': scaler,
        'predictions': athlete_predictions,
        'actuals': y.values,
        'global_r2': global_r2,
        'mixed_r2': mixed_r2,
        'icc': icc,
        'cv_scores': cv_scores
    }

def create_model_plots(model_results):
    """Create visualizations for model results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hierarchical Model Performance Analysis', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    ax.scatter(model_results['actuals'], model_results['predictions'], 
               alpha=0.6, s=50, color='#3498db')
    
    min_val = min(model_results['actuals'].min(), model_results['predictions'].min())
    max_val = max(model_results['actuals'].max(), model_results['predictions'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual 100m Time (s)')
    ax.set_ylabel('Predicted 100m Time (s)')
    ax.set_title(f'Actual vs Predicted (Mixed R² = {model_results["mixed_r2"]:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    residuals = model_results['actuals'] - model_results['predictions']
    ax.hist(residuals, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual (s)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Residuals Distribution (RMSE = {np.sqrt(np.mean(residuals**2)):.3f}s)')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.scatter(model_results['predictions'], residuals, alpha=0.6, s=50, color='#2ecc71')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Fitted Values (s)')
    ax.set_ylabel('Residuals (s)')
    ax.set_title('Residuals vs Fitted Values')
    ax.grid(True, alpha=0.3)
    
    # Q-Q Plot
    ax = axes[1, 1]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Check)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure5_model_diagnostics.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 5: Model diagnostics saved")
    plt.close()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    feature_importance = pd.DataFrame({
        'Feature': model_results['features'],
        'Coefficient': model_results['global_model'].coef_
    }).sort_values('Coefficient', key=abs, ascending=True)
    
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in feature_importance['Coefficient']]
    feature_importance.plot(x='Feature', y='Coefficient', kind='barh', 
                           ax=ax, color=colors, legend=False)
    
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Standardized Coefficient')
    ax.set_title('Feature Importance (Fixed Effects Model)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('figure6_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 6: Feature importance saved")
    plt.close()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    effects_df = model_results['athlete_effects_df'].sort_values('effect')
    colors = ['#2ecc71' if x < 0 else '#e74c3c' for x in effects_df['effect']]
    
    ax.barh(range(len(effects_df)), effects_df['effect'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(effects_df)))
    ax.set_yticklabels(effects_df['Athlete'], fontsize=9)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Effect on 100m Time (seconds)')
    ax.set_title('Athlete-Specific Random Effects (Relative to Population Mean)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('figure6b_athlete_effects.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 6b: Athlete-specific effects saved")
    plt.close()

def wellness_performance_analysis():
    """Analyze relationship between wellness and performance"""
    
    print("\n" + "=" * 80)
    print("WELLNESS-PERFORMANCE CORRELATION ANALYSIS")
    print("=" * 80)
    
    comp_wellness = []
    
    for _, comp in competition.iterrows():
        athlete_id = comp['athlete_id']
        comp_date = comp['date']
        
        # Get wellness from 1 day before competition
        wellness_before = wellness[
            (wellness['athlete_id'] == athlete_id) &
            (wellness['date'] == comp_date - pd.Timedelta(days=1))
        ]
        
        if len(wellness_before) > 0:
            comp_wellness.append({
                **comp.to_dict(),
                **wellness_before.iloc[0][['sleep_hours', 'fatigue', 'soreness', 'mood', 'rpe', 'readiness']].to_dict()
            })
    
    comp_wellness_df = pd.DataFrame(comp_wellness)
    
    if len(comp_wellness_df) == 0:
        print("\nNo competition-wellness matches found (competitions outside study period)")
        return
    
    comp_100m = comp_wellness_df[comp_wellness_df['event'] == '100m']
    comp_200m = comp_wellness_df[comp_wellness_df['event'] == '200m']
    
    wellness_vars = ['sleep_hours', 'fatigue', 'soreness', 'mood', 'rpe', 'readiness']
    
    print(f"\nCompetition-Wellness Matches Found: {len(comp_wellness_df)}")
    print(f"  100m: {len(comp_100m)}")
    print(f"  200m: {len(comp_200m)}")
    
    print("\nCorrelations with Competition Performance:")
    
    if len(comp_100m) > 5:
        print("\n100m Event:")
        print("-" * 40)
        for var in wellness_vars:
            if var in comp_100m.columns and len(comp_100m[var].dropna()) > 0:
                corr = comp_100m[['time', var]].corr().iloc[0, 1]
                print(f"  {var:15s}: {corr:7.3f}")
    
    if len(comp_200m) > 5:
        print("\n200m Event:")
        print("-" * 40)
        for var in wellness_vars:
            if var in comp_200m.columns and len(comp_200m[var].dropna()) > 0:
                corr = comp_200m[['time', var]].corr().iloc[0, 1]
                print(f"  {var:15s}: {corr:7.3f}")
    
    if len(comp_wellness_df) > 10:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Wellness Indicators vs Competition Performance', fontsize=16, fontweight='bold')
        
        for idx, var in enumerate(wellness_vars):
            ax = axes[idx // 3, idx % 3]
            
            if len(comp_100m) > 0 and var in comp_100m.columns:
                ax.scatter(comp_100m[var], comp_100m['time'], alpha=0.6, s=60, 
                          label='100m', color='#3498db')
            
            if len(comp_200m) > 0 and var in comp_200m.columns:
                ax.scatter(comp_200m[var], comp_200m['time'], alpha=0.6, s=60, 
                          label='200m', color='#e74c3c')
            
            ax.set_xlabel(var.replace('_', ' ').title())
            ax.set_ylabel('Time (seconds)')
            ax.set_title(f'{var.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figure7_wellness_performance.png', dpi=300, bbox_inches='tight')
        print("\n✓ Figure 7: Wellness-performance relationships saved")
        plt.close()

# SUMMARY STATISTICS

def print_summary_statistics():
    """Print comprehensive summary statistics"""
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print("\n1. TRAINING SESSION SUMMARY")
    print("-" * 40)
    print(f"Total sessions: {len(training)}")
    print(f"Sessions per athlete: {len(training)/len(athletes):.1f} average")
    print(f"Study period: {training['date'].min().strftime('%B %d')} - {training['date'].max().strftime('%B %d, %Y')}")
    print(f"Duration: {(training['date'].max() - training['date'].min()).days} days ({training['week_number'].max()} weeks)")
    
    print("\nSession Types:")
    session_counts = training['session_type'].value_counts()
    for session_type, count in session_counts.items():
        print(f"  {session_type:15s}: {count:3d} ({count/len(training)*100:.1f}%)")
    
    print("\n2. WELLNESS MONITORING SUMMARY")
    print("-" * 40)
    print(f"Total responses: {len(wellness)}")
    print(f"Compliance rate: {len(wellness)/(70*12)*100:.1f}%")
    print(f"Responses per athlete: {len(wellness)/len(athletes):.1f} average")
    
    wellness_metrics = ['sleep_hours', 'fatigue', 'soreness', 'mood', 'readiness', 'rpe']
    print("\nWellness Metrics (Mean ± SD):")
    for metric in wellness_metrics:
        mean_val = wellness[metric].mean()
        std_val = wellness[metric].std()
        print(f"  {metric.replace('_', ' ').title():15s}: {mean_val:5.2f} ± {std_val:4.2f}")
    
    print("\n3. COMPETITION SUMMARY")
    print("-" * 40)
    print(f"Total races: {len(competition)}")
    print(f"Athletes with competition data: {competition['athlete_id'].nunique()}")
    print(f"Competition period: {competition['date'].min().strftime('%B %Y')} - {competition['date'].max().strftime('%B %Y')}")
    
    print("\nEvents:")
    for event in competition['event'].unique():
        event_data = competition[competition['event'] == event]
        print(f"  {event}: {len(event_data)} races")
        print(f"    Mean time: {event_data['time'].mean():.2f}s")
        print(f"    Best time: {event_data['time'].min():.2f}s")

# MAIN

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ATHLETE PERFORMANCE ANALYTICS - STATISTICAL ANALYSIS")
    print("DATA 698 Capstone Project - Michael Robinson")
    print("=" * 80)
    
    print_summary_statistics()
    
    print("\nGenerating exploratory data analysis plots...")
    create_eda_plots()
    
    print("\nPreparing data for hierarchical modeling...")
    modeling_data = prepare_modeling_data()
    
    model_results = hierarchical_model_analysis(modeling_data)
    
    print("\nGenerating model diagnostic plots...")
    create_model_plots(model_results)
    
    wellness_performance_analysis()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated visualizations:")
    print("  Figure 1: Training patterns and distributions")
    print("  Figure 2: Wellness metrics time series")
    print("  Figure 3: Wellness correlation matrix")
    print("  Figure 4: Historic competition performance progression (2023-2025)")
    print("  Figure 5: Model diagnostic plots (actual vs predicted, residuals)")
    print("  Figure 6: Feature importance (fixed effects)")
    print("  Figure 6b: Athlete-specific random effects")
    print("  Figure 7: Wellness-performance relationships (if data available)")
    print("\nAll figures saved as high-resolution PNG files (300 DPI)")
    print("\n" + "=" * 80)
