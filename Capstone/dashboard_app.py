"""
Athlete Performance Analytics Dashboard

Author: Michael Robinson
Course: DATA 698 - Capstone Project
Professor: George Hagstrom
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Athlete Performance Analytics Dashboard",
    page_icon="🏃‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    h1 {
        color: #1e3a5f;
        padding-bottom: 1rem;
    }
    h2 {
        color: #16a085;
        padding-top: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Athlete Performance Analytics Dashboard")

# Load data 
@st.cache_data
def load_all_data():
    """Load all datasets"""
    try:
        athletes = pd.read_csv('athletes.csv')
        training = pd.read_csv('training_data.csv')
        wellness = pd.read_csv('wellness_data.csv')
        competition = pd.read_csv('competition_data.csv')
        
        # Convert dates
        training['date'] = pd.to_datetime(training['date'])
        wellness['date'] = pd.to_datetime(wellness['date'])
        competition['date'] = pd.to_datetime(competition['date'])
        
        return athletes, training, wellness, competition
    except FileNotFoundError as e:
        st.error(f"Data files not found: {str(e)}")
        st.info("Please ensure all CSV files are in the same directory as this dashboard.")
        return None, None, None, None

# Load data
athletes, training, wellness, competition = load_all_data()

# Main dashboard
if athletes is not None and training is not None and wellness is not None and competition is not None:
    
    # Sidebar
    st.sidebar.header("Athlete Selection")
    
    # Get athlete names
    athlete_names = ['TEAM OVERVIEW'] + sorted(athletes['Athlete'].tolist())
    selected_athlete = st.sidebar.selectbox("Select Athlete", athlete_names)
    
    # TEAM OVERVIEW
    
    if selected_athlete == 'TEAM OVERVIEW':
        st.header("Team Overview - St. John's Women's Sprint Team")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Athletes", len(athletes))
        with col2:
            st.metric("Training Sessions", len(training))
        with col3:
            st.metric("Wellness Responses", len(wellness))
            st.caption(f"{len(wellness)/(101*13)*100:.1f}% compliance")
        with col4:
            st.metric("Historic Races", len(competition))
        
        st.markdown("---")
        
        # Study period info
        st.subheader("Study Period")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Training Period:** {training['date'].min().strftime('%B %d, %Y')} - {training['date'].max().strftime('%B %d, %Y')} (14 weeks)")
        with col2:
            st.info(f"**Historic Competition Data:** {competition['date'].min().strftime('%B %Y')} - {competition['date'].max().strftime('%B %Y')}")
        
        # Team roster
        st.markdown("---")
        st.subheader("Team Roster")
        
        roster_display = athletes[['Athlete', 'classification', '100m_PB', '200m_PB']].copy()
        roster_display.columns = ['Athlete', 'Year', '100m PR', '200m PR']
        roster_display['100m PR'] = roster_display['100m PR'].apply(lambda x: f"{x:.2f}s" if not pd.isna(x) else "N/A")
        roster_display['200m PR'] = roster_display['200m PR'].apply(lambda x: f"{x:.2f}s" if not pd.isna(x) else "N/A")
        
        st.dataframe(roster_display, use_container_width=True, hide_index=True)
        
        # Training summary
        st.markdown("---")
        st.subheader("Training Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Session type distribution
            session_counts = training['session_type'].value_counts()
            fig = px.pie(
                values=session_counts.values,
                names=session_counts.index,
                title='Training Session Distribution',
                color_discrete_sequence=px.colors.sequential.Teal
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average sessions per athlete
            sessions_per_athlete = training.groupby('athlete_id').size()
            fig = px.bar(
                x=athletes['Athlete'],
                y=sessions_per_athlete.values,
                title='Training Sessions per Athlete',
                labels={'x': 'Athlete', 'y': 'Sessions'},
                color=sessions_per_athlete.values,
                color_continuous_scale='Teal'
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Wellness summary
        st.markdown("---")
        st.subheader("Wellness Monitoring Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Readiness", f"{wellness['readiness'].mean():.2f}/10")
            st.caption("Composite wellness score")
        with col2:
            st.metric("Average Sleep", f"{wellness['sleep_hours'].mean():.1f} hrs")
            st.caption("Daily sleep duration")
        with col3:
            st.metric("Average Fatigue", f"{wellness['fatigue'].mean():.2f}/10")
            st.caption("Self-reported fatigue level")
        
        # Wellness trends over time
        st.markdown("---")
        st.subheader("Wellness Trends Over Training Period")
        
        wellness_daily = wellness.groupby('date')[['readiness', 'fatigue', 'sleep_hours']].mean().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wellness_daily['date'], y=wellness_daily['readiness'],
                                name='Readiness', line=dict(color='#16a085', width=2)))
        fig.add_trace(go.Scatter(x=wellness_daily['date'], y=wellness_daily['fatigue'],
                                name='Fatigue', line=dict(color='#e74c3c', width=2)))
        fig.add_trace(go.Scatter(x=wellness_daily['date'], y=wellness_daily['sleep_hours'],
                                name='Sleep Hours', line=dict(color='#3498db', width=2)))
        
        fig.update_layout(
            title='Daily Team Average Wellness Metrics',
            xaxis_title='Date',
            yaxis_title='Score',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Team Risk Overview
        st.markdown("---")
        st.subheader(" Team Overtraining Risk Overview")
        
        # Calculate risk for each athlete (last 7 days)
        at_risk_athletes = []
        
        for _, athlete in athletes.iterrows():
            athlete_id = athlete['athlete_id']
            athlete_name = athlete['Athlete']
            recent_wellness = wellness[wellness['athlete_id'] == athlete_id].sort_values('date').tail(7)
            
            if len(recent_wellness) >= 3:
                avg_readiness = recent_wellness['readiness'].mean()
                avg_fatigue = recent_wellness['fatigue'].mean()
                avg_soreness = recent_wellness['soreness'].mean()
                
                # Simple risk score
                risk_score = 0
                if avg_readiness < 6.0:
                    risk_score += 30
                if avg_fatigue > 6.0:
                    risk_score += 25
                if avg_soreness > 6.0:
                    risk_score += 20
                
                if risk_score >= 30:
                    at_risk_athletes.append({
                        'Athlete': athlete_name,
                        'Risk Score': risk_score,
                        'Readiness': f"{avg_readiness:.1f}",
                        'Fatigue': f"{avg_fatigue:.1f}",
                        'Soreness': f"{avg_soreness:.1f}"
                    })
        
        if len(at_risk_athletes) > 0:
            st.warning(f"**{len(at_risk_athletes)} athlete(s) showing elevated risk indicators**")
            risk_df = pd.DataFrame(at_risk_athletes).sort_values('Risk Score', ascending=False)
            st.dataframe(risk_df, use_container_width=True, hide_index=True)
        else:
            st.success("**No athletes currently at elevated risk** - Team wellness indicators within normal ranges")
        
        # Competition summary
        st.markdown("---")
        st.subheader("Historic Competition Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            comp_100m = competition[competition['event'] == '100m']
            if len(comp_100m) > 0:
                st.metric("100m Average", f"{comp_100m['time'].mean():.2f}s")
                st.caption(f"Best: {comp_100m['time'].min():.2f}s | {len(comp_100m)} races")
        
        with col2:
            comp_200m = competition[competition['event'] == '200m']
            if len(comp_200m) > 0:
                st.metric("200m Average", f"{comp_200m['time'].mean():.2f}s")
                st.caption(f"Best: {comp_200m['time'].min():.2f}s | {len(comp_200m)} races")
    
    # INDIVIDUAL ATHLETE VIEW
    
    else:
        # Get athlete ID
        athlete_id = athletes[athletes['Athlete'] == selected_athlete]['athlete_id'].values[0]
        athlete_info = athletes[athletes['athlete_id'] == athlete_id].iloc[0]
        
        # Display athlete info in sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("Athlete Profile")
        st.sidebar.write(f"**Classification:** {athlete_info['classification']}")
        st.sidebar.write(f"**100m PR:** {athlete_info['100m_PB']:.2f}s" if not pd.isna(athlete_info['100m_PB']) else "**100m PR:** N/A")
        st.sidebar.write(f"**200m PR:** {athlete_info['200m_PB']:.2f}s" if not pd.isna(athlete_info['200m_PB']) else "**200m PR:** N/A")
        
        # Filter data for selected athlete
        athlete_training = training[training['athlete_id'] == athlete_id]
        athlete_wellness = wellness[wellness['athlete_id'] == athlete_id]
        athlete_competition = competition[competition['athlete_id'] == athlete_id]
        
        # Main content
        st.header(f"Performance Analysis: {selected_athlete}")
        
        # Training period info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Sessions", len(athlete_training))
        with col2:
            st.metric("Wellness Responses", len(athlete_wellness))
            st.caption(f"{len(athlete_wellness)/101*100:.1f}% compliance")
        with col3:
            st.metric("Historic Races", len(athlete_competition))
        
        st.info(f"**Training Period:** {athlete_training['date'].min().strftime('%B %d, %Y')} to {athlete_training['date'].max().strftime('%B %d, %Y')}")
        
        # Training load over time
        st.markdown("---")
        st.subheader("Training Load Progression")
        
        weekly_load = athlete_training.groupby('week_number')['training_load'].mean().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weekly_load['week_number'],
            y=weekly_load['training_load'],
            mode='lines+markers',
            name='Training Load',
            line=dict(color='#16a085', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f'Weekly Average Training Load - {selected_athlete}',
            xaxis_title='Week Number',
            yaxis_title='Training Load (1-100)',
            height=400,
            hovermode='x'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Wellness metrics
        st.markdown("---")
        st.subheader("Wellness Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Readiness", f"{athlete_wellness['readiness'].mean():.2f}/10")
        with col2:
            st.metric("Sleep", f"{athlete_wellness['sleep_hours'].mean():.1f} hrs")
        with col3:
            st.metric("Fatigue", f"{athlete_wellness['fatigue'].mean():.2f}/10")
        with col4:
            st.metric("Soreness", f"{athlete_wellness['soreness'].mean():.2f}/10")
        
        # Wellness trends
        wellness_metrics = athlete_wellness[['date', 'readiness', 'fatigue', 'soreness']].copy()
        wellness_metrics = wellness_metrics.sort_values('date')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wellness_metrics['date'], y=wellness_metrics['readiness'],
                                name='Readiness', line=dict(color='#16a085', width=2)))
        fig.add_trace(go.Scatter(x=wellness_metrics['date'], y=wellness_metrics['fatigue'],
                                name='Fatigue', line=dict(color='#e74c3c', width=2)))
        fig.add_trace(go.Scatter(x=wellness_metrics['date'], y=wellness_metrics['soreness'],
                                name='Soreness', line=dict(color='#f39c12', width=2)))
        
        fig.update_layout(
            title='Wellness Trends Over Time',
            xaxis_title='Date',
            yaxis_title='Score (1-10)',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk Detection & Alerts
        st.markdown("---")
        st.subheader(" Overtraining Risk Detection")
        
        # Calculate recent wellness (last 7 days)
        recent_wellness = athlete_wellness.sort_values('date').tail(7)
        
        if len(recent_wellness) > 0:
            avg_readiness = recent_wellness['readiness'].mean()
            avg_fatigue = recent_wellness['fatigue'].mean()
            avg_soreness = recent_wellness['soreness'].mean()
            avg_sleep = recent_wellness['sleep_hours'].mean()
            
            risk_score = 0
            risk_factors = []
            
            if avg_readiness < 6.0:
                risk_score += 30
                risk_factors.append(f"Low readiness ({avg_readiness:.1f}/10)")
            elif avg_readiness < 6.5:
                risk_score += 15
                risk_factors.append(f"Below-average readiness ({avg_readiness:.1f}/10)")
            
            if avg_fatigue > 6.0:
                risk_score += 25
                risk_factors.append(f"High fatigue ({avg_fatigue:.1f}/10)")
            elif avg_fatigue > 5.5:
                risk_score += 10
                risk_factors.append(f"Elevated fatigue ({avg_fatigue:.1f}/10)")
            
            if avg_soreness > 6.0:
                risk_score += 20
                risk_factors.append(f"High soreness ({avg_soreness:.1f}/10)")
            elif avg_soreness > 5.5:
                risk_score += 10
                risk_factors.append(f"Elevated soreness ({avg_soreness:.1f}/10)")
            
            if avg_sleep < 6.5:
                risk_score += 20
                risk_factors.append(f"Insufficient sleep ({avg_sleep:.1f} hrs)")
            elif avg_sleep < 7.0:
                risk_score += 10
                risk_factors.append(f"Below-optimal sleep ({avg_sleep:.1f} hrs)")
            
            recent_injuries = recent_wellness[recent_wellness['injury_status'] != 'None']
            if len(recent_injuries) > 0:
                risk_score += 15
                risk_factors.append(f"Recent injury reported")
            
            if len(recent_wellness) >= 5:
                readiness_trend = recent_wellness['readiness'].iloc[-3:].mean() - recent_wellness['readiness'].iloc[:3].mean()
                if readiness_trend < -0.5:
                    risk_score += 15
                    risk_factors.append(f"Declining readiness trend")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if risk_score >= 60:
                    st.error(f"### HIGH RISK")
                    st.metric("Risk Score", f"{risk_score}/100", delta=None)
                    risk_color = "#e74c3c"
                elif risk_score >= 30:
                    st.warning(f"### MODERATE RISK")
                    st.metric("Risk Score", f"{risk_score}/100", delta=None)
                    risk_color = "#f39c12"
                else:
                    st.success(f"### LOW RISK")
                    st.metric("Risk Score", f"{risk_score}/100", delta=None)
                    risk_color = "#16a085"
            
            with col2:
                if risk_score >= 60:
                    st.error("**IMMEDIATE ATTENTION REQUIRED**")
                    st.markdown("**Risk Factors Identified:**")
                    for factor in risk_factors:
                        st.markdown(f"• {factor}")
                    st.markdown("**Recommendations:**")
                    st.markdown("• Reduce training intensity immediately")
                    st.markdown("• Schedule recovery day(s)")
                    st.markdown("• Consult with coaching staff")
                    st.markdown("• Consider sports medicine evaluation")
                
                elif risk_score >= 30:
                    st.warning("** MONITOR CLOSELY**")
                    st.markdown("**Risk Factors Identified:**")
                    for factor in risk_factors:
                        st.markdown(f"• {factor}")
                    st.markdown("**Recommendations:**")
                    st.markdown("• Monitor wellness daily")
                    st.markdown("• Adjust training load if symptoms worsen")
                    st.markdown("• Prioritize sleep and recovery")
                    st.markdown("• Consider lighter training session")
                
                else:
                    st.success("**GOOD RECOVERY STATUS**")
                    st.markdown("**Current Status:**")
                    st.markdown(f"• Readiness: {avg_readiness:.1f}/10")
                    st.markdown(f"• Fatigue: {avg_fatigue:.1f}/10")
                    st.markdown(f"• Soreness: {avg_soreness:.1f}/10")
                    st.markdown(f"• Sleep: {avg_sleep:.1f} hours")
                    st.markdown("**Recommendations:**")
                    st.markdown("• Continue current training plan")
                    st.markdown("• Maintain consistent wellness monitoring")
            
            # Risk score visualization
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={'text': "Overtraining Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 30], 'color': "#d5f4e6"},
                        {'range': [30, 60], 'color': "#fff3cd"},
                        {'range': [60, 100], 'color': "#f8d7da"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 60
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No recent wellness data available for risk assessment")
        
        # Performance times
        st.markdown("---")
        st.subheader("Training Performance Times")
        
        times_100m = athlete_training[athlete_training['100m_time'].notna()][['date', '100m_time']].sort_values('date')
        times_200m = athlete_training[athlete_training['200m_time'].notna()][['date', '200m_time']].sort_values('date')
        
        if len(times_100m) > 0 or len(times_200m) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                if len(times_100m) > 0:
                    fig = px.scatter(times_100m, x='date', y='100m_time',
                                   title='100m Training Times',
                                   labels={'100m_time': 'Time (seconds)', 'date': 'Date'})
                    fig.update_traces(marker=dict(size=8, color='#3498db'))
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if len(times_200m) > 0:
                    fig = px.scatter(times_200m, x='date', y='200m_time',
                                   title='200m Training Times',
                                   labels={'200m_time': 'Time (seconds)', 'date': 'Date'},
                                   trendline='lowess', trendline_color_override='red')
                    fig.update_traces(marker=dict(size=8, color='#e74c3c'))
                    st.plotly_chart(fig, use_container_width=True)
        
        # Performance Prediction
        st.markdown("---")
        st.subheader("Performance Prediction")
        
        # Select event for prediction
        available_events = []
        if not pd.isna(athlete_info['100m_PB']):
            available_events.append('100m')
        if not pd.isna(athlete_info['200m_PB']):
            available_events.append('200m')
        
        if len(available_events) > 0:
            selected_event = st.selectbox("Select Event for Prediction", available_events)
            
            if st.button(f"Generate {selected_event} Prediction", type="primary"):
                with st.spinner("Training ML model and generating prediction..."):
                    
                    # Prediction
                    def predict_performance(athlete_id, event):
                        event_col = f'{event}_time'
                        pb_col = f'{event}_PB'
                        
                        # Merge training + wellness
                        merged = pd.merge(
                            athlete_training,
                            athlete_wellness,
                            on=['athlete_id', 'date'],
                            how='inner',
                            suffixes=('_train', '_well')
                        )
                        
                        model_data = merged[merged[event_col].notna()].copy()
                        
                        if len(model_data) < 20:
                            return None, None, None
                        
                        model_data = model_data.sort_values('date')
                        model_data['rolling_readiness'] = model_data['readiness'].rolling(3, min_periods=1).mean()
                        model_data['rolling_fatigue'] = model_data['fatigue'].rolling(3, min_periods=1).mean()
                        model_data['cumulative_load'] = model_data['training_load'].cumsum()
                        
                        feature_cols = [
                            'training_load', 'readiness', 'fatigue', 'sleep_hours',
                            'soreness', 'stress', 'resting_hr', 'rolling_readiness',
                            'rolling_fatigue', 'cumulative_load'
                        ]
                        
                        X = model_data[feature_cols].fillna(method='ffill').fillna(method='bfill')
                        y = model_data[event_col]
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Train Ridge Regression
                        model = Ridge(alpha=1.0)
                        model.fit(X_train_scaled, y_train)
                        
                        train_r2 = model.score(X_train_scaled, y_train)
                        test_r2 = model.score(X_test_scaled, y_test)
                        y_pred = model.predict(X_test_scaled)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        
                        latest = model_data.iloc[[-1]][feature_cols].fillna(method='ffill').fillna(method='bfill')
                        latest_scaled = scaler.transform(latest)
                        prediction = model.predict(latest_scaled)[0]
                        
                        importance = pd.DataFrame({
                            'Feature': feature_cols,
                            'Importance': np.abs(model.coef_)
                        }).sort_values('Importance', ascending=False)
                        
                        metrics = {
                            'train_r2': train_r2,
                            'test_r2': test_r2,
                            'rmse': rmse,
                            'n_samples': len(model_data)
                        }
                        
                        return prediction, metrics, importance
                    
                    prediction, metrics, importance = predict_performance(athlete_id, selected_event)
                    
                    if prediction is not None:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Predicted Time", f"{prediction:.2f}s")
                            diff_from_pb = prediction - athlete_info[f'{selected_event}_PB']
                            st.caption(f"{diff_from_pb:+.2f}s from PR")
                        
                        with col2:
                            st.metric("Model R²", f"{metrics['test_r2']:.3f}")
                            st.caption(f"RMSE: {metrics['rmse']:.3f}s")
                        
                        with col3:
                            st.metric("Training Samples", metrics['n_samples'])
                            st.caption(f"Train R²: {metrics['train_r2']:.3f}")
                        
                        st.markdown("**Feature Importance (Top 5)**")
                        top_features = importance.head(5)
                        
                        fig = px.bar(
                            top_features,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            color='Importance',
                            color_continuous_scale='Teal'
                        )
                        fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        diff_from_pb = prediction - athlete_info[f'{selected_event}_PB']
                        
                        if diff_from_pb < 0:
                            st.success(f"✅ **Excellent form!** Model predicts {abs(diff_from_pb):.2f}s FASTER than your PR!")
                        elif diff_from_pb < 0.2:
                            st.success(f"✅ **Great shape!** Within {diff_from_pb:.2f}s of PR")
                        elif diff_from_pb < 0.5:
                            st.warning(f"⚠️ **Good, but room for improvement.** {diff_from_pb:.2f}s slower than PR")
                        else:
                            st.error(f"⚠️ **Performance concern.** {diff_from_pb:.2f}s slower than PR. Consider reviewing training load and recovery.")
                    
                    else:
                        st.warning("Unable to generate prediction - insufficient training data (minimum 20 sessions required)")
        
        else:
            st.info(f"{selected_athlete} does not have PR data for sprint events")
        
        # Competition History
        st.markdown("---")
        st.subheader("Historical Competition Results")
        st.info(f"Competition data from {competition['date'].min().strftime('%B %Y')} to {competition['date'].max().strftime('%B %Y')}")
        
        if len(athlete_competition) > 0:
            comp_display = athlete_competition[['date', 'competition', 'event', 'time', 'wind']].copy()
            comp_display['date'] = comp_display['date'].dt.strftime('%Y-%m-%d')
            comp_display = comp_display.sort_values('date', ascending=False)
            st.dataframe(comp_display, use_container_width=True, hide_index=True)
        else:
            st.warning("No competition data available for this athlete")

else:
    st.error("**Data files not found.** Please ensure all CSV files are in the same directory as this dashboard.")

