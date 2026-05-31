"""
# Author: Komal Shahid
# Course: DSC640 - Data Presentation and Visualization
# Assignment: Milestone 3 - Interactive Dashboard and Visualizations
# Date: January 2024
#
# Description: This script creates an interactive dashboard and static visualizations
# analyzing childcare costs across different states, income levels, and time periods.
# The analysis includes geographic distribution, cost burden analysis, correlation
# networks, and temporal trends.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from plotly.subplots import make_subplots
import os
from datetime import datetime

class ChildcareCostAnalysis:
    def __init__(self):
        """Initialize with the childcare dataset"""
        self.data = pd.read_excel('../../../data/nationaldatabaseofchildcareprices.xlsx')
        # Create output directory
        os.makedirs('../output', exist_ok=True)
        os.makedirs('../output/temp', exist_ok=True)
        
        # Add custom color schemes
        self.color_schemes = {
            'main': px.colors.qualitative.Set3,
            'sequential': px.colors.sequential.Viridis,
            'diverging': px.colors.diverging.RdYlBu_r
        }
        
        # Preprocess data
        self.preprocess_data()
    
    def preprocess_data(self):
        """Preprocess the data for visualizations"""
        # Create income brackets with more intuitive labels
        self.data['Income_Bracket'] = pd.qcut(
            self.data['MHI_2018'], 
            q=5, 
            labels=['Very Low Income', 'Low Income', 'Middle Income', 'Upper Middle', 'High Income']
        )
        #data cleaning
        # Calculate annual costs and cost ratios
        self.data['Annual_Cost_Infant'] = self.data['MCInfant'] * 12
        self.data['Cost_Income_Ratio'] = (self.data['Annual_Cost_Infant'] / self.data['MHI_2018']) * 100
        
        # Convert StudyYear to datetime
        self.data['Year'] = pd.to_datetime(self.data['StudyYear'].astype(str), format='%Y')
        
        # Clean up any missing values
        self.data = self.data.dropna(subset=['MCInfant', 'MCToddler', 'MCPreschool', 'MHI_2018'])
        
    def create_dashboard(self):
        """Create an interactive dashboard with key visualizations"""
      
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '<b>Geographic Distribution of Childcare Costs</b>',
                '<b>Cost Burden Across States</b>',
                '<b>Cost Trends by Income Level</b>',
                '<b>Income Level Cost Distribution</b>',
                '<b>Variable Correlation Analysis</b>',
                '<b>State Cost Impact Analysis</b>'
            ),
            specs=[
                [{"type": "choropleth"}, {"type": "scattergeo"}],
                [{"type": "scatter"}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # Add main dashboard title
        fig.update_layout(
            title=dict(
                text='<b>Childcare Cost Analysis Dashboard</b>',
                x=0.5,
                y=0.98,
                xanchor='center',
                yanchor='top',
                font=dict(size=24)
            )
        )
        
        # Choropleth Map
        state_costs = self.data.groupby('State_Abbreviation').agg({
            'MCInfant': 'mean',
            'MHI_2018': 'mean'
        }).reset_index()
        
        # Dynamic bubble sizing
        state_metrics = state_costs.copy()
        state_metrics['Annual_Cost'] = state_metrics['MCInfant'] * 12
        state_metrics['CostIncomeRatio'] = (state_metrics['Annual_Cost'] / state_metrics['MHI_2018']) * 100
        
        state_metrics['bubble_size'] = 20 + (
            (state_metrics['CostIncomeRatio'] - state_metrics['CostIncomeRatio'].min()) * 
            (50 - 20) / (state_metrics['CostIncomeRatio'].max() - state_metrics['CostIncomeRatio'].min())
        )
        
        # Add choropleth with left-side legend
        fig.add_trace(
            go.Choropleth(
                locations=state_costs['State_Abbreviation'],
                z=state_costs['MCInfant'],
                locationmode='USA-states',
                colorscale=self.color_schemes['sequential'],
                colorbar=dict(
                    title="Monthly Cost ($)",
                    x=-0.15,
                    len=0.7,
                    thickness=15
                ),
                hovertemplate=(
                    "<b>%{location}</b><br>" +
                    "Average Monthly Cost: $%{z:,.0f}<br>" +
                    "<extra></extra>"
                ),
                name='Monthly Childcare Cost'
            ),
            row=1, col=1
        )

        # Update geo layout for US only
        fig.update_geos(
            scope='usa',
            projection_type='albers usa',
            showlakes=True,
            lakecolor='rgb(255, 255, 255)',
            showcountries=True,
            countrycolor='rgb(200, 200, 200)',
            showsubunits=True,
            subunitcolor='rgb(255, 255, 255)'
        )

        # Add cost burden map with right-side legend
        fig.add_trace(
            go.Scattergeo(
                locations=state_metrics['State_Abbreviation'],
                locationmode='USA-states',
                text=state_metrics['State_Abbreviation'],
                mode='markers+text',
                marker=dict(
                    size=state_metrics['bubble_size'],
                    color=state_metrics['CostIncomeRatio'],
                    colorscale=self.color_schemes['diverging'],
                    showscale=True,
                    colorbar=dict(
                        title="Cost/Income Ratio (%)",
                        x=1.02,  # Adjust position
                        len=0.7,
                        thickness=15
                    )
                ),
                name='Cost Burden by State',
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    "Annual Cost: $%{customdata[0]:,.0f}<br>" +
                    "Median Income: $%{customdata[1]:,.0f}<br>" +
                    "Cost Burden: %{marker.color:.1f}%<br>" +
                    "<extra></extra>"
                ),
                customdata=state_metrics[['Annual_Cost', 'MHI_2018']].values
            ),
            row=1, col=2
        )

        # Update second geo layout for US only
        fig.update_geos(
            scope='usa',
            showlakes=True,
            lakecolor='rgb(255, 255, 255)',
            showcountries=True,
            countrycolor='rgb(200, 200, 200)',
            showsubunits=True,
            subunitcolor='rgb(255, 255, 255)',
            projection_scale=1.2,  # Zoom in slightly
            center=dict(lat=38, lon=-96),  # Center on continental US
            row=1, col=2
        )

        # Add income distribution donut chart
        income_dist = self.data.groupby('Income_Bracket').agg({
            'Annual_Cost_Infant': 'mean',
            'MHI_2018': 'mean'
        }).reset_index()
        
        fig.add_trace(
            go.Pie(
                labels=income_dist['Income_Bracket'],
                values=income_dist['Annual_Cost_Infant'],
                hole=0.6,
                marker=dict(colors=px.colors.qualitative.Set3),
                textinfo='label+percent',
                textposition='outside',
                showlegend=True,
                legendgroup="right",
                legendgrouptitle_text="Income Levels",
                hovertemplate=(
                    "<b>%{label}</b><br>" +
                    "Average Annual Cost: $%{value:,.0f}<br>" +
                    "<extra></extra>"
                )
            ),
            row=2, col=2
        )

        # Correlation Analysis
        variables = {
            'MCInfant': 'Infant Care',
            'MCToddler': 'Toddler Care',
            'MCPreschool': 'Preschool',
            'MHI_2018': 'Median Income',
            'TotalPop': 'Population',
            'H_Under6_BothWork': 'Working Parents'
        }
        
        corr_matrix = self.data[list(variables.keys())].corr()
        
        # Create heatmap
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=[variables[col] for col in corr_matrix.columns],
                y=[variables[col] for col in corr_matrix.columns],
                text=corr_matrix.values.round(2),
                texttemplate='%{text:.2f}',
                textfont=dict(
                    size=12,
                    color='white'
                ),
                colorscale='RdBu_r',
                zmid=0,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Correlation",
                        side='right'
                    ),
                    thickness=15,
                    len=0.7,
                    x=1.02
                )
            ),
            row=3, col=1
        )
        
        # Update axes
        fig.update_xaxes(
            tickangle=45,
            title="",
            showgrid=False,
            row=3, col=1
        )
        
        fig.update_yaxes(
            title="",
            showgrid=False,
            row=3, col=1
        )

        # Rsunburst chart
        state_hierarchy = self.data.groupby(['State_Abbreviation', 'Income_Bracket']).agg({
            'Annual_Cost_Infant': 'mean',
            'MHI_2018': 'mean'
        }).reset_index()
        
        # Calculate cost burden for each group
        state_hierarchy['Cost_Burden'] = (state_hierarchy['Annual_Cost_Infant'] / state_hierarchy['MHI_2018']) * 100
        
        # Create labels and parents for sunburst
        labels = list(state_hierarchy['State_Abbreviation'].unique()) + \
                list(state_hierarchy.apply(lambda x: f"{x['State_Abbreviation']}-{x['Income_Bracket']}", axis=1))
        
        parents = [''] * len(state_hierarchy['State_Abbreviation'].unique()) + \
                 list(state_hierarchy['State_Abbreviation'])
        
        values = list(state_hierarchy.groupby('State_Abbreviation')['Cost_Burden'].mean()) + \
                list(state_hierarchy['Cost_Burden'])
        
        # Create color scale based on cost burden
        colors = px.colors.sequential.Viridis
        
        fig.add_trace(
            go.Bar(
                x=state_hierarchy['State_Abbreviation'],
                y=state_hierarchy['Cost_Burden'],
                name='Cost Burden',
                marker_color='rgb(239, 85, 59)',
                opacity=0.8,
                hovertemplate=(
                    "<b>%{x}</b><br>" +
                    "Cost Burden: %{y:.1f}%<br>" +
                    "<extra></extra>"
                )
            ),
            row=3, col=2
        )
        
        # State Cost Impact Analysis
        state_impact = self.data.groupby('State_Abbreviation').agg({
            'Annual_Cost_Infant': 'mean',
            'MHI_2018': 'mean',
            'H_Under6_BothWork': 'sum',
            'TotalPop': 'mean'
        }).reset_index()
        
        state_impact['Cost_Burden'] = (state_impact['Annual_Cost_Infant'] / state_impact['MHI_2018']) * 100
        state_impact['Working_Parent_Ratio'] = (state_impact['H_Under6_BothWork'] / state_impact['TotalPop']) * 100
        
        # Sort states by cost burden
        state_impact = state_impact.sort_values('Cost_Burden', ascending=True)
        
        fig.add_trace(
            go.Bar(
                x=state_impact['State_Abbreviation'],
                y=state_impact['Cost_Burden'],
                name='Cost Burden',
                marker_color='rgb(239, 85, 59)',
                opacity=0.8,
                hovertemplate=(
                    "<b>%{x}</b><br>" +
                    "Cost Burden: %{y:.1f}%<br>" +
                    "<extra></extra>"
                )
            ),
            row=3, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=state_impact['State_Abbreviation'],
                y=state_impact['Working_Parent_Ratio'],
                name='Working Parent Ratio',
                marker_color='rgb(99, 110, 250)',
                opacity=0.8,
                hovertemplate=(
                    "<b>%{x}</b><br>" +
                    "Working Parent Ratio: %{y:.1f}%<br>" +
                    "<extra></extra>"
                )
            ),
            row=3, col=2
        )
        
        # Update layout for state impact subplot
        fig.update_xaxes(
            tickangle=45,
            title_text="State",
            row=3, col=2
        )
        
        fig.update_yaxes(
            title_text="Percentage (%)",
            row=3, col=2
        )
        
        # Update subplot title
        fig.update_annotations(
            text="<b>State Cost Impact Analysis</b>",
            selector=dict(text="<b>State Cost Impact Analysis</b>")
        )

        # Update layout for better legend positioning and visibility
        fig.update_layout(
            width=2400,
            height=2400,
            template='plotly_white',
            showlegend=True,
            legend=dict(
                x=1.1,  # Adjust legend position
                y=0.5,
                xanchor='left',
                yanchor='middle',
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1,
                font=dict(size=12),
                groupclick="toggleitem",
                tracegroupgap=40
            ),
            margin=dict(t=150, b=100, r=200, l=200)  # Adjust margins
        )

        # Update subplot titles font size
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=16, color='black')

        # Cost Trend Comparison
        yearly_trends = self.data.groupby(['Year', 'Income_Bracket']).agg({
            'Annual_Cost_Infant': 'mean',
            'MHI_2018': 'mean'
        }).reset_index()

        colors = px.colors.qualitative.Set3
        for i, income_level in enumerate(['Very Low Income', 'Low Income', 'Middle Income', 'Upper Middle', 'High Income']):
            mask = yearly_trends['Income_Bracket'] == income_level
            fig.add_trace(
                go.Scatter(
                    x=yearly_trends[mask]['Year'],
                    y=yearly_trends[mask]['Annual_Cost_Infant'],
                    name=income_level,
                    mode='lines+markers',
                    line=dict(width=3, color=colors[i]),
                    marker=dict(size=8, color=colors[i]),
                    legendgroup="income_trends",
                    legendgrouptitle_text="Income Levels",
                    hovertemplate=(
                        "<b>%{x|%Y}</b><br>" +
                        "Income Level: " + income_level + "<br>" +
                        "Annual Cost: $%{y:,.0f}<br>" +
                        "<extra></extra>"
                    )
                ),
                row=2, col=1
            )

        # Save the dashboard
        fig.write_html("../output/dashboard.html")
        return fig

    def generate_visualizations(self):
        """Generate static visualizations for the report"""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Correlation Analysis
        variables = {
            'MCInfant': 'Infant Care',
            'MCToddler': 'Toddler Care',
            'MCPreschool': 'Preschool',
            'MHI_2018': 'Median Income',
            'TotalPop': 'Population',
            'H_Under6_BothWork': 'Working Parents'
        }
        
        # Create correlation matrix
        corr_data = self.data[list(variables.keys())].copy()
        corr_data.columns = [variables[col] for col in corr_data.columns]
        
        # Create correlation matrix plot
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_data.corr(), dtype=bool))
        
        # Create heatmap
        sns.heatmap(corr_data.corr(), 
                   mask=mask,
                   annot=True,
                   fmt='.2f',
                   cmap='RdBu_r',
                   center=0,
                   square=True,
                   annot_kws={'size': 10, 'color': 'white'},
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title('Variable Correlations Analysis', pad=20, fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('../output/correlation.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    analysis = ChildcareCostAnalysis()
    analysis.create_dashboard()
    analysis.generate_visualizations()