import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(results, output_file=None):
    """
    Plot feature importance from the model
    
    Args:
        results (pd.DataFrame): DataFrame with prediction results and features
        output_file (str): Path to save the plot
    """
    # Get feature importance from the model
    importance = results.drop(['text', 'depression_severity', 'predicted_severity'], axis=1).mean()
    importance = importance.sort_values(ascending=False)
    
    # Plot the feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance.values, y=importance.index, palette='viridis')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Save the plot if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show() 