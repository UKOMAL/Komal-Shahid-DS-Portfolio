import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_attention_weights(text, attention_weights, output_file=None):
    """
    Plot attention weights from the transformer model
    
    Args:
        text (str): Input text
        attention_weights (list): List of attention weights for each token
        output_file (str): Path to save the plot
    """
    # Tokenize the text
    tokens = text.split()
    
    # Trim or pad attention weights to match token length
    attention_weights = attention_weights[:len(tokens)]
    attention_weights += [0] * (len(tokens) - len(attention_weights))
    
    # Create a DataFrame for plotting
    data = {'Token': tokens, 'Attention': attention_weights}
    df = pd.DataFrame(data)
    
    # Plot the attention weights
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Token', y='Attention', data=df)
    plt.title('Attention Weights')
    plt.xlabel('Token')
    plt.ylabel('Attention Weight')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show() 