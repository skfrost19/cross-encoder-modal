import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
import glob
from pathlib import Path

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

def create_output_folder():
    """Create gpu_usage folder if it doesn't exist"""
    output_folder = "gpu_usage"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder

def extract_gpu_columns(df, optimizer):
    """Extract GPU memory columns for a specific optimizer"""
    gpu_cols = []
    for col in df.columns:
        # Case-insensitive matching for optimizer names
        col_lower = col.lower()
        optimizer_lower = optimizer.lower()
        if optimizer_lower in col_lower and 'memoryallocated' in col_lower and '__min' not in col_lower and '__max' not in col_lower:
            gpu_cols.append(col)
    return gpu_cols

def process_csv_file(csv_file, output_folder):
    """Process a single CSV file and generate plot + stats"""
    print(f"Processing {csv_file}...")
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Get the base filename without extension
    base_name = Path(csv_file).stem
    
    # Clean the data - replace empty strings with NaN and convert to float
    for col in df.columns:
        if col != 'Relative Time (Process)':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where time is NaN
    df = df.dropna(subset=['Relative Time (Process)'])
    
    # Identify optimizers in the data
    optimizers = []
    for col in df.columns:
        col_lower = col.lower()
        if 'adamw' in col_lower:
            if 'AdamW' not in optimizers:
                optimizers.append('AdamW')
        elif 'lion' in col_lower:
            if 'Lion' not in optimizers:
                optimizers.append('Lion')
    
    if not optimizers:
        print(f"No optimizer data found in {csv_file}")
        return
    
    # Initialize statistics dictionary
    stats = {
        'file_name': base_name,
        'optimizers': {}
    }
    
    # Create the plot with seaborn styling
    plt.figure(figsize=(12, 8))
    
    # Use seaborn color palette
    colors = sns.color_palette("husl", len(optimizers))
    
    for i, optimizer in enumerate(optimizers):
        # Get GPU columns for this optimizer
        gpu_cols = extract_gpu_columns(df, optimizer)
        
        if not gpu_cols:
            print(f"No GPU columns found for {optimizer} in {csv_file}")
            continue
        
        print(f"Found {len(gpu_cols)} GPU columns for {optimizer}")
        
        # Calculate average across all GPUs for this optimizer
        gpu_data = df[gpu_cols].copy()
          # Forward fill NaN values for each GPU column
        for col in gpu_cols:
            gpu_data[col] = gpu_data[col].ffill()
        
        # Calculate average across GPUs
        avg_usage = gpu_data.mean(axis=1)
        
        # Remove any remaining NaN values
        valid_indices = ~np.isnan(avg_usage)
        time_values = df['Relative Time (Process)'][valid_indices]
        avg_usage_clean = avg_usage[valid_indices]
        
        if len(avg_usage_clean) == 0:
            print(f"No valid data for {optimizer} in {csv_file}")
            continue
        
        # Plot the average line with seaborn styling
        plt.plot(time_values, avg_usage_clean, 
                label=f'{optimizer} (Average)', 
                linewidth=2.5, 
                color=colors[i],
                alpha=0.8)
        
        # Calculate statistics
        optimizer_stats = {
            'mean_usage': float(np.mean(avg_usage_clean)),
            'std_usage': float(np.std(avg_usage_clean)),
            'peak_usage': float(np.max(avg_usage_clean)),
            'min_usage': float(np.min(avg_usage_clean)),
            'num_gpus': len(gpu_cols),
            'data_points': len(avg_usage_clean)
        }
        
        stats['optimizers'][optimizer] = optimizer_stats
        
        print(f"{optimizer} Stats:")
        print(f"  Mean GPU Usage: {optimizer_stats['mean_usage']:.2f}")
        print(f"  Std GPU Usage: {optimizer_stats['std_usage']:.2f}")
        print(f"  Peak GPU Usage: {optimizer_stats['peak_usage']:.2f}")
        print(f"  Min GPU Usage: {optimizer_stats['min_usage']:.2f}")
        print(f"  Number of GPUs: {optimizer_stats['num_gpus']}")
        print()
    
    # Add comparison statistics if we have multiple optimizers
    if len(stats['optimizers']) > 1:
        opt_names = list(stats['optimizers'].keys())
        if len(opt_names) == 2:
            opt1, opt2 = opt_names
            stats['comparison'] = {
                'mean_difference': stats['optimizers'][opt1]['mean_usage'] - stats['optimizers'][opt2]['mean_usage'],
                'peak_difference': stats['optimizers'][opt1]['peak_usage'] - stats['optimizers'][opt2]['peak_usage'],
                'more_efficient': opt1 if stats['optimizers'][opt1]['mean_usage'] < stats['optimizers'][opt2]['mean_usage'] else opt2
            }
    
    # Customize the plot with seaborn styling
    plt.xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    plt.ylabel('GPU Usage', fontsize=13, fontweight='bold')
    plt.title(f'GPU Memory Usage Comparison - {base_name}', fontsize=15, fontweight='bold', pad=20)
    
    # Enhance legend with seaborn styling
    legend = plt.legend(fontsize=12, frameon=True, shadow=True, fancybox=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Set tick parameters
    plt.tick_params(axis='both', which='major', labelsize=11)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot with DPI=300
    plot_path = os.path.join(output_folder, f'{base_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # Save the statistics as JSON
    json_path = os.path.join(output_folder, f'{base_name}.json')
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Plot saved to: {plot_path}")
    print(f"Stats saved to: {json_path}")
    print("-" * 50)

def main():
    """Main function to process all CSV files in the data folder"""
    # Set matplotlib style to work well with seaborn
    plt.style.use('default')
    
    # Create output folder
    output_folder = create_output_folder()
    
    # Find all CSV files in the data folder
    csv_files = glob.glob("data/*.csv")
    
    if not csv_files:
        print("No CSV files found in the data folder")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    print("=" * 50)
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            process_csv_file(csv_file, output_folder)
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            print("-" * 50)
    
    print("All files processed successfully!")
    print(f"Outputs saved in the '{output_folder}' folder")

if __name__ == "__main__":
    main()