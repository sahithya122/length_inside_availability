import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the dataset from a CSV file
df = pd.read_csv('a.csv')  # Replace 'a.csv' with your actual file path

# Calculate the length of ENTITY_DESCRIPTION
df['ENTITY_LENGTH'] = df['ENTITY_DESCRIPTION'].str.len()

# Function to extract entity names (customize the extraction logic as needed)
def extract_entity_name(description):
    # Example extraction logic (adjust regex as necessary)
    match = re.search(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', description)  # Matches capitalized names
    return match.group(0) if match else 'Unknown'

# Apply the function to create a new column for entity names
df['ENTITY_NAME'] = df['ENTITY_DESCRIPTION'].apply(extract_entity_name)

# Group by CATEGORY_ID and calculate summary statistics
summary_stats = df.groupby('CATEGORY_ID')['ENTITY_LENGTH'].agg(
    count='count',
    mean='mean',
    std='std',
    min='min',
    max='max',
    q25=lambda x: np.percentile(x, 25),
    q50=lambda x: np.percentile(x, 50),
    q75=lambda x: np.percentile(x, 75),
    entity_names=lambda x: ', '.join(x.unique())  # Aggregate unique entity names
).reset_index()

# Filter CATEGORY_IDs from 0 to 1000
filtered_means = summary_stats[(summary_stats['CATEGORY_ID'] >= 0) & (summary_stats['CATEGORY_ID'] <= 1000)]

# Save the summary statistics to a new Excel file
filtered_means.to_excel('category_summary_stat.xlsx', index=False)

# Create a folder to save the graphs if it doesn't exist
output_folder = 'category_plots'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Folder '{output_folder}' created.")
else:
    print(f"Folder '{output_folder}' already exists.")

# Plot heatmap and boxplot for each CATEGORY_ID
for category_id in df['CATEGORY_ID'].unique():
    # Filter the data for the current category
    category_data = df[df['CATEGORY_ID'] == category_id]

    if category_data.empty:  # Skip if no data for the category
        print(f"No data available for CATEGORY_ID {category_id}")
        continue

    # Plot boxplot
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=category_data['CATEGORY_ID'], y=category_data['ENTITY_LENGTH'])
    plt.title(f'Boxplot for CATEGORY_ID {category_id}')
    boxplot_path = os.path.join(output_folder, f'boxplot_category_{category_id}.png')
    print(f"Saving boxplot to {boxplot_path}")
    plt.savefig(boxplot_path)
    plt.close()

    # Plot heatmap
    plt.figure(figsize=(8, 4))
    heatmap_data = category_data.pivot_table(index=category_data.index, values='ENTITY_LENGTH')
    sns.heatmap(heatmap_data, cmap='coolwarm', annot=False, cbar=True)
    plt.title(f'Heatmap for CATEGORY_ID {category_id}')
    heatmap_path = os.path.join(output_folder, f'heatmap_category_{category_id}.png')
    print(f"Saving heatmap to {heatmap_path}")
    plt.savefig(heatmap_path)
    plt.close()

print("Boxplots and heatmaps saved for each CATEGORY_ID.")
