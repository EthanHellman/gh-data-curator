def _add_quality_scatter_plot(self, pdf):
    """Add an improved quality scatter plot colored by repository."""
    # Filter for PRs that passed at least one filter for better visualization
    filtered_df = self.pr_data[self.pr_data['passed_bot_filter'] == True]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Set up colors for repositories - using a more distinct color palette
    distinct_colors = sns.color_palette("tab10", len(self.repository_colors))
    for i, repo_key in enumerate(sorted(self.repository_colors.keys())):
        self.repository_colors[repo_key] = distinct_colors[i % len(distinct_colors)]
    
    colors = []
    for repo in filtered_df['repo_key']:
        colors.append(self.repository_colors.get(repo, 'gray'))
    
    # Create scatter plot with size as the marker size
    scatter = ax.scatter(
        filtered_df['size_score'], 
        filtered_df['relevance_score'],
        c=colors,
        s=filtered_df['code_file_count'] * 10 + 20,  # Adjust size based on code files
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5
    )
    
    # Only label passed PRs that are exemplary (quality > 0.95) to reduce clutter
    passed_prs = filtered_df[filtered_df['passed_filter'] == True]
    exemplary_prs = passed_prs[passed_prs['quality_score'] > 0.95]
    
    # Don't label any PRs as requested
    # Create a custom legend for repositories
    legend_elements = []
    for repo_key, color in self.repository_colors.items():
        if repo_key in filtered_df['repo_key'].values:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor=color, markersize=10,
                                          label=repo_key.replace('_', '/')))
    
    # Add a legend for marker size 
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                                   markeredgecolor='white', markersize=15, alpha=0.5,
                                   label='Larger PR (more files)'))
    
    # Add the legend to the plot
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # Add reference lines
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Content Filter Threshold')
    ax.axvline(x=0.5, color='blue', linestyle='--', alpha=0.3, label='Size Filter Threshold')
    
    # Highlight different quality zones
    high_quality = plt.Rectangle((0.6, 0.6), 0.4, 0.4, color='green', alpha=0.1)
    low_quality = plt.Rectangle((0, 0), 0.4, 0.4, color='red', alpha=0.1)
    ax.add_patch(high_quality)
    ax.add_patch(low_quality)
    ax.text(0.8, 0.8, 'High Quality Zone', fontsize=9, ha='center', va='center', alpha=0.7)
    ax.text(0.2, 0.2, 'Low Quality Zone', fontsize=9, ha='center', va='center', alpha=0.7)
    
    # Enhance chart appearance
    ax.set_xlabel('Size Score (higher is better)', fontsize=12, weight='bold')
    ax.set_ylabel('Content Relevance Score', fontsize=12, weight='bold')
    ax.set_title('PR Quality Distribution by Repository', fontsize=16, weight='bold', pad=20)
    
    # Set axis limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Add grid
    ax.grid(linestyle='--', alpha=0.3)
    
    # Add annotations explaining the plot
    annotation_text = (
        "This plot shows the relationship between size score and content relevance score for PRs.\n"
        "Each point represents a PR, with color indicating repository and size indicating number of files.\n"
        "High-quality PRs appear in the upper-right quadrant."
    )
    fig.text(0.5, 0.01, annotation_text, ha='center', fontsize=9, style='italic',
            color='#555555', bbox=dict(facecolor='white', alpha=0.7, pad=5))
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])  # Adjust layout to make room for the annotation
    
    # Save to figures directory for reference
    plt.savefig(self.figures_dir / "quality_scatter_plot.png", dpi=300)
    
    # Add to PDF
    plt.suptitle('', y=0.98)  # Add space at top for PDF formatting
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def _add_dimension_correlation_heatmap(self, pdf):
    """Add a correlation heatmap of quality dimensions with improved clarity."""
    # Select numerical features for correlation analysis
    features = [
        'bot_confidence', 'file_count', 'code_file_count', 'total_changes',
        'additions', 'deletions', 'size_score', 'complexity_score',
        'relevance_score', 'problem_solving_score', 'code_quality_score',
        'quality_score'
    ]
    
    # Filter data to PRs that have been processed by all filters
    corr_data = self.pr_data[self.pr_data['passed_bot_filter'] == True][features]
    
    # Calculate correlation matrix
    corr_matrix = corr_data.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a more readable correlation heatmap with an improved color scheme
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True  # Only mask the upper triangle, keep diagonal
    
    # Generate a custom diverging colormap for better readability
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap with improved formatting
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
              square=True, linewidths=.5, cbar_kws={"shrink": .7}, annot=True,
              fmt=".2f", annot_kws={"size": 9}, ax=ax)
    
    # Improve feature labels for readability
    feature_labels = [
        'Bot Confidence', 'File Count', 'Code Files', 'Total Changes',
        'Additions', 'Deletions', 'Size Score', 'Complexity Score',
        'Relevance Score', 'Problem Solving', 'Code Quality',
        'Overall Quality'
    ]
    
    ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(feature_labels, rotation=0, fontsize=10)
    
    # Enhance chart appearance
    ax.set_title('Correlation Between Quality Dimensions', fontsize=16, weight='bold', pad=20)
    
    # Find strongest correlations for annotation
    correlation_values = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            correlation_values.append((
                feature_labels[i], 
                feature_labels[j], 
                corr_matrix.iloc[i, j]
            ))
    
    # Sort by absolute correlation value
    correlation_values.sort(key=lambda x: abs(x[2]), reverse=True)
    top_correlations = correlation_values[:5]
    
    # Create annotation text
    correlation_text = "Notable Correlations:\n"
    for feat1, feat2, corr in top_correlations:
        correlation_text += f"• {feat1} & {feat2}: {corr:.2f}\n"
    
    # Add the annotation to the figure
    fig.text(0.15, 0.02, correlation_text, ha='left', fontsize=9,
            color='#333333', bbox=dict(facecolor='white', alpha=0.7, pad=5))
    
    # Add interpretation note
    interpretation = (
        "Correlation values range from -1 (strong negative) to +1 (strong positive).\n"
        "This heatmap reveals how different quality metrics relate to each other across all repositories."
    )
    fig.text(0.65, 0.02, interpretation, ha='left', fontsize=9, style='italic',
            color='#555555', bbox=dict(facecolor='white', alpha=0.7, pad=5))
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])  # Adjust layout to make room for annotations
    
    # Save to figures directory for reference
    plt.savefig(self.figures_dir / "dimension_correlation_heatmap.png", dpi=300)
    
    # Add to PDF
    plt.suptitle('', y=0.98)  # Add space at top for PDF formatting
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def _add_cross_repo_metrics_heatmap(self, pdf):
    """Add a heatmap of cross-repository metrics with improved color scheme."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Extract data for the heatmap
    repo_keys = sorted(self.repo_data.keys())
    repo_names = [key.replace("_", "/") for key in repo_keys]
    
    # Metrics to include in the heatmap
    metrics = [
        'Total PRs', 'Passed PRs', 'Bot Filtered', 'Size Filtered', 
        'Content Filtered', 'Pass Rate (%)', 'Data Reduction (%)', 
        'Avg Quality Score'
    ]
    
    # Create data matrix
    data = np.zeros((len(repo_keys), len(metrics)))
    
    for i, repo_key in enumerate(repo_keys):
        repo_data = self.repo_data[repo_key]
        
        total_prs = len(repo_data["filter_metadata"])
        passed_prs = len(repo_data["filtered_prs"])
        
        # Count PRs filtered at each stage
        bot_filtered = 0
        size_filtered = 0
        content_filtered = 0
        
        for meta in repo_data["filter_metadata"]:
            if not meta.get("bot_filter", {}).get("passed", False):
                bot_filtered += 1
            elif not meta.get("size_filter", {}).get("passed", False):
                size_filtered += 1
            elif not meta.get("content_filter", {}).get("passed", False):
                content_filtered += 1
        
        # Calculate pass rate and data reduction
        pass_rate = passed_prs / total_prs * 100 if total_prs > 0 else 0
        data_reduction = (1 - passed_prs / total_prs) * 100 if total_prs > 0 else 0
        
        # Calculate average quality score
        quality_scores = [
            meta.get("quality_score", 0) 
            for meta in repo_data["filter_metadata"] 
            if meta.get("passed_filter", False)
        ]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Fill the data matrix
        data[i, 0] = total_prs
        data[i, 1] = passed_prs
        data[i, 2] = bot_filtered
        data[i, 3] = size_filtered
        data[i, 4] = content_filtered
        data[i, 5] = pass_rate
        data[i, 6] = data_reduction
        data[i, 7] = avg_quality
    
    # Create a pandas DataFrame for the heatmap
    df = pd.DataFrame(data, index=repo_names, columns=metrics)
    
    # Create the heatmap with enhanced styling
    # Use a green-to-red colormap for better visual indication of good-to-bad values
    # For metrics where higher is better (pass rate, quality score)
    positive_cols = ['Passed PRs', 'Pass Rate (%)', 'Avg Quality Score']
    
    # For metrics where lower is better (filtered counts, data reduction)
    negative_cols = ['Bot Filtered', 'Size Filtered', 'Content Filtered', 'Data Reduction (%)']
    
    # Create normalized data for coloring
    norm_data = df.copy()
    for col in df.columns:
        col_max = df[col].max()
        if col_max > 0:
            if col in positive_cols:
                # Higher values get green (good)
                norm_data[col] = df[col] / col_max
            elif col in negative_cols:
                # Higher values get red (bad)
                norm_data[col] = 1 - (df[col] / col_max)
            else:
                # Neutral normalization
                norm_data[col] = df[col] / col_max
    
    # Plot the heatmap with custom colors
    # Use a colormap from green (good) to red (bad)
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    
    # Plot the heatmap
    sns.heatmap(norm_data, annot=df, fmt=".1f", cmap=cmap, 
               linewidths=0.5, cbar=False, ax=ax)
    
    # Enhance the heatmap appearance
    ax.set_title("Cross-Repository Metrics Comparison", fontsize=16, weight='bold', pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Add a note about normalization
    fig.text(0.5, 0.02, "Note: Colors represent normalized values within each column for better visibility.", 
            ha='center', fontsize=10, style='italic', color='#555555')
    
    plt.tight_layout()
    
    # Save to figures directory for reference
    plt.savefig(self.figures_dir / "enhanced_cross_repo_metrics_heatmap.png", dpi=300)
    
    # Add to PDF
    plt.suptitle('', y=0.98)  # Add space at top for PDF formatting
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def _add_pr_clustering_visualization(self, pdf):
    """Add a clustering visualization of PRs using code-based PCA."""
    # Get PRs that have been processed
    cluster_data = self.pr_data[self.pr_data['passed_bot_filter'] == True].copy()
    
    # Check if we have enough data
    if len(cluster_data) < 10:
        logger.warning("Not enough data for clustering visualization")
        return
    
    # For code embedding, we'll primarily focus on code-based features
    code_features = [
        'code_file_count', 'total_changes', 
        'size_score', 'complexity_score', 'relevance_score',
        'problem_solving_score', 'code_quality_score'
    ]
    
    # Prepare data for PCA
    X = cluster_data[code_features]
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Create a DataFrame with PCA results and cluster labels
    pca_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': cluster_labels,
        'Repository': cluster_data['repository'],
        'PR': cluster_data['pr_number'],
        'Passed': cluster_data['passed_filter'],
        'Quality': cluster_data['quality_score'],
        'repo_key': cluster_data['repo_key']
    })
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define distinct colors for clusters
    cluster_colors = sns.color_palette("Set2", 3)
    
    # Create scatter plot with custom styling
    scatter = ax.scatter(
        pca_df['PCA1'], 
        pca_df['PCA2'],
        c=[cluster_colors[label] for label in pca_df['Cluster']],
        s=pca_df['Quality'] * 100 + 30,  # Size based on quality score
        alpha=0.7,
        edgecolors=[self.repository_colors.get(repo, 'gray') for repo in pca_df['repo_key']],
        linewidth=1.5
    )
    
    # Add markers for passed PRs
    passed_prs = pca_df[pca_df['Passed'] == True]
    ax.scatter(
        passed_prs['PCA1'],
        passed_prs['PCA2'],
        s=passed_prs['Quality'] * 100 + 30,
        facecolors='none',
        edgecolors='black',
        linewidth=1.5,
        alpha=0.6
    )
    
    # Add cluster centers
    centers = pca.transform(scaler.transform(kmeans.cluster_centers_))
    ax.scatter(
        centers[:, 0], 
        centers[:, 1], 
        s=200,
        marker='X',
        color=cluster_colors,
        edgecolor='black',
        linewidth=2,
        alpha=0.8,
        label='Cluster Centers'
    )
    
    # Add labels for cluster centers
    for i, (x, y) in enumerate(centers):
        ax.text(x, y + 0.2, f'Cluster {i+1}', fontsize=12, ha='center', va='center',
              bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
    
    # Create a custom legend for repositories
    legend_elements = []
    for repo_key, color in self.repository_colors.items():
        if repo_key in pca_df['repo_key'].values:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor='gray', markeredgecolor=color,
                                          markersize=10, markeredgewidth=2,
                                          label=repo_key.replace('_', '/')))
    
    # Add legend elements for clusters
    for i, color in enumerate(cluster_colors):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=color, markersize=10,
                                       label=f'Cluster {i+1}'))
    
    # Add legend element for passed PRs
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor='none', markeredgecolor='black',
                                   markersize=10, markeredgewidth=1.5,
                                   label='Passed All Filters'))
    
    # Add the legend to the plot
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Enhance chart appearance
    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                fontsize=12, weight='bold')
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                fontsize=12, weight='bold')
    ax.set_title('PR Clustering Analysis', fontsize=16, weight='bold', pad=20)
    
    # Add grid
    ax.grid(linestyle='--', alpha=0.3)
    
    # Get feature importances
    feature_importances = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=code_features
    )
    
    # Find most important features for each component
    pc1_features = feature_importances.sort_values('PC1', key=abs, ascending=False)['PC1'].head(3)
    pc2_features = feature_importances.sort_values('PC2', key=abs, ascending=False)['PC2'].head(3)
    
    # Format nice feature names
    feature_names = {
        'code_file_count': 'Code Files',
        'total_changes': 'Total Changes',
        'size_score': 'Size Score',
        'complexity_score': 'Complexity Score',
        'relevance_score': 'Relevance Score',
        'problem_solving_score': 'Problem Solving',
        'code_quality_score': 'Code Quality'
    }
    
    # Create interpretation text
    pc1_text = "PC1 reflects: " + ", ".join([f"{feature_names.get(feat, feat)} ({val:.2f})" 
                                         for feat, val in pc1_features.items()])
    pc2_text = "PC2 reflects: " + ", ".join([f"{feature_names.get(feat, feat)} ({val:.2f})" 
                                         for feat, val in pc2_features.items()])
    
    interpretation = (
        "This plot shows PRs clustered by their code quality dimensions, reduced to 2 dimensions via PCA.\n"
        f"{pc1_text}\n"
        f"{pc2_text}\n"
        "Point size indicates quality score. Marker edge color indicates repository.\n"
        "Black outlines indicate PRs that passed all filters."
    )
    
    fig.text(0.5, 0.01, interpretation, ha='center', fontsize=9, style='italic',
            color='#555555', bbox=dict(facecolor='white', alpha=0.7, pad=5))
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.98])  # Adjust layout to make room for the annotation
    
    # Save to figures directory for reference
    plt.savefig(self.figures_dir / "pr_clustering.png", dpi=300)
    
    # Add to PDF
    plt.suptitle('', y=0.98)  # Add space at top for PDF formatting
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def generate_report(self) -> Path:
    """
    Generate a comprehensive PDF report with enhanced visualizations.
    
    Returns:
        Path to the generated report
    """
    logger.info("Generating enhanced comprehensive filtering report...")
    
    # Create output PDF
    report_path = self.output_dir / f"enhanced_data_curation_report_{self.timestamp}.pdf"
    
    with PdfPages(report_path) as pdf:
        # Cover page
        self._add_enhanced_cover_page(pdf)
        
        # Executive summary
        self._add_enhanced_executive_summary(pdf)
        
        # Cross-repository comparison
        self._add_enhanced_cross_repo_comparison(pdf)
        
        # Add enhanced cross-repository visualizations
        self._add_quality_scatter_plot(pdf)
        self._add_dimension_correlation_heatmap(pdf)
        self._add_pr_clustering_visualization(pdf)
        
        # Skip parallel coordinates as requested in the improvements
        # self._add_parallel_coordinates_plot(pdf)
        
        # Individual repository analyses
        for repo_key in sorted(self.repo_data.keys()):
            self._add_enhanced_repo_section(pdf, repo_key)
        
        # Add exemplary PR profiles
        self._add_enhanced_quality_profiles(pdf)
        
        # Methodology
        self._add_enhanced_methodology_section(pdf)
    
    logger.info(f"Enhanced report generated successfully: {report_path}")
    return report_path

def _add_enhanced_repo_section(self, pdf, repo_key):
    """Add an enhanced repository-specific section to the report with better formatting."""
    repo_name = repo_key.replace("_", "/")
    repo_data = self.repo_data.get(repo_key, {})
    if not repo_data:
        logger.warning(f"No data found for repository {repo_key}")
        return
    
    filter_metadata = repo_data.get("filter_metadata", [])
    filtered_prs = repo_data.get("filtered_prs", [])
    metrics = repo_data.get("metrics", {})
    
    if not filter_metadata:
        logger.warning(f"No filter metadata found for repository {repo_key}")
        return
    
    # Create basic info page with better layout and spacing
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Add repository styling with a gradient background
    gradient = np.linspace(0, 1, 100).reshape(-1, 1) * np.ones((100, 100))
    ax.imshow(gradient, cmap='Blues', alpha=0.05, aspect='auto',
             extent=[0, 1, 0, 1], transform=fig.transFigure)
    
    # Repository title with enhanced styling
    fig.text(0.5, 0.95, f"Repository: {repo_name}", 
            fontsize=22, ha='center', weight='bold', color='#2c3e50')
    
    # Add a horizontal line under the title with gradient
    line_gradient = np.linspace(0.1, 0.9, 100)
    for i, x in enumerate(line_gradient):
        alpha = 1 - abs(2 * (x - 0.5))
        ax.plot([x, x+0.01], [0.92, 0.92], color='#3498db', alpha=alpha, linewidth=2, transform=fig.transFigure)
    
    # Create a styled box for the summary metrics
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((0.1, 0.75), 0.8, 0.12, fill=True, 
                       facecolor='#ecf0f1', alpha=0.8, 
                       boxstyle="round,pad=0.02",
                       transform=fig.transFigure,
                       edgecolor='#bdc3c7', linewidth=1)
    ax.add_patch(box)
    
    # Calculate repository metrics
    total_prs = len(filter_metadata)
    passed_prs = len(filtered_prs)
    pass_rate = passed_prs / total_prs if total_prs > 0 else 0
    data_reduction = 1 - pass_rate
    
    # Count PRs filtered at each stage
    bot_filtered = sum(1 for meta in filter_metadata if not meta.get("bot_filter", {}).get("passed", False))
    size_filtered = sum(1 for meta in filter_metadata 
                       if meta.get("bot_filter", {}).get("passed", False) 
                       and not meta.get("size_filter", {}).get("passed", False))
    content_filtered = sum(1 for meta in filter_metadata 
                          if meta.get("bot_filter", {}).get("passed", False) 
                          and meta.get("size_filter", {}).get("passed", False) 
                          and not meta.get("content_filter", {}).get("passed", False))
    
    # Calculate average quality score
    quality_scores = [meta.get("quality_score", 0) for meta in filter_metadata if meta.get("passed_filter", False)]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    # Summary metrics in styled box
    summary_text = [
        f"Total PRs: {total_prs}",
        f"Passed PRs: {passed_prs} ({pass_rate:.1%})",
        f"Data Reduction: {data_reduction:.1%}",
        f"Average Quality Score: {avg_quality:.2f}"
    ]
    
    fig.text(0.5, 0.81, "\n".join(summary_text), 
            fontsize=12, va='center', ha='center', color='#2c3e50', linespacing=1.8)
    
    # Add filtering breakdown
    fig.text(0.1, 0.7, "Filtering Breakdown:", 
            fontsize=14, va='top', weight='bold', color='#2c3e50')
    
    filter_breakdown = [
        f"• Bot Filter: {bot_filtered} PRs ({bot_filtered/total_prs:.1%})",
        f"• Size Filter: {size_filtered} PRs ({size_filtered/total_prs:.1%})",
        f"• Content Filter: {content_filtered} PRs ({content_filtered/total_prs:.1%})"
    ]
    
    fig.text(0.12, 0.65, "\n".join(filter_breakdown), 
            fontsize=11, va='top', color='#2c3e50', linespacing=1.5)
    
    # Add repository visualization - filter funnel (positioned lower for better spacing)
    self._add_enhanced_filter_funnel(fig, ax, repo_key, filter_metadata, y_position=0.3)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    
    # Add quality distribution in a separate page
    self._add_enhanced_quality_distribution(pdf, repo_key, filter_metadata)
    
    # Add filter analysis in a separate page
    self._add_enhanced_filter_analysis(pdf, repo_key, filter_metadata)

def _add_enhanced_filter_funnel(self, fig, ax, repo_key, filter_metadata, y_position=0.25):
    """Add an enhanced filter funnel visualization to the repository page with better positioning."""
    # Create axes for the funnel chart
    funnel_ax = fig.add_axes([0.1, y_position-0.35, 0.8, 0.35])
    
    # Calculate funnel data
    total_prs = len(filter_metadata)
    
    # Count PRs at each stage
    after_bot = sum(1 for meta in filter_metadata if meta.get("bot_filter", {}).get("passed", False))
    after_size = sum(1 for meta in filter_metadata 
                    if meta.get("bot_filter", {}).get("passed", False) 
                    and meta.get("size_filter", {}).get("passed", False))
    after_content = sum(1 for meta in filter_metadata if meta.get("passed_filter", False))
    
    # Funnel data
    stages = ['Total PRs', 'After Bot Filter', 'After Size Filter', 'After Content Filter']
    counts = [total_prs, after_bot, after_size, after_content]
    
    # Create funnel with enhanced styling
    # Use a gradient of the repository color
    repo_color = self.repository_colors.get(repo_key, 'steelblue')
    colors = [repo_color] + [self._adjust_color_brightness(repo_color, factor) 
                          for factor in [0.8, 0.6, 0.4]]
    
    bars = funnel_ax.bar(stages, counts, color=colors, width=0.6, edgecolor='white', linewidth=0.5)
    
    # Add count and percentage labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = height / total_prs * 100 if total_prs > 0 else 0
        
        funnel_ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, 
                     f"{int(height)}\n({percentage:.1f}%)", 
                     ha='center', va='bottom', fontsize=10, weight='bold',
                     color='#333333')
    
    # Enhance chart appearance
    funnel_ax.set_title('Filter Funnel Analysis', fontsize=14, weight='bold')
    funnel_ax.set_ylabel('Number of PRs', fontsize=11)
    funnel_ax.set_ylim(0, total_prs * 1.2)  # Add space for labels
    funnel_ax.set_xticks(range(len(stages)))
    funnel_ax.set_xticklabels(stages, rotation=30, ha='right')
    
    # Add horizontal grid lines
    funnel_ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add arrows between stages to emphasize the flow
    for i in range(len(counts) - 1):
        # Calculate arrow positions
        x1 = i + 0.25
        x2 = i + 0.75
        y1 = min(counts) * 0.1  # Position arrows lower
        y2 = y1
        
        # Add an arrow
        funnel_ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5, alpha=0.6))

def _add_enhanced_quality_profiles(self, pdf):
    """Add enhanced quality profiles for exemplary PRs with improved formatting."""
    # Find exemplary PRs with high quality scores from each repository
    exemplary_prs = []
    
    for repo_key, data in self.repo_data.items():
        # Get PRs that passed all filters
        filtered_prs = data.get("filtered_prs", [])
        filter_metadata = data.get("filter_metadata", [])
        
        # Create a mapping from PR number to metadata
        pr_to_meta = {
            meta.get("pr_number"): meta 
            for meta in filter_metadata 
            if meta.get("passed_filter", False)
        }
        
        # Sort filtered PRs by quality score (highest first)
        sorted_prs = sorted(
            [(pr, pr_to_meta.get(pr.get("pr_number", 0), {}).get("quality_score", 0)) 
             for pr in filtered_prs],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Take the top PR if available
        if sorted_prs:
            top_pr, score = sorted_prs[0]
            top_pr["repository"] = repo_key.replace("_", "/")
            top_pr["quality_score"] = score
            top_pr["metadata"] = pr_to_meta.get(top_pr.get("pr_number", 0), {})
            exemplary_prs.append(top_pr)
    
    if not exemplary_prs:
        logger.warning("No exemplary PRs found for quality profiles")
        return
    
    # Create introduction page
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Add a light background
    gradient = np.linspace(0, 1, 100).reshape(-1, 1) * np.ones((100, 100))
    ax.imshow(gradient, cmap='Blues', alpha=0.1, aspect='auto',
             extent=[0, 1, 0, 1], transform=fig.transFigure)
    
    # Add title with styling
    fig.text(0.5, 0.95, "Quality Profiles of Exemplary PRs", 
            fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
    # Add a horizontal line under the title with gradient
    line_gradient = np.linspace(0.1, 0.9, 100)
    for i, x in enumerate(line_gradient):
        alpha = 1 - abs(2 * (x - 0.5))
        ax.plot([x, x+0.01], [0.92, 0.92], color='#3498db', alpha=alpha, linewidth=2, transform=fig.transFigure)
    
    # Create a styled box for the explanation
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((0.1, 0.75), 0.8, 0.12, fill=True, 
                       facecolor='#e8f4f8', alpha=0.7, 
                       boxstyle="round,pad=0.02",
                       transform=fig.transFigure,
                       edgecolor='#3498db', linewidth=1)
    ax.add_patch(box)
    
    # Explanation text with improved formatting
    explanation = [
        "The following sections showcase high-quality PRs that passed all filtering stages.",
        "These PRs represent exemplary software engineering data with meaningful",
        "problem-solving content, appropriate size, and high relevance scores.",
        "",
        "Each scorecard provides detailed metrics on the PR's quality dimensions,",
        "including file composition, code changes, and identified relevant files",
        "that provide context for understanding the changes."
    ]
    
    fig.text(0.15, 0.81, "\n".join(explanation), 
            fontsize=12, va='top', color='#2c3e50', linespacing=1.5)
    
    # Add a preview of PR scorecards
    y_pos = 0.6
    for i, pr in enumerate(exemplary_prs[:3]):  # Show up to 3 previews
        # Create a preview box
        preview_box = FancyBboxPatch((0.1, y_pos-0.15), 0.8, 0.15, fill=True, 
                                  facecolor='#f8f9fa', alpha=0.7, 
                                  boxstyle="round,pad=0.02",
                                  transform=fig.transFigure,
                                  edgecolor=self.repository_colors.get(
                                    pr.get("repo_key", pr.get("repository").replace("/", "_")), 
                                    '#bdc3c7'), 
                                  linewidth=2)
        ax.add_patch(preview_box)
        
        # Add PR title and info
        repo_name = pr.get("repository")
        pr_number = pr.get("pr_number")
        title = pr.get("title", "")
        quality_score = pr.get("quality_score", 0)
        
        pr_title = f"PR #{pr_number}: {title}"
        if len(pr_title) > 70:  # Reduced characters to prevent overflow
            pr_title = pr_title[:67] + "..."
            
        fig.text(0.15, y_pos-0.03, pr_title, 
                fontsize=11, va='center', weight='bold', color='#2c3e50')
        
        fig.text(0.15, y_pos-0.07, f"Repository: {repo_name}", 
                fontsize=10, va='center', color='#34495e')
        
        fig.text(0.15, y_pos-0.11, f"Quality Score: {quality_score:.2f}", 
                fontsize=10, va='center', color='#34495e')
        
        # Add a color bar indicating quality
        quality_rect = plt.Rectangle((0.7, y_pos-0.09), 0.15, 0.03, 
                                  color=self._get_quality_color(quality_score))
        ax.add_patch(quality_rect)
        
        # Add icons for quality metrics
        # Use text symbols as a simple alternative to icons
        icons = "★" * int(quality_score * 5)
        fig.text(0.7, y_pos-0.03, icons, 
                fontsize=10, va='center', ha='left', color='goldenrod')
        
        y_pos -= 0.17  # Move down for next preview
    
    # Add note about viewing detailed profiles
    fig.text(0.5, 0.2, "Detailed PR quality scorecards are presented on the following pages", 
            fontsize=12, ha='center', style='italic', color='#7f8c8d')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    
    # Add individual PR scorecards
    for pr in exemplary_prs:
        self._add_enhanced_pr_scorecard(pdf, pr)

def _add_enhanced_pr_scorecard(self, pdf, pr):
    """Add an enhanced quality scorecard for a PR with better formatting."""
    pr_number = pr.get("pr_number")
    repo_name = pr.get("repository")
    repo_key = repo_name.replace("/", "_")
    metadata = pr.get("metadata", {})
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 1.2]})
    
    # Flatten axes for easier access
    axes = axes.flatten()
    
    # Add title with more space above to prevent overlap
    fig.suptitle(f"PR #{pr_number} Quality Scorecard - {repo_name}", 
                fontsize=16, weight='bold', y=0.98)
    
    # 1. Filter scores (top left)
    filter_scores = {
        'Bot Filter': 1.0 - metadata.get("bot_filter", {}).get("details", {}).get("confidence", 0.0),
        'Size Filter': metadata.get("size_filter", {}).get("details", {}).get("normalized_score", 0.0),
        'Content Filter': metadata.get("content_filter", {}).get("details", {}).get("relevance_score", 0.0),
        'Overall Quality': metadata.get("quality_score", 0.0)
    }
    
    # Use enhanced styling for bar chart
    colors = sns.color_palette("viridis", len(filter_scores))
    bars = axes[0].bar(filter_scores.keys(), filter_scores.values(), 
                     color=colors, edgecolor='white', linewidth=0.5)
    
    # Add value labels
    for i, (key, value) in enumerate(filter_scores.items()):
        axes[0].text(i, value + 0.02, f"{value:.2f}", ha='center', va='bottom', 
                   fontsize=9, weight='bold')
    
    # Enhance chart appearance
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Filter Scores", fontsize=12, weight='bold')
    axes[0].set_ylabel("Score (0-1)", fontsize=10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add reference line for quality threshold
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
    axes[0].text(0, 0.5, "Threshold", fontsize=8, color='red', 
               va='bottom', ha='left', alpha=0.7)
    
    # 2. File composition (top right)
    file_counts = {
        'Code': metadata.get("size_filter", {}).get("details", {}).get("code_file_count", 0),
        'Docs': metadata.get("size_filter", {}).get("details", {}).get("doc_file_count", 0),
        'Config': metadata.get("size_filter", {}).get("details", {}).get("config_file_count", 0),
        'Generated': metadata.get("size_filter", {}).get("details", {}).get("generated_file_count", 0),
        'Other': metadata.get("size_filter", {}).get("details", {}).get("other_file_count", 0)
    }
    
    # Remove zero values
    file_counts = {k: v for k, v in file_counts.items() if v > 0}
    
    if file_counts:
        # Use enhanced styling for pie chart
        wedges, texts, autotexts = axes[1].pie(
            file_counts.values(), 
            labels=file_counts.keys(),
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("Set2", len(file_counts)),
            wedgeprops={'edgecolor': 'white', 'linewidth': 1},
            textprops={'fontsize': 9},
        )
        
        # Enhance text appearance
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_weight('bold')
        
        axes[1].set_title("File Type Composition", fontsize=12, weight='bold')
    else:
        axes[1].text(0.5, 0.5, "No file data available", ha='center', va='center')
        axes[1].set_title("File Type Composition (No Data)", fontsize=12, weight='bold')
        axes[1].axis('off')
    
    # 3. Code changes (bottom left)
    change_data = {
        'Additions': metadata.get("size_filter", {}).get("details", {}).get("additions", 0),
        'Deletions': metadata.get("size_filter", {}).get("details", {}).get("deletions", 0)
    }
    
    # Use enhanced styling for bar chart
    bars = axes[2].bar(change_data.keys(), change_data.values(), 
                     color=['green', 'red'], edgecolor='white', linewidth=0.5)
    
    # Add value labels
    for i, (key, value) in enumerate(change_data.items()):
        axes[2].text(i, value + max(change_data.values()) * 0.02, str(int(value)), 
                   ha='center', va='bottom', fontsize=9, weight='bold')
    
    # Enhance chart appearance
    axes[2].set_title("Code Changes", fontsize=12, weight='bold')
    axes[2].set_ylabel("Number of Lines", fontsize=10)
    axes[2].grid(axis='y', linestyle='--', alpha=0.3)
    
    # 4. Relevant files (bottom right) - with improved spacing
    relevant_files = pr.get("relevant_files", [])
    num_relevant = len(relevant_files)
    
    if relevant_files:
        axes[3].axis('off')
        axes[3].set_title("Relevant Files", fontsize=12, weight='bold')
        
        # Create a styled list of relevant files with better spacing
        file_list = f"Files that provide context ({num_relevant} total):"
        axes[3].text(0.5, 0.95, file_list, 
                   ha='center', va='top', fontsize=10, weight='bold')
        
        # Show up to 6 files, with ellipsis if there are more
        display_files = relevant_files[:6]
        if len(relevant_files) > 6:
            display_files.append("... and more")
            
        # Use a cool background for the file list
        file_bg = plt.Rectangle((0.1, 0.1), 0.8, 0.8, 
                              facecolor='#f8f9fa', alpha=0.5, 
                              edgecolor='#bdc3c7', linewidth=1)
        axes[3].add_patch(file_bg)
        
        # Position files with better spacing
        for i, file in enumerate(display_files):
            y_pos = 0.85 - (i * 0.09)  # Increased spacing
            
            # Use different styling for different file types
            if file.endswith(".py"):
                color = "#3572A5"  # Python color
                prefix = "🐍 "
            elif file.endswith(".js"):
                color = "#f1e05a"  # JavaScript color
                prefix = "📜 "
            elif file.endswith(".md"):
                color = "#083fa1"  # Markdown color
                prefix = "📄 "
            elif file.endswith(".json") or file.endswith(".yml") or file.endswith(".yaml"):
                color = "#cb171e"  # Config color
                prefix = "⚙️ "
            elif "..." in file:
                color = "#666666"  # For ellipsis
                prefix = ""
            else:
                color = "#333333"  # Default color
                prefix = "📁 "
            
            # Display filename with word wrapping for long filenames
            if len(file) > 30:
                # Split long filenames to multiple lines
                parts = file.split('/')
                if len(parts) > 2:
                    # Group directory parts on one line, filename on another
                    dir_path = '/'.join(parts[:-1])
                    file_name = parts[-1]
                    axes[3].text(0.15, y_pos, f"{prefix}{dir_path}/", 
                               ha='left', va='center', fontsize=8, color=color)
                    axes[3].text(0.25, y_pos-0.04, f"{file_name}", 
                               ha='left', va='center', fontsize=8, color=color)
                else:
                    axes[3].text(0.15, y_pos, f"{prefix}{file}", 
                               ha='left', va='center', fontsize=8, color=color)
            else:
                axes[3].text(0.15, y_pos, f"{prefix}{file}", 
                           ha='left', va='center', fontsize=9, color=color)
    else:
        axes[3].axis('off')
        axes[3].text(0.5, 0.5, "No relevant files identified", 
                   ha='center', va='center', fontsize=12)
        axes[3].set_title("Relevant Files", fontsize=12, weight='bold')
    
    # Add PR title and description at the bottom with better positioning
    pr_title = pr.get("title", "")
    pr_desc = pr.get("body", "")
    
    # Truncate description if too long
    if pr_desc and len(pr_desc) > 150:  # Further reduced for clarity
        pr_desc = pr_desc[:147] + "..."
        
    # Use a more subtle box for the PR details
    pr_box = plt.Rectangle((0.05, 0.02), 0.9, 0.07, 
                         facecolor='#e8f4f8', alpha=0.5, 
                         edgecolor='#3498db', linewidth=1)
    fig.add_artist(pr_box)
    
    # Add title and truncated description with smaller font
    fig.text(0.07, 0.07, f"Title: {pr_title}", fontsize=9, weight='bold')
    if pr_desc:
        fig.text(0.07, 0.04, f"Description: {pr_desc}", fontsize=8, linespacing=1.2)
    
    # Add more space between sections
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust layout to make room for title and footer
    
    # Add to PDF
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def _add_enhanced_methodology_section(self, pdf):
    """Add an enhanced methodology section to the report with better layout."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Add a light background
    gradient = np.linspace(0, 1, 100).reshape(-1, 1) * np.ones((100, 100))
    ax.imshow(gradient, cmap='Blues', alpha=0.1, aspect='auto',
             extent=[0, 1, 0, 1], transform=fig.transFigure)
    
    # Add section title with styling
    fig.text(0.5, 0.95, "Methodology", 
            fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
    # Add a horizontal line under the title with gradient
    line_gradient = np.linspace(0.1, 0.9, 100)
    for i, x in enumerate(line_gradient):
        alpha = 1 - abs(2 * (x - 0.5))
        ax.plot([x, x+0.01], [0.92, 0.92], color='#3498db', alpha=alpha, linewidth=2, transform=fig.transFigure)
    
    # Introduction to methodology
    intro_text = [
        "The data curation pipeline implements a multi-stage filtering approach inspired by the",
        "SWE-RL paper, focusing on extracting high-quality software engineering data from",
        "GitHub repositories. The pipeline consists of the following key components:"
    ]
    
    # Add introduction text with better spacing
    fig.text(0.1, 0.87, "\n".join(intro_text), 
            fontsize=12, va='top', color='#2c3e50', linespacing=1.5)
    
    # Use styled boxes for each component with increased vertical spacing
    component_colors = ['#e8f8f5', '#eafaf1', '#fef9e7', '#fae5d3']
    component_borders = ['#1abc9c', '#2ecc71', '#f1c40f', '#e67e22']
    component_icons = ['🔍', '⚖️', '🧩', '📊']
    
    # Components with enhanced styling
    components = [
        ("1. Data Acquisition", [
            "• GitHub API integration for PR events and metadata",
            "• Repository cloning for file content access",
            "• Linked issue resolution and context gathering"
        ]),
        ("2. Multi-Stage Filtering", [
            "• Bot and Automation Detection: Identifies and filters out automated PRs",
            "• Size and Complexity Filtering: Ensures PRs are neither trivial nor unwieldy",
            "• Content Relevance Filtering: Focuses on meaningful software engineering content"
        ]),
        ("3. Relevant Files Prediction", [
            "• Identifies semantically related files not modified in the PR",
            "• Uses import analysis and directory structure heuristics",
            "• Enhances context for understanding code changes"
        ]),
        ("4. Quality Metrics Generation", [
            "• Comprehensive quality scoring across multiple dimensions",
            "• Metadata extraction for filtering decisions",
            "• Relevance scoring based on problem-solving indicators"
        ])
    ]
    
    # Position for components with increased spacing
    y_pos = 0.75
    for i, (title, details) in enumerate(components):
        # Create box with enhanced styling
        from matplotlib.patches import FancyBboxPatch
        box_height = 0.13
        box = FancyBboxPatch((0.1, y_pos-box_height), 0.8, box_height, 
                           fill=True, facecolor=component_colors[i], alpha=0.7,
                           boxstyle="round,pad=0.02",
                           transform=fig.transFigure, edgecolor=component_borders[i], 
                           linewidth=2, zorder=1)
        ax.add_patch(box)
        
        # Add icon and title with enhanced styling
        fig.text(0.15, y_pos-0.03, component_icons[i], fontsize=18, ha='left', 
                va='center', color=component_borders[i], weight='bold')
        fig.text(0.2, y_pos-0.03, title, fontsize=14, ha='left', 
                va='center', color='#34495e', weight='bold')
        
        # Add details with better styling and line spacing
        detail_text = "\n".join(details)
        fig.text(0.2, y_pos-0.06, detail_text, fontsize=10, 
                va='top', ha='left', color='#34495e', linespacing=1.3)
        
        # Increase spacing between components
        y_pos -= 0.18
    
    # Add process flow diagram with cleaner arrows
    ax.arrow(0.3, 0.35, 0, -0.05, head_width=0.02, head_length=0.01, 
            fc=component_borders[0], ec=component_borders[0], transform=fig.transFigure)
    ax.arrow(0.5, 0.35, 0, -0.05, head_width=0.02, head_length=0.01, 
            fc=component_borders[1], ec=component_borders[1], transform=fig.transFigure)
    ax.arrow(0.7, 0.35, 0, -0.05, head_width=0.02, head_length=0.01, 
            fc=component_borders[2], ec=component_borders[2], transform=fig.transFigure)
    
    # Final summary with enhanced styling
    from matplotlib.patches import FancyBboxPatch
    summary_box = FancyBboxPatch((0.1, 0.1), 0.8, 0.1, fill=True, 
                               facecolor='#eaecee', alpha=0.7, 
                               boxstyle="round,pad=0.02",
                               transform=fig.transFigure,
                               edgecolor='#7f8c8d', linewidth=1)
    ax.add_patch(summary_box)
    
    conclusion = [
        "The filtering pipeline maintains high precision by using progressive refinement,",
        "ensuring that only PRs with genuine software engineering value are retained",
        "while capturing detailed metadata about filtering decisions and related file context."
    ]
    
    fig.text(0.5, 0.15, "\n".join(conclusion), fontsize=11, ha='center', 
            va='center', color='#2c3e50', style='italic', linespacing=1.3)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)