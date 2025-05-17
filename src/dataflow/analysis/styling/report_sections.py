def add_pr_scorecard(generator, pdf, pr):
    """Add an enhanced quality scorecard for a PR with better formatting and spacing."""
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
        file_bg = Rectangle((0.1, 0.1), 0.8, 0.8, 
                          facecolor='#f8f9fa', alpha=0.5, 
                          edgecolor='#bdc3c7', linewidth=1)
        axes[3].add_patch(file_bg)
        
        # Position files with better spacing
        for i, file in enumerate(display_files):
            y_pos = 0.85 - (i * 0.09)  # Increased spacing
            
            # Use different styling for different file types
            if file.endswith(".py"):
                color = "#3572A5"  # Python color
                prefix = "PY "
            elif file.endswith(".js"):
                color = "#f1e05a"  # JavaScript color
                prefix = "JS "
            elif file.endswith(".md"):
                color = "#083fa1"  # Markdown color
                prefix = "MD "
            elif file.endswith(".json") or file.endswith(".yml") or file.endswith(".yaml"):
                color = "#cb171e"  # Config color
                prefix = "CF "
            elif "..." in file:
                color = "#666666"  # For ellipsis
                prefix = ""
            else:
                color = "#333333"  # Default color
                prefix = "FL "
            
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
    pr_box = Rectangle((0.05, 0.02), 0.9, 0.07, 
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

def add_methodology_section(generator, pdf):
    """Add an enhanced methodology section to the report with better layout."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Add a light background
    add_gradient_background(ax, alpha=0.1)
    
    # Add section title with styling
    fig.text(0.5, 0.95, "Methodology", 
            fontsize=24, ha='center', weight='bold', color='#2c3e50')
    
    # Add a horizontal line under the title with gradient
    add_gradient_line(fig, 0.1, 0.9, 0.92, color='#3498db')
    
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
    component_icons = ['1️⃣', '2️⃣', '3️⃣', '4️⃣']
    
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
        box_height = 0.13
        box = FancyBboxPatch((0.1, y_pos-box_height), 0.8, box_height, 
                           fill=True, facecolor=component_colors[i], alpha=0.7,
                           boxstyle="round,pad=0.02",
                           transform=fig.transFigure, edgecolor=component_borders[i], 
                           linewidth=2, zorder=1)
        ax.add_patch(box)
        
        # Add number and title with enhanced styling
        fig.text(0.15, y_pos-0.03, component_icons[i], fontsize=14, ha='left', 
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