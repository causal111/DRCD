import csv
import os
import sys
import subprocess
from collections import defaultdict

# Function to read CSV file
def read_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

# Function to extract edges for each method
def extract_edges(data, method):
    edges = []
    
    for row in data:
        parts = row['Dataset'].split('_vs_')
        if len(parts) == 2:
            left_node, right_node = parts
            try:
                value = float(row[method])
                if value == 1:
                    # Edge from top to bottom
                    edges.append({'source': left_node, 'target': right_node})
                elif value == 2 or value == 2.0:
                    # Edge from bottom to top
                    edges.append({'source': right_node, 'target': left_node})
            except (ValueError, TypeError):
                # Skip if value is not a number
                pass
    
    return edges

# Function to separate bipartite graph nodes into top and bottom
def determine_bipartite_nodes(data):
    top_set = set()
    bottom_set = set()
    
    # Collect top and bottom nodes from dataset
    for row in data:
        parts = row['Dataset'].split('_vs_')
        if len(parts) == 2:
            top_set.add(parts[0])
            bottom_set.add(parts[1])
    
    # Sort alphabetically
    top_nodes = sorted(list(top_set))
    bottom_nodes = sorted(list(bottom_set))
    
    return {
        'top': top_nodes,
        'bottom': bottom_nodes
    }


def optimize_edge_connections(edges, top_positions, bottom_positions):
    node_in_edges = defaultdict(list)
    node_out_edges = defaultdict(list)
    
    for edge in edges:
        node_in_edges[edge['target']].append(edge)
        node_out_edges[edge['source']].append(edge)
    
    leftmost_top_node = None
    leftmost_bottom_node = None
    
    if top_positions:
        leftmost_top_x = float('inf')
        for node, pos in top_positions.items():
            if pos['x'] < leftmost_top_x:
                leftmost_top_x = pos['x']
                leftmost_top_node = node
        rightmost_top_x = -float('inf')
        for node, pos in top_positions.items():
            if pos['x'] > rightmost_top_x:
                rightmost_top_x = pos['x']
                rightmost_top_node = node
    
    if bottom_positions:
        leftmost_bottom_x = float('inf')
        for node, pos in bottom_positions.items():
            if pos['x'] < leftmost_bottom_x:
                leftmost_bottom_x = pos['x']
                leftmost_bottom_node = node
        rightmost_bottom_x = -float('inf')
        for node, pos in bottom_positions.items():
            if pos['x'] > rightmost_bottom_x:
                rightmost_bottom_x = pos['x']
                rightmost_bottom_node = node
    
    for edge in edges:
        if edge['source'] == leftmost_top_node and edge['target'] == leftmost_bottom_node:
            edge['target_offset'] = -0.5
            edge['special_case'] = True
    
        if edge['source'] == leftmost_bottom_node and edge['target'] == leftmost_top_node:
            edge['target_offset'] = -0.5
            edge['special_case'] = True
    
        if edge['source'] == rightmost_top_node and edge['target'] == rightmost_bottom_node:
            edge['target_offset'] = 0.5
            edge['special_case'] = True
    
        if edge['source'] == rightmost_bottom_node and edge['target'] == rightmost_top_node:
            edge['target_offset'] = 0.5
            edge['special_case'] = True
    
    for target_node, in_edges in node_in_edges.items():
        num_in = len(in_edges)
        if num_in == 0:
            continue
        
        target_pos = None
        if target_node in top_positions:
            target_pos = top_positions[target_node]
        elif target_node in bottom_positions:
            target_pos = bottom_positions[target_node]
        
        if not target_pos:
            continue
        
        if num_in == 1:
            edge = in_edges[0]

            if edge.get('special_case', False):
                continue
                
            source_node = edge['source']
            source_pos = None
            
            if source_node in top_positions:
                source_pos = top_positions[source_node]
            elif source_node in bottom_positions:
                source_pos = bottom_positions[source_node]
            
            if source_pos:
                source_x = source_pos['x'] + source_pos['width'] / 2
                target_center_x = target_pos['x'] + target_pos['width'] / 2
                is_left_source = source_x < target_center_x
                
                if target_node in bottom_positions:
                    if is_left_source:
                        edge['target_offset'] = -0.7
                    else:
                        edge['target_offset'] = 0.7
                
                elif target_node in top_positions:
                    if is_left_source:
                        edge['target_offset'] = -0.7
                    else:
                        edge['target_offset'] = 0.7
                
                continue
        
        if num_in == 2:
            left_edges = []
            right_edges = []
            
            for edge in in_edges:
                if edge.get('special_case', False):
                    continue
                    
                source_node = edge['source']
                source_pos = None
                
                if source_node in top_positions:
                    source_pos = top_positions[source_node]
                elif source_node in bottom_positions:
                    source_pos = bottom_positions[source_node]
                
                if source_pos:
                    source_x = source_pos['x'] + source_pos['width'] / 2
                    target_center_x = target_pos['x'] + target_pos['width'] / 2
                    
                    if source_x < target_center_x:
                        left_edges.append(edge)
                    else:
                        right_edges.append(edge)
            
            if left_edges and right_edges:
                for edge in left_edges:
                    if target_node in bottom_positions:
                        edge['target_offset'] = -0.7
                    elif target_node in top_positions:
                        edge['target_offset'] = -0.7
                
                for edge in right_edges:
                    if target_node in bottom_positions:
                        edge['target_offset'] = 0.7
                    elif target_node in top_positions:
                        edge['target_offset'] = 0.7
                
                continue
        
        left_sources = []
        right_sources = []
        
        for edge in in_edges:
            if edge.get('special_case', False) or 'target_offset' in edge:
                continue
                
            source_node = edge['source']
            source_pos = None
            
            if source_node in top_positions:
                source_pos = top_positions[source_node]
            elif source_node in bottom_positions:
                source_pos = bottom_positions[source_node]
            
            if source_pos:
                source_x = source_pos['x'] + source_pos['width'] / 2
                target_center_x = target_pos['x'] + target_pos['width'] / 2
                if source_x < target_center_x:
                    left_sources.append((edge, source_x))
                else:
                    right_sources.append((edge, source_x))
        
        left_sources.sort(key=lambda x: x[1])
        right_sources.sort(key=lambda x: -x[1])
        
        num_left = len(left_sources)
        if num_left > 0:
            left_half_width = target_pos['width'] / 2
            segment_width = left_half_width / (num_left + 1)
            
            for i, (edge, _) in enumerate(left_sources):
                segment_pos = i + 1
                offset_from_left = segment_width * segment_pos
                
                edge['target_offset'] = (offset_from_left - left_half_width) / left_half_width
        
        num_right = len(right_sources)
        if num_right > 0:
            right_half_width = target_pos['width'] / 2
            segment_width = right_half_width / (num_right + 1)
            
            for i, (edge, _) in enumerate(right_sources):
                segment_pos = i + 1
                offset_from_right = segment_width * segment_pos
                
                edge['target_offset'] = (right_half_width - offset_from_right) / right_half_width
    
    for edge in edges:
        edge['source_offset'] = 0.0
    
    return edges


# SVG generation function - 3 rows by 2 columns grid layout
def generate_grid_bipartite_svg(methods, bipartite_nodes, edges_by_method, DRCDonly):
    # Grid layout settings
    columns = 2  # 2 columns
    rows = 3     # 3 rows
    graph_width = 350  # Width of each graph
    graph_height = 140  # Graph height reduced to 0.7x (about 0.7x of original 170 ≈ 120, but set to 140 for margin)
    if DRCDonly:
        columns = 1
        rows = 1
    total_width = graph_width * columns  # Total width
    total_height = graph_height * rows   # Total height
    
    # SVG start tag
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="{total_height}" viewBox="0 0 {total_width} {total_height}">'
    
    # Font definition (specify more readable font)
    svg += '<defs>'
    # Font family definition
    svg += '<style type="text/css">@import url("https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&amp;display=swap");</style>'
    
    # Arrow marker definition (separate for top-to-bottom and bottom-to-top)
    for i in range(len(methods)):
        # Top-to-bottom arrow (black)
        svg += f'<marker id="arrow-down-{i}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">'
        svg += '<polygon points="0 0, 10 3.5, 0 7" fill="black" />'
        svg += '</marker>'
        
        # Bottom-to-top arrow (blue)
        svg += f'<marker id="arrow-up-{i}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">'
        svg += '<polygon points="0 0, 10 3.5, 0 7" fill="blue" />'
        svg += '</marker>'
    svg += '</defs>'
    
    # Create graph for each method
    for method_index, method in enumerate(methods):
        # Calculate position in grid
        row = method_index // columns
        col = method_index % columns
        x_offset = col * graph_width
        y_offset = row * graph_height
        
        top_nodes = bipartite_nodes['top']  # Top nodes (alphabetical order)
        bottom_nodes = bipartite_nodes['bottom']  # Bottom nodes (alphabetical order)
        
        # Copy edges
        edges = edges_by_method[method].copy()
        
        # Group element
        svg += f'<g transform="translate({x_offset}, {y_offset})">'
        
        # Border and background
        svg += f'<rect x="2" y="2" width="{graph_width-4}" height="{graph_height-4}" fill="none" stroke="#ddd" stroke-width="1" rx="5" />'
        
        # Title (larger font, bold)
        svg += f'<text x="{graph_width/2}" y="20" text-anchor="middle" font-weight="bold" font-size="16" font-family="Roboto, Arial, sans-serif">{method}</text>'
        
        # Node size and placement calculation
        node_width = 42  # Width reduced to 0.8x
        node_height = 14  # Height reduced to 0.6x (0.6x of original 24 ≈ 14)
        margin_x = 10  # Minimize X margin
        top_y = 30  # Start Y coordinate for top nodes slightly higher
        bottom_y = graph_height - node_height - 10  # Bottom nodes position moved up
        
        # Available width
        available_width = graph_width - 2 * margin_x
        
        # Calculate space between nodes
        top_spacing = available_width / max(len(top_nodes), 1)
        if top_spacing < node_width + 5:  # Prevent overlap
            # Need to split into 2 rows
            top_row1_count = len(top_nodes) // 2 + (len(top_nodes) % 2)
            top_row2_count = len(top_nodes) // 2
            top_spacing1 = available_width / max(top_row1_count, 1)
            top_spacing2 = available_width / max(top_row2_count, 1)
        else:
            top_row1_count = len(top_nodes)
            top_row2_count = 0
            top_spacing1 = top_spacing
            top_spacing2 = 0
        
        bottom_spacing = available_width / max(len(bottom_nodes), 1)
        if bottom_spacing < node_width + 5:  # Prevent overlap
            # Need to split into 2 rows
            bottom_row1_count = len(bottom_nodes) // 2 + (len(bottom_nodes) % 2)
            bottom_row2_count = len(bottom_nodes) // 2
            bottom_spacing1 = available_width / max(bottom_row1_count, 1)
            bottom_spacing2 = available_width / max(bottom_row2_count, 1)
        else:
            bottom_row1_count = len(bottom_nodes)
            bottom_row2_count = 0
            bottom_spacing1 = bottom_spacing
            bottom_spacing2 = 0
        
        # Draw top nodes - move slightly to the right
        top_node_positions = {}
        
        # Make left margin slightly larger to move to the right (half the movement)
        margin_adjustment = margin_x + 5  # Move 5 pixels to the right (half of original 10)
        
        for i, node in enumerate(top_nodes):
            if i < top_row1_count:
                # First row
                x = margin_adjustment + (i * top_spacing1)
                y = top_y
            else:
                # Second row
                x = margin_adjustment + ((i - top_row1_count) * top_spacing2)
                y = top_y + node_height
            
            # Add half the width to keep the node center
            x_center = x + node_width / 2
            if x_center > graph_width - node_width / 2:
                x = graph_width - node_width - 5
                
            top_node_positions[node] = {'x': x, 'y': y, 'bottomY': y + node_height, 'width': node_width}
            
            # Node background with slightly lighter color
            svg += f'<rect x="{x}" y="{y}" width="{node_width}" height="{node_height}" rx="5" stroke="black" fill="#f8f8f8" stroke-width="1" />'
            # Text perfectly centered
            svg += f'<text x="{x + node_width/2}" y="{y + node_height/2}" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="bold" font-family="Roboto, Arial, sans-serif">{node}</text>'
        
        # Draw bottom nodes - back to original position
        bottom_node_positions = {}
        
        for i, node in enumerate(bottom_nodes):
            if i < bottom_row1_count:
                # First row
                x = margin_x + (i * bottom_spacing1)
                y = bottom_y
            else:
                # Second row
                x = margin_x + ((i - bottom_row1_count) * bottom_spacing2)
                y = bottom_y - node_height
            
            # Add half the width to keep the node center
            x_center = x + node_width / 2
            if x_center > graph_width - node_width / 2:
                x = graph_width - node_width - 5
                
            bottom_node_positions[node] = {'x': x, 'y': y, 'topY': y, 'width': node_width}
            
            # Node background with slightly lighter color
            svg += f'<rect x="{x}" y="{y}" width="{node_width}" height="{node_height}" rx="5" stroke="black" fill="#f8f8f8" stroke-width="1" />'
            # Text perfectly centered
            svg += f'<text x="{x + node_width/2}" y="{y + node_height/2}" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="bold" font-family="Roboto, Arial, sans-serif">{node}</text>'
        
        # Edge preprocessing - optimize connection points using specified method
        edges = optimize_edge_connections(edges, top_node_positions, bottom_node_positions)
        
        # Draw edges
        for edge in edges:
            source_pos = None
            target_pos = None
            
            # Calculate source node position (always center)
            if edge['source'] in top_node_positions:
                source_node = top_node_positions[edge['source']]
                source_pos = {
                    'x': source_node['x'] + source_node['width']/2,  # Center
                    'y': source_node['bottomY']
                }
            elif edge['source'] in bottom_node_positions:
                source_node = bottom_node_positions[edge['source']]
                source_pos = {
                    'x': source_node['x'] + source_node['width']/2,  # Center
                    'y': source_node['topY']
                }
            
            # Calculate target node position (based on divisions)
            if edge['target'] in top_node_positions:
                target_node = top_node_positions[edge['target']]
                target_offset = edge.get('target_offset', 0) * target_node['width'] / 2
                target_pos = {
                    'x': target_node['x'] + target_node['width']/2 + target_offset,
                    'y': target_node['bottomY']+1.0
                }
            elif edge['target'] in bottom_node_positions:
                target_node = bottom_node_positions[edge['target']]
                target_offset = edge.get('target_offset', 0) * target_node['width'] / 2
                target_pos = {
                    'x': target_node['x'] + target_node['width']/2 + target_offset,
                    'y': target_node['topY']-1.0
                }
            
            if source_pos and target_pos:
                # Change line style for top-to-bottom and bottom-to-top
                svg += f'<path d="M{source_pos["x"]},{source_pos["y"]} '
                
                # Straight line case
                if edge['source'] in top_node_positions and edge['target'] in bottom_node_positions:
                    # Top-to-bottom edge (solid line, black)
                    svg += f'L{target_pos["x"]},{target_pos["y"]}" '
                    svg += f'stroke="black" stroke-width="0.8" fill="none" stroke-dasharray="none" '
                    svg += f'marker-end="url(#arrow-down-{method_index})" />'
                elif edge['source'] in bottom_node_positions and edge['target'] in top_node_positions:
                    # Bottom-to-top edge (dotted line, blue)
                    svg += f'L{target_pos["x"]},{target_pos["y"]}" '
                    svg += f'stroke="blue" stroke-width="0.8" fill="none" stroke-dasharray="3,2" '
                    svg += f'marker-end="url(#arrow-up-{method_index})" />'
            
        svg += '</g>'
    
    svg += '</svg>'
    return svg

# Function to convert SVG to PDF
def convert_svg_to_pdf(svg_path, pdf_path):
    try:
        # Convert with cairosvg if installed
        import cairosvg
        cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
        print(f"PDF file saved: {pdf_path}")
        return True
    except ImportError:
        # Try external commands if cairosvg is not available
        try:
            # Method using Inkscape
            result = subprocess.run(['inkscape', '--export-filename=' + pdf_path, svg_path], 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                print(f"PDF file saved: {pdf_path} (using Inkscape)")
                return True
        except FileNotFoundError:
            pass
            
        try:
            # Method using rsvg-convert
            result = subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', pdf_path, svg_path], 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                print(f"PDF file saved: {pdf_path} (using rsvg-convert)")
                return True
        except FileNotFoundError:
            pass
        
        print("Warning: PDF conversion tool not found. Please install cairosvg, Inkscape, or rsvg-convert.")
        return False

# Main function
def main(csv_file_path, output_svg_path, DRCDonly=False):
    # Read CSV file
    data = read_csv(csv_file_path)
    
    # List of methods
    methods = ["DRCD", "LiM", "MIC", "CRACK", "HCM", "GSF"]
    if DRCDonly:
        methods = ["DRCD"]
    
    # Bipartite graph node division (sorted alphabetically)
    bipartite_nodes = determine_bipartite_nodes(data)
    
    # Extract edges for each method
    edges_by_method = {}
    for method in methods:
        edges_by_method[method] = extract_edges(data, method)
    
    # Generate SVG
    svg = generate_grid_bipartite_svg(methods, bipartite_nodes, edges_by_method, DRCDonly)
    
    # Save to SVG file
    with open(output_svg_path, 'w', encoding='utf-8') as f:
        f.write(svg)
    print(f"SVG file saved: {output_svg_path}")
    
    # Convert to PDF
    output_pdf_path = output_svg_path.replace('.svg', '.pdf')
    convert_svg_to_pdf(output_svg_path, output_pdf_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_improved_readability_graph.py <input CSV file> <output SVG file>")
        sys.exit(1)
    
    csv_file_path = sys.argv[1]
    output_svg_path = sys.argv[2]
    
    main(csv_file_path, output_svg_path)