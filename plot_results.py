import sys

def parse_results(filename):
    n_vals = []
    speedup_v1_vals = []
    speedup_v2_vals = []
    speedup_v3_vals = []
    speedup_v4_vals = []
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        start_parsing = False
        for line in lines:
            if "N,Naive(ms)" in line:
                start_parsing = True
                continue
            if start_parsing:
                parts = line.strip().split(',')
                # skip empty lines or headers
                if not line.strip() or "N,Naive(ms)" in line or len(parts) < 8:
                    continue
                    try:
                        n = int(parts[0])
                        # n:0, na:1, fl:2, v2:3, v3:4, v4:5, s2:6, s4:7
                        
                        naive = float(parts[1])
                        time_v1 = float(parts[2])
                        time_v2 = float(parts[3])
                        time_v3 = float(parts[4])
                        time_v4 = float(parts[5])
                        
                        if naive > 0: 
                            n_vals.append(n)
                            speedup_v1_vals.append(naive/time_v1 if time_v1>0 else 0)
                            speedup_v2_vals.append(naive/time_v2 if time_v2>0 else 0)
                            speedup_v3_vals.append(naive/time_v3 if time_v3>0 else 0)
                            speedup_v4_vals.append(naive/time_v4 if time_v4>0 else 0)
                    except ValueError:
                        continue
    except FileNotFoundError:
        print("File not found")
        return [], [], [], [], []

    return n_vals, speedup_v1_vals, speedup_v2_vals, speedup_v3_vals, speedup_v4_vals

def generate_svg(n_vals, speedup_v1, speedup_v2, speedup_v3, speedup_v4, filename='speedup_plot.svg'):
    width = 800
    height = 600
    padding = 60
    
    if not n_vals:
        return

    min_n = min(n_vals)
    max_n = max(n_vals)
    max_speedup = max(max(speedup_v1), max(speedup_v2), max(speedup_v3), max(speedup_v4)) * 1.1
    
    import math
    log_min_n = math.log2(min_n)
    log_max_n = math.log2(max_n)
    
    def get_x(n):
        return padding + (math.log2(n) - log_min_n) / (log_max_n - log_min_n) * (width - 2 * padding)
        
    def get_y(s):
        return height - padding - (s / max_speedup) * (height - 2 * padding)
        
    svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n'
    
    # Background
    svg += f'<rect width="100%" height="100%" fill="white" />\n'
    
    # Axes
    svg += f'<line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="black" stroke-width="2"/>\n'
    svg += f'<line x1="{padding}" y1="{height-padding}" x2="{padding}" y2="{padding}" stroke="black" stroke-width="2"/>\n'
    
    # Grid and Labels (Y Axis)
    for i in range(0, int(max_speedup) + 1, max(1, int(max_speedup)//10)):
        y = get_y(i)
        svg += f'<line x1="{padding}" y1="{y}" x2="{width-padding}" y2="{y}" stroke="lightgray" stroke-dasharray="5,5"/>\n'
        svg += f'<text x="{padding-10}" y="{y+5}" text-anchor="end" font-family="Arial" font-size="12">{i}</text>\n'
        
    # X Axis
    for n in n_vals:
        x = get_x(n)
        svg += f'<line x1="{x}" y1="{height-padding}" x2="{x}" y2="{padding}" stroke="lightgray" stroke-dasharray="5,5"/>\n'
        svg += f'<text x="{x}" y="{height-padding+20}" text-anchor="middle" font-family="Arial" font-size="12">{n}</text>\n'
        
    # Title and Labels
    svg += f'<text x="{width/2}" y="{padding/2}" text-anchor="middle" font-family="Arial" font-size="20" font-weight="bold">FlashAttention Speedup (vs Naive)</text>\n'
    
    # Legend
    svg += f'<rect x="{width-150}" y="{padding}" width="10" height="10" fill="blue" />'
    svg += f'<text x="{width-135}" y="{padding+10}" font-family="Arial" font-size="12">Flash V1</text>'
    svg += f'<rect x="{width-150}" y="{padding+20}" width="10" height="10" fill="green" />'
    svg += f'<text x="{width-135}" y="{padding+30}" font-family="Arial" font-size="12">Flash V2 (Float4)</text>'
    svg += f'<rect x="{width-150}" y="{padding+40}" width="10" height="10" fill="red" />'
    svg += f'<text x="{width-135}" y="{padding+50}" font-family="Arial" font-size="12">Flash V3 (Mat FP32)</text>'
    svg += f'<rect x="{width-150}" y="{padding+60}" width="10" height="10" fill="purple" />'
    svg += f'<text x="{width-135}" y="{padding+70}" font-family="Arial" font-size="12">Flash V4 (Mat FP16)</text>'
    
    # Plot Lines
    def plot_line(vals, color):
        nonlocal svg
        points = []
        for n, s in zip(n_vals, vals):
            points.append(f'{get_x(n)},{get_y(s)}')
        svg += f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>\n'
        for n, s in zip(n_vals, vals):
            x = get_x(n)
            y = get_y(s)
            svg += f'<circle cx="{x}" cy="{y}" r="4" fill="{color}" />\n'
            svg += f'<text x="{x}" y="{y-10}" text-anchor="middle" font-family="Arial" font-size="10" fill="{color}">{s:.1f}x</text>\n'

    plot_line(speedup_v1, "blue")
    plot_line(speedup_v2, "green")
    plot_line(speedup_v3, "red")
    plot_line(speedup_v4, "purple")
    
    svg += '</svg>'
    
    with open(filename, 'w') as f:
        f.write(svg)
    print(f"SVG saved to {filename}")

if __name__ == "__main__":
    n, v1, v2, v3, v4 = parse_results('benchmark_results.csv')
    generate_svg(n, v1, v2, v3, v4)
