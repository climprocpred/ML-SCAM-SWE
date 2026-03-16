
# %%
import math

def analyze_disco_for_resolution(nlat, nlon=None, theta_cutoffs=None):
    """
    Analyze Morlet kernel options for a given grid resolution. 
    
    Parameters
    ----------
    nlat : int
        Number of latitude points
    nlon : int, optional
        Number of longitude points (default: 2 * nlat)
    theta_cutoffs :  list, optional
        List of theta_cutoff values to analyze
    """
    if nlon is None: 
        nlon = 2 * nlat
    
    if theta_cutoffs is None:
        grid_spacing = math.pi / (nlat - 1)
        theta_cutoffs = [grid_spacing * mult for mult in [2, 3, 4, 5, 6, 8, 10]]
    
    dlat = math.pi / (nlat - 1)
    dlon = 2 * math. pi / nlon
    equator_km = 40075 / nlon  # Earth circumference / nlon
    
    print(f"Grid Resolution: {nlat} × {nlon}")
    print(f"Latitude spacing:   {dlat:.4f} rad ({math.degrees(dlat):.2f}°)")
    print(f"Longitude spacing: {dlon:.4f} rad ({math. degrees(dlon):.2f}°)")
    print(f"Approx. spacing at equator: ~{equator_km:.0f} km")
    print()
    print("=" * 95)
    print(f"{'theta_cutoff':^20} | {'FoV diameter':^14} | {'Points @equator':^15} | {'Suggested Morlet kernel_shape':<30}")
    print(f"{'(rad / degrees)':^20} | {'(km @equator)':^14} | {'(worst case)':^15} | {'(n_radial, n_angular) → size':<30}")
    print("-" * 95)
    
    for theta_cutoff in theta_cutoffs: 
        points_equator = estimate_points_in_disk_at_latitude(nlat, nlon, theta_cutoff, latitude_deg=0)
        fov_km = 2 * theta_cutoff * 6371  # Earth radius * angle = arc length
        suggestions = suggest_morlet_kernel_shapes(points_equator)
        
        print(f"{theta_cutoff:>8.4f} / {math.degrees(theta_cutoff):>6.2f}°  | "
              f"{fov_km:>10.0f} km  | {points_equator:>11.0f}    | {suggestions}")
    
    print("=" * 95)


def estimate_points_in_disk_at_latitude(nlat, nlon, theta_cutoff, latitude_deg):
    """Estimate grid points in disk at given latitude."""
    dlat = math.pi / (nlat - 1)
    dlon = 2 * math.pi / nlon
    
    colat_center = math.pi / 2 - math.radians(latitude_deg)
    n_lat_cells = 2 * theta_cutoff / dlat
    
    sin_colat = math.sin(colat_center)
    effective_lon_spacing = dlon * max(sin_colat, 0.01)
    n_lon_cells = 2 * theta_cutoff / effective_lon_spacing
    
    points = (math.pi / 4) * n_lat_cells * n_lon_cells
    return max(1, points)


def suggest_morlet_kernel_shapes(points_in_disk):
    """Suggest Morlet kernel_shape options."""
    suggestions = []
    
    options = [(2, 2), (3, 3), (4, 4), (5, 5), (4, 8), (6, 6), (8, 8)]
    
    for kernel_shape in options:
        kernel_size = kernel_shape[0] * kernel_shape[1]
        
        if points_in_disk >= kernel_size: 
            ratio = points_in_disk / kernel_size
            if ratio >= 3:
                quality = "✓✓"
            elif ratio >= 1.5:
                quality = "✓ "
            else:
                quality = "~ "
            suggestions. append(f"{quality}{kernel_shape}→{kernel_size}w")
    
    if not suggestions:
        return "increase theta_cutoff"
    
    return ", ".join(suggestions[: 3])


# Compare different resolutions
if __name__ == "__main__":
    nlat = 256
    nlon = 512
    analyze_disco_for_resolution(nlat, nlon=256)
    print("\n")
# %%
