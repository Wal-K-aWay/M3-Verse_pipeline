import math

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_angle(p1, p2, p3):
    a = calculate_distance(p2, p3)
    b = calculate_distance(p1, p3)
    c = calculate_distance(p1, p2)
    # Cosine rule
    if 2 * a * c == 0:
        return 0
    angle = math.acos((a**2 + c**2 - b**2) / (2 * a * c))
    return math.degrees(angle)

def is_l_shape(points):
    if len(points) != 6:
        return False

    angles = []
    for i in range(6):
        p1 = points[i]
        p2 = points[(i + 1) % 6]
        p3 = points[(i + 2) % 6]
        angles.append(round(calculate_angle(p1, p2, p3), 2))
    
    ninety_degree_angles = sum(1 for angle in angles if abs(angle - 90) < 1e-9)
    two_seventy_degree_angles = sum(1 for angle in angles if abs(angle - 270) < 1e-9)

    # This is a very specific check for a standard L-shape polygon's internal angles.
    # A simpler check for rectilinear polygons (all angles are 90 or 270)
    is_rectilinear = all(abs(angle % 90) < 1e-9 for angle in angles)
    if not is_rectilinear:
        return False

    # Check for the characteristic angles of an L-shape (4 right angles, 2 reflex angles)
    if ninety_degree_angles == 4 and two_seventy_degree_angles == 2:
        return True

    # Alternative check for convex hull properties or other geometric features might be more robust.
    # For now, we stick to a simple angle check.
    # A simple L-shape has 6 vertices, 4 interior angles of 90 degrees, and 2 of 270 degrees (or one of 270 and one of -90 depending on vertex order and concavity)
    # Let's refine the check for angles
    angle_counts = {90: 0, 270: 0}
    internal_angles = []
    for i in range(len(points)):
        p_prev = points[i-1]
        p_curr = points[i]
        p_next = points[(i+1)%len(points)]
        
        v1 = (p_prev[0] - p_curr[0], p_prev[1] - p_curr[1])
        v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
        
        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        cross_product = v1[0]*v2[1] - v1[1]*v2[0]
        
        angle = math.atan2(cross_product, dot_product)
        internal_angle = 180 - math.degrees(angle)
        if internal_angle < 0:
            internal_angle += 360
        internal_angles.append(round(internal_angle))

    for angle in internal_angles:
        if abs(angle - 90) < 5:
            angle_counts[90] += 1
        elif abs(angle - 270) < 5:
            angle_counts[270] += 1

    return angle_counts[90] == 5 and angle_counts[270] == 1

def classify_octagon_shape(points):
    """Classifies octagon shapes: Concave, Staircase, T-shape, Z-shape"""
    if len(points) != 8:
        return None

    # Calculate all internal angles
    internal_angles = []
    for i in range(len(points)):
        p_prev = points[i-1]
        p_curr = points[i]
        p_next = points[(i+1)%len(points)]
        
        v1 = (p_prev[0] - p_curr[0], p_prev[1] - p_curr[1])
        v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
        
        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        cross_product = v1[0]*v2[1] - v1[1]*v2[0]
        
        angle = math.atan2(cross_product, dot_product)
        internal_angle = 180 - math.degrees(angle)
        if internal_angle < 0:
            internal_angle += 360
        internal_angles.append(round(internal_angle))

    # Check if it's a rectilinear polygon (all angles are 90 or 270 degrees)
    is_rectilinear = all(abs(angle % 90) < 5 for angle in internal_angles)
    if not is_rectilinear:
        return None

    # Count angles
    angle_counts = {90: 0, 270: 0}
    for angle in internal_angles:
        if abs(angle - 90) < 5:
            angle_counts[90] += 1
        elif abs(angle - 270) < 5:
            angle_counts[270] += 1

    # Must have 6 90-degree angles and 2 270-degree angles
    if not (angle_counts[90] == 6 and angle_counts[270] == 2):
        return None

    # Find the positions of the two 270-degree angles
    angle_270_positions = []
    for i, angle in enumerate(internal_angles):
        if abs(angle - 270) < 5:
            angle_270_positions.append(i)
    
    if len(angle_270_positions) != 2:
        return None
    
    pos1, pos2 = angle_270_positions
    
    # Calculate the distance between the two 270-degree angles (considering circular array)
    distance1 = (pos2 - pos1) % 8
    distance2 = (pos1 - pos2) % 8
    min_distance = min(distance1, distance2)
    
    # Determine shape type based on distance
    if min_distance == 1:
        return "Concave"  # Two 270-degree angles are adjacent
    elif min_distance == 2:
        return "Staircase"  # Two 270-degree angles separated by one 90-degree angle
    elif min_distance == 3:
        return "T-shape"    # Two 270-degree angles separated by two 90-degree angles
    elif min_distance == 4:
        return "Z-shape"    # Two 270-degree angles separated by three 90-degree angles
    else:
        return None

def _extract_vertices(points):
    if len(points) < 3:
        return points
    
    vertices = []
    for i in range(len(points)):
        p_prev = points[i - 1]
        p_curr = points[i]
        p_next = points[(i + 1) % len(points)]
        
        # Check for collinearity using the cross-product. If it's non-zero, the point is a vertex.
        val = (p_curr[1] - p_prev[1]) * (p_next[0] - p_curr[0]) - \
              (p_curr[0] - p_prev[0]) * (p_next[1] - p_curr[1])
              
        if abs(val) > 1e-9:  # Not collinear
            vertices.append(p_curr)
            
    return vertices

def analyze_polygon(points):
    points = _extract_vertices(points)
    num_points = len(points)
    
    if is_l_shape(points):
        return "L-shape"
    
    # Check for special octagon shapes
    if num_points == 8:
        octagon_shape = classify_octagon_shape(points)
        if octagon_shape:
            return octagon_shape

    if num_points == 4:
        sides = [calculate_distance(points[i], points[(i + 1) % 4]) for i in range(4)]
        angles = [calculate_angle(points[i], points[(i + 1) % 4], points[(i + 2) % 4]) for i in range(4)]
        
        is_rectangle = all(abs(angle - 90) < 1e-9 for angle in angles)

        if is_rectangle:
            return "Rectangle"

    return None