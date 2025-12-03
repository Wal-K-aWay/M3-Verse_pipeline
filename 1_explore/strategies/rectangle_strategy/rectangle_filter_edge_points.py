from typing import List, Tuple, Set
from collections import deque

def get_channel_width(x: float, z: float, coords_set: Set[Tuple[float, float]], 
                        grid_size: float, direction: str) -> float:
    if direction == 'horizontal':
        directions = [(-grid_size, 0), (grid_size, 0)]
        current_pos = [x, z]
        pos_idx = 0
    else:  # vertical
        directions = [(0, -grid_size), (0, grid_size)]
        current_pos = [x, z]
        pos_idx = 1
    
    total_extent = grid_size
    
    for dx, dz in directions:
        extent = 0
        current = current_pos[:]
        while True:
            current[0] = round(current[0] + dx, 2)
            current[1] = round(current[1] + dz, 2)
            if tuple(current) in coords_set:
                extent += grid_size
            else:
                break
        total_extent += extent
    
    return total_extent

def is_narrow_channel(x: float, z: float, coords_set: Set[Tuple[float, float]], 
                        grid_size: float, channel_width: int = 3) -> bool:
    horizontal_width = get_channel_width(x, z, coords_set, grid_size, 'horizontal')
    vertical_width = get_channel_width(x, z, coords_set, grid_size, 'vertical')
    
    max_narrow_width = channel_width * grid_size + 1e-6
    return horizontal_width <= max_narrow_width or vertical_width <= max_narrow_width

def detect_narrow_channel_edges(coords_set: Set[Tuple[float, float]], 
                                edge_points: Set[Tuple[float, float]], 
                                grid_size: float) -> Set[Tuple[float, float]]:
    return {point for point in edge_points 
            if is_narrow_channel(point[0], point[1], coords_set, grid_size)}

def find_edge_points(coords_set: Set[Tuple[float, float]], 
                        grid_size: float) -> Set[Tuple[float, float]]:
    edge_points = set()
    directions = [(grid_size, 0), (-grid_size, 0), (0, grid_size), (0, -grid_size)]
    
    for x, z in coords_set:
        neighbors = [(round(x + dx, 2), round(z + dz, 2)) for dx, dz in directions]
        if any(neighbor not in coords_set for neighbor in neighbors):
            edge_points.add((x, z))
    
    return edge_points

def is_connected(coords_set: Set[Tuple[float, float]], grid_size: float) -> bool:
    """检查坐标集合是否连通"""
    if not coords_set:
        return True
    
    # BFS检查连通性
    start = next(iter(coords_set))
    visited = {start}
    queue = deque([start])
    directions = [(grid_size, 0), (-grid_size, 0), (0, grid_size), (0, -grid_size)]
    
    while queue:
        x, z = queue.popleft()
        for dx, dz in directions:
            neighbor = (round(x + dx, 2), round(z + dz, 2))
            if neighbor in coords_set and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return len(visited) == len(coords_set)

def is_bridge_point(point: Tuple[float, float], coords_set: Set[Tuple[float, float]], 
                   grid_size: float) -> bool:
    """检查某个点是否是桥接点（删除后会导致图不连通）"""
    # 临时移除该点
    coords_without_point = coords_set - {point}
    
    # 检查移除后是否仍然连通
    return not is_connected(coords_without_point, grid_size)

def get_connected_components(coords_set: Set[Tuple[float, float]], 
                           grid_size: float) -> List[Set[Tuple[float, float]]]:
    """获取所有连通分量"""
    components = []
    unvisited = coords_set.copy()
    directions = [(grid_size, 0), (-grid_size, 0), (0, grid_size), (0, -grid_size)]
    
    while unvisited:
        start = next(iter(unvisited))
        component = {start}
        queue = deque([start])
        unvisited.remove(start)
        
        while queue:
            x, z = queue.popleft()
            for dx, dz in directions:
                neighbor = (round(x + dx, 2), round(z + dz, 2))
                if neighbor in unvisited:
                    component.add(neighbor)
                    queue.append(neighbor)
                    unvisited.remove(neighbor)
        
        components.append(component)
    
    return components

def filter_edge_points(coords: List[Tuple[float, float]], grid_size: float, agent_position: Tuple[float, float]) -> List[Tuple[float, float]]:
    coords_set = set(coords)
    # from .rectangle_visualization import visualize_sample_points
    
    edge_points = find_edge_points(coords_set, grid_size)
    # visualize_sample_points(filtered_reachable_coords=coords, intersection_points=edge_points, save_path='edge_points.png')

    narrow_channel_edges = detect_narrow_channel_edges(coords_set, edge_points, grid_size)
    # visualize_sample_points(filtered_reachable_coords=coords, intersection_points=narrow_channel_edges, save_path='narrow_channel_edges.png')
    
    candidate_removable = edge_points - narrow_channel_edges
    
    agent_pos_rounded = (round(agent_position[0], 2), round(agent_position[1], 2))
    candidate_removable.discard(agent_pos_rounded)
    
    safe_removable = set()
    current_coords = coords_set.copy()
    
    for point in candidate_removable:
        if not is_bridge_point(point, current_coords, grid_size):
            safe_removable.add(point)
            current_coords.remove(point)  # 从当前坐标集中移除，用于后续检查
    
    # visualize_sample_points(filtered_reachable_coords=coords, intersection_points=safe_removable, save_path='removable_edges.png')
    # import pdb;pdb.set_trace()
    
    filtered_coords = [coord for coord in coords if coord not in safe_removable]
    
    if not is_connected(set(filtered_coords), grid_size):
        print("Warning: Filtered result is not connected, returning original coordinates")
        return coords
    
    print(f"Successfully removed {len(safe_removable)} edge points while maintaining connectivity")
    return filtered_coords