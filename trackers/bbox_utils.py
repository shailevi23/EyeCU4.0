"""
Utility functions for handling bounding boxes
"""

def get_center_of_bbox(bbox):
    """
    Get the center point of a bounding box
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        tuple: (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox):
    """
    Get the width of a bounding box
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        int: Width of bounding box
    """
    return bbox[2] - bbox[0]

def measure_distance(p1, p2):
    """
    Measure Euclidean distance between two points
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        
    Returns:
        float: Euclidean distance
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def measure_xy_distance(p1, p2):
    """
    Measure separate x and y distances between two points
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        
    Returns:
        tuple: (x_distance, y_distance)
    """
    return p1[0] - p2[0], p1[1] - p2[1]

def get_foot_position(bbox):
    """
    Get the foot position of a player (bottom center of bbox)
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        tuple: (center_x, bottom_y)
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)