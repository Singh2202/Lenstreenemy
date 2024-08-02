#!/usr/bin/env python
# coding: utf-8

# In[127]:


import random
import math
import matplotlib.pyplot as plt
import time


# In[26]:


class Point():
    """
    A class used to represent point deflectors.
    """
    def __init__(self, x, y, mass):
        """
        Initialize a Point object.
        
        Parameters
        __________
        
        x (float): x-coord of the point (units tbd)
        y (float): y-coord of the point (units tbd)
        mass (float): mass of the point (units tbd) 
        """
        self.x = x
        self.y = y
        self.mass = mass


# In[27]:


class Node():
    """
    A class used to represent a node of a quadtree.
    Each node corresponds spatially to a square cell
    in space, within which Point objects are contained.
    """
    def __init__(self, x_bl, y_bl, dimension, points, ID):
        """
        Initialize a Node object.
        
        Parameters
        __________
        
        x_bl (float): bottom-left x-coord of the corresponding cell (units tbd)
        y_bl (float): bottom-left x-coord of the corresponding cell (units tbd)
        dimension (float): side length of the corresponding cell (units tbd)
        points (list of Point objects): points contained within the corresponding cell
        ID (int): unique identifer for each cell, ID = \sum_{i=1}^{n}a_i \times 4^{n-i} 
                  n is the number of the level of the square (root n = 0), a_i = 1, 2,
                  3, 4 for cell positions top left, top right, bottom left, bottom right
                  in the level below (2.3.2 in (1))
        """
        self.x_bl = x_bl
        self.y_bl = y_bl
        self.dimension = dimension
        self.points = points
        self.ID = ID
        self.children = []
        self.children_IDs = [4*self.ID + 1, 4*self.ID + 2, 4*self.ID + 3, 4*self.ID + 4]
        self.mass = total_mass(points)
        self.center_of_mass = point_center_of_mass(points)


# In[28]:


def total_mass(points):
    """
    Returns the total mass (float) of points, a
    list of Point objects
    """
    return sum([point.mass for point in points])


# In[29]:


def point_center_of_mass(points):
    """
    Returns the center of mass (Point object) 
    of points, a list of Point objects. The coords
    of the new Point object are given by 
    q_{cm} = \frac{\sum_{i=1}^{N} m_iq_i}{M}, and
    the mass of the new Point object is taken to be 
    total mass of points (units tbd).
    """
    M = total_mass(points)
    x_com = sum([((point.x * point.mass) / M) for point in points])
    y_com = sum([((point.y * point.mass) / M) for point in points])
    return Point(x_com, y_com, M)


# In[30]:


def within_square(x_bl, y_bl, dimension, points):
    """
    Returns a sub-list of Point objects contained within
    a given square, given x_bl, y_bl -- the coordinates
    of the bottom-left corner of the square; dimension --
    the side length of the square; and points, a list of 
    Point objects (all units tbd).
    """
    points_in_square = []
    for point in points:
        if ((x_bl <= point.x) and (point.x <= x_bl + dimension)) and ((y_bl <= point.y) and (point.y <= y_bl + dimension)):
            points_in_square.append(point)
    return points_in_square


# In[31]:


def subdivide(node, internal_nodes_dict, lenses_dict, max_points = 1):
    """
    Given a node, will recursively subdivide the Node into four children
    Nodes until each Node contains at most max_points (int) number 
    of Point objects. Note that this sub-division function operates
    depth-first, i.e. it explores a node and all of its descendants
    before it explores a node's siblings (downwards rather than
    lateral operation). While the subdivision is happening, the data
    is being stored in two dictionaries, internal_nodes_dict and 
    lenses_dict. The keys to these dictionaries are the Node IDs, and
    the entries are the Node objects themselves. The former contains
    those nodes that have children (have more than max_points number
    of points), and the latter contains those nodes that have exactly
    max_points number of points (non-empty leaf nodes). This function
    itself is a helper function for the build_quadtree function.
    """
    
    no_points = len(node.points)
    if no_points <= max_points:
        if no_points == 1:
            lenses_dict[node.ID] = node
        return
    
    # if function does not terminate above it contines to here
    
    internal_nodes_dict[node.ID] = node

    new_dimension = node.dimension / 2.0
  
    points_within_new_tl = within_square(node.x_bl, node.y_bl + new_dimension, new_dimension, node.points)
    id_new_tl = 4 * node.ID + 1
    node_at_new_tl = Node(node.x_bl, node.y_bl + new_dimension, new_dimension, points_within_new_tl, id_new_tl)
    subdivide(node_at_new_tl, internal_nodes_dict, lenses_dict)
    
    points_within_new_tr = within_square(node.x_bl + new_dimension, node.y_bl + new_dimension, new_dimension, node.points)
    id_new_tr = 4 * node.ID + 2
    node_at_new_tr = Node(node.x_bl + new_dimension, node.y_bl + new_dimension, new_dimension, points_within_new_tr, id_new_tr)
    subdivide(node_at_new_tr, internal_nodes_dict, lenses_dict)
    
    points_within_new_bl = within_square(node.x_bl, node.y_bl, new_dimension, node.points)
    id_new_bl = 4 * node.ID + 3
    node_at_new_bl = Node(node.x_bl, node.y_bl, new_dimension, points_within_new_bl, id_new_bl)
    subdivide(node_at_new_bl, internal_nodes_dict, lenses_dict)
    
    points_within_new_br = within_square(node.x_bl + new_dimension, node.y_bl, new_dimension, node.points)
    id_new_br = 4 * node.ID + 4
    node_at_new_br = Node(node.x_bl + new_dimension, node.y_bl, new_dimension, points_within_new_br, id_new_br)
    subdivide(node_at_new_br, internal_nodes_dict, lenses_dict)
    
    node.children = [node_at_new_bl, node_at_new_br, node_at_new_tl, node_at_new_tr]
    
def build_quadtree(root):
    """
    Essentially a wrapper for the subdivide function. Provide root,
    a Node object corresponding to the level 0 square cell that 
    encompasses all of our region. This function will then call the 
    subdivide function on the root node and build the quadtree for the
    entire space, storing the intermediate cells in internal_nodes_dict
    and the non-empty leaf cells in lenses_dict. The reason for this 
    function is that internal_nodes_dict and lenses_dict cannot exist within
    the subdivide function--or else, due to the recursive nature of this function
    they would be cleared on each iteration. Neither can they exist outside of
    the subdivide function--or else they would be global, and would fill with
    redundant information for each run of the whole program.
    """
    
    internal_nodes_dict = {}
    lenses_dict = {}
    subdivide(root, internal_nodes_dict, lenses_dict)
    return internal_nodes_dict, lenses_dict


# In[32]:


class LightRay():
    """
    A class used to represent a lightray, to be used in
    the inverse ray shooting calculation from lens plane
    to source plane.
    """
    def __init__(self, x, y):
        """
        Initialize a LightRay object.
        
        Parameters
        __________
        
        x (float): x-coord of the ray in lens plane (units tbd)
        y (float): y-coord of the ray in lens plane (units tbd)
        """
        self.x = x
        self.y = y

def distance(point, lightray):
    """
    Returns the Euclidean distance (float) between a Point object,
    point, and a LightRay object, lightray.
    """
    
    return math.sqrt((point.x - lightray.x) ** 2 + (point.y - lightray.y) ** 2)

def small_enough(node, ray, delta=0.5):
    """
    For a given ray, certain groupings of stars will be far enough 
    away to have their lensing effect determined as a group (for this
    goes as the inverse of the ray-lens distance), and other stars
    will be close enough to require direct computation of their
    lensing effect (i.e. the deflection angle). Therefore we need a 
    criterion to determine which nearby stars are to be included by 
    direct summation, and which of the more distant stars are to be 
    bunched together in cells of which size. We adopt a criterion
    that is something like the "opening angle" of the cell, as seen 
    from the position of the light ray. If the opening angle is smaller 
    than a specified value of the "accuracy parameter," delta (typically,
    0.4-0.9), then the cell is treated as a pseudo-lens; else, it requires 
    further resolution. This function simply determines whether a cell 
    satisfies the opening angle criterion.
    """
    opening_angle = node.dimension / distance(node.center_of_mass, ray)
    if opening_angle > delta:
        return False
    else:
        return True


# In[33]:


def find_relevant_nodes(node, ray, internal_nodes_dict, lenses_dict):
    """
    For each ray, we must compute a "cell-lens configuration"--i.e.
    we need to determine which groupings of stars can be treated as
    pseudolenses and which lenses need to be treated individually,
    on the basis of distance from the ray. This is done via the 
    opening-angle criterion, implemented in the small_enough method.
    When called on the Root node for a specific ray, this function
    performs lookups in internal_nodes_dict to determine which cells
    satisfy the opening angle criterion, and in lenses_dict otherwise.
    Returns cells_to_be_used, and lenses_to_be_used, lists of Node objects.
    We save the need to do a sort and search as done in Wambsganss, since the
    lookup time in a dictionary is constant (O(1) vs O(n)). In fact, this code
    is ~10**4 times faster than determining cell-lens configurations over
    lists.
    """
    cells_to_be_used = []
    lenses_to_be_used = []

    if small_enough(node, ray):
        cells_to_be_used.append(node)
    else:
        children_keys = [4*node.ID + 1, 4*node.ID + 2, 4*node.ID + 3, 4*node.ID + 4]
        for child_key in children_keys:
            if child_key in internal_nodes_dict:
                child_cells, child_lenses = find_relevant_nodes(internal_nodes_dict[child_key], ray, internal_nodes_dict, lenses_dict)
                cells_to_be_used.extend(child_cells)
                lenses_to_be_used.extend(child_lenses)
            elif child_key in lenses_dict:
                lenses_to_be_used.append(lenses_dict[child_key])
    return cells_to_be_used, lenses_to_be_used


# In[19]:


# example calls 
points = [Point(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)) for _ in range(10**3)]
root = Node(0, 0, 1, points, ID=0)
internal_nodes_dict, lenses_dict = build_quadtree(root)
lightrays = [LightRay(random.uniform(0,1), random.uniform(0,1)) for i in range(5)]
cells, lenses = find_relevant_nodes(root, lightrays[0], internal_nodes_dict, lenses_dict)


# In[34]:


def deflection_angle(node, ray):
    """
    Given a Node object (corresponding to a (psuedo)star) and 
    a LightRay object, ray, this function will compute the 
    deflection of the ray induced by the presence of the psuedostar.
    The deflection angle is a vector quantity; as such the result is 
    reported as a tuple, with its first element the x-element of the
    deflection and the second element the y-element. Units tbd.
    """
    G = 1.0 # replace with physical value
    c = 1.0 # replace with physical value 
    dx = node.center_of_mass.x - ray.x # use here units consistent with G and c
    dy = node.center_of_mass.y - ray.y
    xi_squared = dx**2 + dy**2
    const = (4*G*node.mass)/(c**2*xi_squared)
    return (const*dx, const*dy) # want this in arcseconds -- by default, the result will be in radians


# In[22]:


# deprecated 

def grid(center, x_extent, y_extent, no_points_along_x, no_points_along_y):
    leftmost_x = center[0] - x_extent / 2
    bottommost_y = center[1] - y_extent / 2
    spacing_between_x_vals = x_extent / no_points_along_x
    spacing_between_y_vals = y_extent / no_points_along_y
    x_start = leftmost_x + spacing_between_x_vals / 2
    y_start = bottommost_y +  spacing_between_y_vals / 2
    xs = [x_start + i*spacing_between_x_vals for i in range(0, no_points_along_x) for j in range(no_points_along_y)]
    ys = [y_start + i*spacing_between_y_vals for i in range(0, no_points_along_y)]*no_points_along_x
    lightrays = [LightRay(x, y) for x, y in zip(xs, ys)]
    return lightrays


# In[35]:


def grid(center, side_length, no_points):
    """
    Given a center (tuple), a side_length (float), and a number 
    of points, no_points (int), this function will return a list of 
    LightRay objects corresponding to a no_points * no_points regular
    gridding of the square specified by center and side length.
    """
    leftmost_x = center[0] - side_length / 2
    bottommost_y = center[1] - side_length / 2
    spacing_between_x_vals = side_length / no_points
    spacing_between_y_vals = side_length / no_points
    x_start = leftmost_x + spacing_between_x_vals / 2
    y_start = bottommost_y +  spacing_between_y_vals / 2
    xs = [x_start + i*spacing_between_x_vals for i in range(0, no_points) for j in range(no_points)]
    ys = [y_start + i*spacing_between_y_vals for i in range(0, no_points)]*no_points
    lightrays = [LightRay(x, y) for x, y in zip(xs, ys)]
    return lightrays


# In[39]:


# example calls 
#points = [Point(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)) for _ in range(10**3)]
#root = Node(0, 0, 1, points, ID=0)
#internal_nodes_dict, lenses_dict = build_quadtree(root)
#lightrays = [LightRay(random.uniform(0,1), random.uniform(0,1)) for i in range(5)]
#cells, lenses = find_relevant_nodes(root, lightrays[0], internal_nodes_dict, lenses_dict)

def deflection_angle_map(x_bl, y_bl, side_length, stars, no_primary_rays, n_fix):
    """
    The essence of the program. Specify the bottom left coordinates of the 
    image plane region, as well its side length. Specify the star field as a 
    list of Point objects -- this will be integrated with the IMF sampling 
    functionality. Specify the number of primary rays -- a regular grid of 
    no_primary_rays * no_primary_rays will be superimposed on the square. 
    Specify the number n_fix -- around each primary ray, a grid centered 
    on this ray of n_fix * n_fix rays will be shot, using the same cell-lens
    configuration as the primary ray. This function will return a list of 
    coordinates in the image plane along with the deflection experienced 
    by a light ray at that point, in the format (x, y, (alpha_x, alpha_y)).
    """
    root = Node(x_bl, y_bl, side_length, stars, ID=0)
    internal_nodes_dict, lenses_dict = build_quadtree(root)
    center = (x_bl + side_length / 2, y_bl + side_length / 2)
    primary_rays = grid(center, side_length, no_primary_rays)
    deflections_at_points = []
    for primary_ray in primary_rays:
        relevant_cells, relevant_lenses = find_relevant_nodes(root, primary_ray, internal_nodes_dict, lenses_dict)
        for secondary_ray in grid((primary_ray.x, primary_ray.y), side_length / no_primary_rays, n_fix):
            dx_due_to_pseudostars = 0
            dy_due_to_pseudostars = 0
            dx_due_to_stars = 0
            dy_due_to_stars = 0 
            for pseudostar in relevant_cells:
                dx, dy = deflection_angle(pseudostar, secondary_ray)
                dx_due_to_pseudostars += dx
                dy_due_to_pseudostars += dy
            for star in relevant_lenses:
                dx, dy = deflection_angle(star, secondary_ray)
                dx_due_to_stars += dx
                dy_due_to_stars += dy
            total_dx = dx_due_to_pseudostars + dx_due_to_stars
            total_dy = dy_due_to_pseudostars + dy_due_to_stars
            deflections_at_points.append((secondary_ray.x, secondary_ray.y, (total_dx, total_dy))) 
    return deflections_at_points

# note -- 10**6 rays with 10**3 stars takes ~25 seconds


# In[123]:


def plot_magnification_map(deflections_at_points):
    """
    The deflections_at_points list is the output from the 
    deflection_angle_map function, representing the deflection 
    experienced by a lightray shot through a given point on the
    image plane. To obtain the light collection on the source plane,
    then, all we need to do is (vectorially) add the lightray position
    and the deflection angle.
    """
    # actually this is not true -- there is some trig that needs to happen 
    # here -- you can't just add an angle to "Cartesian coordinates"
    x_in_source_plane = [data[0] + data[2][0] for data in deflections_at_points]
    y_in_source_plane = [data[1] + data[2][1] for data in deflections_at_points]
    plt.scatter(x_in_source_plane, y_in_source_plane)


# In[129]:


# example use with timing
"""
start = time.time()
x_bl = 0
y_bl = 0 
side_length = 1
stars = [Point(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)) for _ in range(10**3)]
no_primary_rays = 100
n_fix = 10
m = deflection_angle_map(x_bl, y_bl, side_length, stars, no_primary_rays, n_fix) 
plot_magnification_map(m)
end = time.time()
print(end-start)
"""

