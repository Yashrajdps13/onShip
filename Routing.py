import numpy as np
import heapq
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from math import radians, sin, cos, sqrt, atan2
import math
import pickle

with open('metaData.pkl', 'rb') as file:
    metaData = pickle.load(file)
    
with open('tree.pkl', 'rb') as file:
    tree = pickle.load(file)

with open('wards_gdf.pkl', 'rb') as file:
    wards_gdf = pickle.load(file)

with open('ward_metaData.pkl', 'rb') as file:
    ward_metaData = pickle.load(file)
    
with open('distance_matrix.pkl', 'rb') as file:
    distance_matrix = pickle.load(file)

with open('wards.pkl', 'rb') as file:
    wards = pickle.load(file)    
    
def lat_lon_to_cartesian(lat, lon):
    # Convert latitude and longitude to radians
    lat = math.radians(lat)
    lon = math.radians(lon)

    # Earth radius in meters (mean radius)
    R = 6371  # km

    # Cartesian coordinates
    x = R * math.cos(lat) * math.cos(lon)
    y = R * math.cos(lat) * math.sin(lon)
    z = R * math.sin(lat)

    return x, y, z

# Helper function to convert Cartesian coordinates back to lat/lon
def cartesian_to_lat_lon(x, y, z):
    # Earth radius in meters (mean radius)
    R = 6371  # km

    # Latitude and longitude in radians
    lat = math.asin(z / R)
    lon = math.atan2(y, x)

    # Convert back to degrees
    lat = math.degrees(lat)
    lon = math.degrees(lon)

    return lat, lon

# Function to find the center of a quadrilateral given 4 vertices (lat, lon)
def find_centroid(quad_vertices):
    # Convert each vertex from lat/lon to Cartesian coordinates
    cartesian_coords = [lat_lon_to_cartesian(lat, lon) for lat, lon in quad_vertices]

    # Calculate the average of the Cartesian coordinates (centroid)
    x_avg = sum([coord[0] for coord in cartesian_coords]) / 4
    y_avg = sum([coord[1] for coord in cartesian_coords]) / 4
    z_avg = sum([coord[2] for coord in cartesian_coords]) / 4

    # Convert the centroid back to latitude and longitude
    centroid_lat, centroid_lon = cartesian_to_lat_lon(x_avg, y_avg, z_avg)

    return centroid_lat, centroid_lon

    
def dijkstra_matrix(matrix, start, end):
    n = len(matrix)  
    distances = [float('inf')] * n  
    distances[start] = 0  
    previous_nodes = [None] * n  
    visited = [False] * n  
    pq = [(0, start)]  

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if visited[current_node]:
            continue

        
        if current_node == end:
            break

        visited[current_node] = True

        
        for neighbor in range(n):
            if matrix[current_node][neighbor] != float('inf') and not visited[neighbor]:
                distance = current_distance + matrix[current_node][neighbor]
                
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))

    
    path = []
    current_node = end
    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]
    
    
    if distances[end] == float('inf'):
        return None, float('inf')  

    return path, distances[end]


def find_ward(lat, lon):
    point = Point(lon, lat)  
    for _, row in wards_gdf.iterrows():
        if row['geometry'].contains(point):
            return row['ward_name']
    return None  

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c  
    return distance

def update_matrix_with_cyclone(distance_matrix, ward_centroids, cyclone_center, cyclone_radius):
    
    cyclone_lat, cyclone_lon = cyclone_center

    
    affected_wards = []
    for i, (ward_lat, ward_lon) in ward_centroids.items():
        distance_to_cyclone = haversine(cyclone_lat, cyclone_lon, ward_lat, ward_lon)
        if distance_to_cyclone <= cyclone_radius:
            affected_wards.append(i)

    
    for ward in affected_wards:
        for i in range(len(distance_matrix)):
            if distance_matrix[ward][i] != float('inf'):
                distance_matrix[ward][i] *= 1000
            if distance_matrix[i][ward] != float('inf'):
                distance_matrix[i][ward] *= 1000

    return distance_matrix

def getPath(start_node, end_node):
    
    start_ward=find_ward(start_node[0],start_node[1])
    end_ward=find_ward(end_node[0],end_node[1])
    ward_number_str1 = start_ward.split('_')[1]
    start = int(ward_number_str1) - 1
    ward_number_str2 = end_ward.split('_')[1]
    end = int(ward_number_str2) - 1
    path, shortest_distance = dijkstra_matrix(distance_matrix, start,end)


    
    coordinates=[]
    coordinates.append(start_node)
    coordinates=[ward_metaData[x] for x in path]
    coordinates.append(end_node)
    return coordinates,shortest_distance

def getCyclonePath(start_node, end_node, cyclone_center, cyclone_radius):
    
    start_ward=find_ward(start_node[0],start_node[1])
    end_ward=find_ward(end_node[0],end_node[1])
    ward_number_str1 = start_ward.split('_')[1]
    start = int(ward_number_str1) - 1
    ward_number_str2 = end_ward.split('_')[1]
    end = int(ward_number_str2) - 1
    
    updated_matrix = update_matrix_with_cyclone(distance_matrix.copy(), ward_metaData, cyclone_center, cyclone_radius)
    path, shortest_distance = dijkstra_matrix(updated_matrix, start, end)
    
    coordinates=[]
    coordinates.append(start_node)
    coordinates=[ward_metaData[x] for x in path]
    coordinates.append(end_node)
    
    return coordinates,shortest_distance
