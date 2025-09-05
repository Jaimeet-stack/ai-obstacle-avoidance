import pygame
import numpy as np
import math
import random
import time
import sys
import json
from pygame.locals import *
from collections import deque
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Initialize pygame
pygame.init()
pygame.font.init()

# Screen dimensions
WIDTH, HEIGHT = 1600, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Ultra-Advanced AI-Powered Indoor Obstacle Avoidance System")

# Colors
BACKGROUND = (240, 240, 245)
GRID_COLOR = (220, 220, 230)
WALL_COLOR = (50, 50, 70)
AGENT_COLORS = [
    (41, 128, 185),  # Blue
    (231, 76, 60),   # Red
    (46, 204, 113),  # Green
    (155, 89, 182),  # Purple
    (241, 196, 15),  # Yellow
    (230, 126, 34)   # Orange
]
AGENT_OUTLINE = (30, 30, 30)
PATH_COLORS = [
    (52, 152, 219, 180),
    (231, 76, 60, 180),
    (46, 204, 113, 180),
    (155, 89, 182, 180),
    (241, 196, 15, 180),
    (230, 126, 34, 180)
]
SENSOR_COLOR = (52, 152, 219, 100)
OBSTACLE_COLOR = (149, 165, 166)
TEXT_COLOR = (44, 62, 80)
BUTTON_COLOR = (52, 152, 219)
BUTTON_HOVER_COLOR = (41, 128, 185)
BUTTON_TEXT_COLOR = (255, 255, 255)
PANEL_COLOR = (220, 220, 230, 230)
SLIDER_COLOR = (150, 150, 160)
SLIDER_HANDLE_COLOR = (80, 80, 100)
HEATMAP_COLORS = [(0, 0, 255, 50), (0, 255, 0, 50), (255, 255, 0, 50), (255, 0, 0, 50)]
MENU_COLOR = (30, 40, 50)
MENU_HIGHLIGHT = (41, 128, 185)
SUCCESS_COLOR = (46, 204, 113)
WARNING_COLOR = (241, 196, 15)
ERROR_COLOR = (231, 76, 60)
NEUTRAL_COLOR = (149, 165, 166)

# Fonts
font = pygame.font.SysFont('Arial', 16)
title_font = pygame.font.SysFont('Arial', 28, bold=True)
small_font = pygame.font.SysFont('Arial', 14)
large_font = pygame.font.SysFont('Arial', 36, bold=True)
console_font = pygame.font.SysFont('Courier New', 14)

# Environment templates
ENVIRONMENT_TEMPLATES = {
    "Office": {
        "walls": [
            (100, 100, 1100, 20), (100, 100, 20, 700), (100, 780, 1100, 20), (1180, 100, 20, 700),
            (300, 200, 20, 300), (500, 200, 20, 300), (700, 200, 20, 300), (900, 200, 20, 300),
            (200, 400, 200, 20), (400, 400, 200, 20), (600, 400, 200, 20), (800, 400, 200, 20),
            (1000, 400, 200, 20)
        ],
        "doorways": [
            (350, 400, 50, 20), (550, 400, 50, 20), (750, 400, 50, 20), (950, 400, 50, 20)
        ]
    },
    "Warehouse": {
        "walls": [
            (100, 100, 1100, 20), (100, 100, 20, 700), (100, 780, 1100, 20), (1180, 100, 20, 700),
            (400, 100, 20, 300), (700, 100, 20, 300), (1000, 100, 20, 300),
            (250, 400, 300, 20), (550, 400, 300, 20), (850, 400, 300, 20),
            (250, 600, 300, 20), (550, 600, 300, 20), (850, 600, 300, 20)
        ],
        "doorways": [
            (500, 400, 100, 20), (800, 400, 100, 20), (200, 600, 100, 20), (500, 600, 100, 20),
            (800, 600, 100, 20)
        ]
    },
    "Maze": {
        "walls": [
            (100, 100, 1100, 20), (100, 100, 20, 700), (100, 780, 1100, 20), (1180, 100, 20, 700),
            (200, 200, 20, 400), (300, 100, 20, 400), (400, 300, 20, 400), (500, 100, 20, 400),
            (600, 200, 20, 400), (700, 100, 20, 400), (800, 300, 20, 400), (900, 100, 20, 400),
            (1000, 200, 20, 400), (1100, 100, 20, 400),
            (200, 200, 800, 20), (300, 300, 800, 20), (200, 400, 800, 20), (300, 500, 800, 20),
            (200, 600, 800, 20)
        ],
        "doorways": [
            (250, 200, 50, 20), (450, 300, 50, 20), (650, 400, 50, 20), (850, 500, 50, 20),
            (250, 600, 50, 20)
        ]
    },
    "Smart Home": {
        "walls": [
            (100, 100, 1100, 20), (100, 100, 20, 700), (100, 780, 1100, 20), (1180, 100, 20, 700),
            (400, 100, 20, 250), (700, 100, 20, 250), (1000, 100, 20, 250),
            (250, 350, 300, 20), (550, 350, 300, 20), (850, 350, 300, 20),
            (400, 450, 20, 250), (700, 450, 20, 250), (1000, 450, 20, 250),
            (250, 650, 300, 20), (550, 650, 300, 20), (850, 650, 300, 20)
        ],
        "doorways": [
            (500, 350, 100, 20), (800, 350, 100, 20), (500, 650, 100, 20), (800, 650, 100, 20)
        ]
    }
}

class IndoorEnvironment:
    def __init__(self, width, height, env_type="Office"):
        self.width = width
        self.height = height
        self.env_type = env_type
        self.obstacles = []
        self.walls = ENVIRONMENT_TEMPLATES[env_type]["walls"]
        self.doorways = ENVIRONMENT_TEMPLATES[env_type]["doorways"]
        self.generate_obstacles()
        self.generate_furniture()
        self.heatmap = np.zeros((width//10, height//10))
        self.start_time = time.time()
        self.mission_time = 0
        self.charging_stations = self.generate_charging_stations()
        
    def generate_charging_stations(self):
        stations = []
        if self.env_type == "Office":
            stations = [(1100, 150), (1100, 650), (200, 650)]
        elif self.env_type == "Warehouse":
            stations = [(200, 200), (200, 600), (1100, 200), (1100, 600)]
        elif self.env_type == "Maze":
            stations = [(1100, 150), (300, 700), (800, 700)]
        elif self.env_type == "Smart Home":
            stations = [(150, 150), (150, 700), (1100, 700)]
        
        return [{"pos": pos, "in_use": False} for pos in stations]
    
    def generate_obstacles(self):
        # Generate random static obstacles
        for _ in range(12):
            x = random.randint(120, 1160)
            y = random.randint(120, 760)
            size = random.randint(20, 50)
            self.obstacles.append({
                'type': 'static',
                'rect': pygame.Rect(x, y, size, size),
                'color': OBSTACLE_COLOR
            })
            
        # Add some dynamic obstacles that will move
        for _ in range(6):
            x = random.randint(120, 1160)
            y = random.randint(120, 760)
            size = random.randint(25, 40)
            speed = random.uniform(0.5, 2.0)
            direction = [random.uniform(-1, 1), random.uniform(-1, 1)]
            # Normalize direction
            norm = math.sqrt(direction[0]**2 + direction[1]**2)
            direction[0] /= norm
            direction[1] /= norm
            self.obstacles.append({
                'type': 'dynamic',
                'rect': pygame.Rect(x, y, size, size),
                'speed': speed,
                'direction': direction,
                'color': (100, 150, 150)
            })
    
    def generate_furniture(self):
        # Add some furniture-like obstacles based on environment type
        if self.env_type == "Office":
            furniture = [
                (150, 150, 80, 120), (250, 550, 100, 100), (450, 550, 120, 80),
                (750, 150, 100, 120), (950, 550, 80, 100), (1050, 200, 100, 80)
            ]
        elif self.env_type == "Warehouse":
            furniture = [
                (200, 200, 100, 100), (200, 500, 100, 100), (500, 200, 100, 100),
                (500, 500, 100, 100), (800, 200, 100, 100), (800, 500, 100, 100),
                (1100, 200, 60, 300)
            ]
        elif self.env_type == "Maze":
            furniture = [
                (300, 250, 70, 70), (500, 350, 70, 70), (700, 250, 70, 70),
                (900, 350, 70, 70), (1100, 250, 70, 70), (300, 550, 70, 70),
                (500, 650, 70, 70), (700, 550, 70, 70), (900, 650, 70, 70)
            ]
        elif self.env_type == "Smart Home":
            furniture = [
                (200, 200, 100, 80), (200, 500, 100, 80), (500, 200, 80, 100),
                (500, 500, 80, 100), (800, 200, 120, 80), (800, 500, 120, 80),
                (1100, 400, 60, 200)
            ]
        
        for f in furniture:
            self.obstacles.append({
                'type': 'furniture',
                'rect': pygame.Rect(f),
                'color': (120, 100, 80)
            })
    
    def update_dynamic_obstacles(self):
        for obstacle in self.obstacles:
            if obstacle['type'] == 'dynamic':
                rect = obstacle['rect']
                direction = obstacle['direction']
                speed = obstacle['speed']
                
                # Move the obstacle
                rect.x += direction[0] * speed
                rect.y += direction[1] * speed
                
                # Bounce off walls
                if rect.left < 120 or rect.right > 1180:
                    direction[0] *= -1
                if rect.top < 120 or rect.bottom > 780:
                    direction[1] *= -1
                    
                # Occasionally change direction randomly
                if random.random() < 0.01:
                    new_direction = [random.uniform(-1, 1), random.uniform(-1, 1)]
                    norm = math.sqrt(new_direction[0]**2 + new_direction[1]**2)
                    obstacle['direction'] = [new_direction[0]/norm, new_direction[1]/norm]
    
    def update_heatmap(self, x, y):
        # Update heatmap with agent position
        grid_x, grid_y = min(int(x / 10), self.heatmap.shape[0]-1), min(int(y / 10), self.heatmap.shape[1]-1)
        self.heatmap[grid_x, grid_y] += 1
    
    def update_mission_time(self):
        self.mission_time = time.time() - self.start_time
    
    def draw(self, screen):
        # Draw background grid
        for x in range(100, 1200, 50):
            pygame.draw.line(screen, GRID_COLOR, (x, 100), (x, 780), 1)
        for y in range(100, 800, 50):
            pygame.draw.line(screen, GRID_COLOR, (100, y), (1180, y), 1)
        
        # Draw walls
        for wall in self.walls:
            pygame.draw.rect(screen, WALL_COLOR, wall)
        
        # Draw doorways
        for doorway in self.doorways:
            pygame.draw.rect(screen, BACKGROUND, doorway)
            pygame.draw.rect(screen, WALL_COLOR, doorway, 2)
        
        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(screen, obstacle['color'], obstacle['rect'])
            if obstacle['type'] == 'furniture':
                # Add some detail to furniture
                pygame.draw.rect(screen, (80, 70, 60), obstacle['rect'], 2)
        
        # Draw charging stations
        for station in self.charging_stations:
            color = (46, 204, 113) if not station['in_use'] else (241, 196, 15)
            pygame.draw.circle(screen, color, station['pos'], 15)
            pygame.draw.circle(screen, (30, 30, 30), station['pos'], 15, 2)
            pygame.draw.circle(screen, (30, 30, 30), station['pos'], 8, 1)
    
    def draw_heatmap(self, screen):
        # Draw heatmap of agent positions
        max_val = np.max(self.heatmap) if np.max(self.heatmap) > 0 else 1
        for x in range(self.heatmap.shape[0]):
            for y in range(self.heatmap.shape[1]):
                if self.heatmap[x, y] > 0:
                    intensity = min(255, int(255 * self.heatmap[x, y] / max_val))
                    color = (255, 0, 0, intensity//2)
                    pygame.draw.rect(screen, color, (x*10, y*10, 10, 10))

class Agent:
    def __init__(self, x, y, color, path_color, agent_type="Basic AI"):
        self.x = x
        self.y = y
        self.radius = 15
        self.speed = 3.0
        self.max_speed = 5.0
        self.direction = 0  # Angle in radians
        self.sensors = []
        self.path = []
        self.current_target = 0
        self.max_sensor_range = 200
        self.sensor_angles = [-math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2]
        self.collision_points = []
        self.clusters = []
        self.color = color
        self.path_color = path_color
        self.type = agent_type
        self.target_x, self.target_y = x, y
        self.memory = []  # For storing previously visited locations
        self.memory_size = 100
        self.completed_paths = 0
        self.total_distance = 0
        self.last_position = (x, y)
        self.collisions = 0
        self.efficiency = 1.0
        self.algorithm = self.basic_navigation
        self.set_algorithm(agent_type)
        self.status = "Idle"
        self.battery = 100
        self.battery_drain_rate = 0.05
        self.communication_range = 250
        self.communication_partners = []
        self.data_collected = 0
        self.task_queue = []
        self.current_task = None
        self.performance_history = []
        self.skill_set = self.generate_skill_set()
        self.learning_rate = 0.1
        self.charging = False
        self.charging_rate = 0.5
        self.emergency_mode = False
        self.ai_model = self.initialize_ai_model()
        
    def generate_skill_set(self):
        skills = ["navigation", "data_collection", "mapping", "obstacle_avoidance"]
        # Each agent has different skill levels
        return {skill: random.uniform(0.5, 1.0) for skill in skills}
    
    def initialize_ai_model(self):
        # Simulate a simple AI model that improves over time
        return {
            "obstacle_prediction_accuracy": 0.7,
            "path_optimization": 0.6,
            "battery_management": 0.8,
            "learning_rate": 0.01
        }
    
    def improve_ai_model(self, success):
        # Improve AI model based on experience
        improvement = self.ai_model["learning_rate"] * (1 if success else -0.5)
        self.ai_model["obstacle_prediction_accuracy"] = max(0.5, min(0.95, self.ai_model["obstacle_prediction_accuracy"] + improvement))
        self.ai_model["path_optimization"] = max(0.5, min(0.95, self.ai_model["path_optimization"] + improvement))
    
    def set_algorithm(self, agent_type):
        if agent_type == "Basic AI":
            self.algorithm = self.basic_navigation
        elif agent_type == "Advanced AI":
            self.algorithm = self.advanced_navigation
        elif agent_type == "Smart Navigator":
            self.algorithm = self.smart_navigation
        elif agent_type == "Adaptive AI":
            self.algorithm = self.adaptive_navigation
        elif agent_type == "ML Optimizer":
            self.algorithm = self.ml_optimized_navigation
        elif agent_type == "Swarm Intelligence":
            self.algorithm = self.swarm_intelligence_navigation
    
    def update_sensors(self, environment):
        self.sensors = []
        self.collision_points = []
        
        for angle in self.sensor_angles:
            sensor_angle = self.direction + angle
            end_x = self.x + math.cos(sensor_angle) * self.max_sensor_range
            end_y = self.y + math.sin(sensor_angle) * self.max_sensor_range
            
            closest_collision = None
            min_dist = self.max_sensor_range

            # Check collision with walls
            for wall in environment.walls:
                collision = self.line_rect_intersection(self.x, self.y, end_x, end_y, pygame.Rect(wall))
                if collision:
                    dist = math.sqrt((collision[0] - self.x)**2 + (collision[1] - self.y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_collision = collision
            
            # Check collision with obstacles
            for obstacle in environment.obstacles:
                collision = self.line_rect_intersection(self.x, self.y, end_x, end_y, obstacle['rect'])
                if collision:
                    dist = math.sqrt((collision[0] - self.x)**2 + (collision[1] - self.y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_collision = collision
            
            if closest_collision:
                self.sensors.append((self.x, self.y, closest_collision[0], closest_collision[1]))
                self.collision_points.append(closest_collision)
            else:
                self.sensors.append((self.x, self.y, end_x, end_y))
        
        # Cluster detection points for obstacle avoidance
        if self.collision_points:
            points = np.array(self.collision_points)
            if len(points) > 1:
                # Normalize points for clustering
                scaler = StandardScaler()
                points_scaled = scaler.fit_transform(points)
                
                # Use DBSCAN to cluster obstacles
                clustering = DBSCAN(eps=0.6, min_samples=2).fit(points_scaled)
                self.clusters = []
                for label in set(clustering.labels_):
                    if label != -1:  # Ignore noise
                        cluster_points = points[clustering.labels_ == label]
                        self.clusters.append(cluster_points)
    
    def line_rect_intersection(self, x1, y1, x2, y2, rect):
        # Check if line intersects with rectangle
        lines = [
            (rect.left, rect.top, rect.right, rect.top),     # top
            (rect.right, rect.top, rect.right, rect.bottom), # right
            (rect.right, rect.bottom, rect.left, rect.bottom), # bottom
            (rect.left, rect.bottom, rect.left, rect.top)    # left
        ]
        
        intersections = []
        for line in lines:
            intersection = self.line_line_intersection(x1, y1, x2, y2, line[0], line[1], line[2], line[3])
            if intersection:
                intersections.append(intersection)
                
        if not intersections:
            return None
            
        # Find the closest intersection
        closest = None
        min_dist = float('inf')
        for intersection in intersections:
            dist = math.sqrt((x1 - intersection[0])**2 + (y1 - intersection[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest = intersection
                
        return closest
    
    def line_line_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        # Calculate intersection of two lines
        den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if den == 0:
            return None
            
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
        if ua < 0 or ua > 1:
            return None
            
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den
        if ub < 0 or ub > 1:
            return None
            
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        
        return (x, y)
    
    def plan_path(self, target_x, target_y, environment):
        # Path planning with obstacle avoidance
        self.path = []
        
        # Add current position as first point
        self.path.append((self.x, self.y))
        self.target_x, self.target_y = target_x, target_y
        
        # If there's a clear path to target, go directly
        if not self.check_collision(self.x, self.y, target_x, target_y, environment):
            self.path.append((target_x, target_y))
            self.current_target = 1
            self.status = "Navigating"
            return
        
        # Otherwise, find a path around obstacles using waypoints
        waypoints = self.find_waypoints(target_x, target_y, environment)
        self.path.extend(waypoints)
        self.path.append((target_x, target_y))
        self.current_target = 1
        self.status = "Navigating"
    
    def find_waypoints(self, target_x, target_y, environment):
        # Find safe waypoints around obstacles using AI-improved pathfinding
        waypoints = []
        mid_x, mid_y = (self.x + target_x) / 2, (self.y + target_y) / 2
        
        # Use AI model to determine the number of waypoints to try
        num_waypoints = int(20 * self.ai_model["path_optimization"])
        
        # Try to find safe intermediate points
        for angle in np.linspace(0, 2*math.pi, num_waypoints):
            # Adjust search radius based on AI model
            search_radius = 150 * self.ai_model["obstacle_prediction_accuracy"]
            test_x = mid_x + math.cos(angle) * search_radius
            test_y = mid_y + math.sin(angle) * search_radius
            
            if (120 < test_x < 1180 and 120 < test_y < 780 and 
                not self.check_collision(self.x, self.y, test_x, test_y, environment) and
                not self.check_collision(test_x, test_y, target_x, target_y, environment)):
                
                waypoints.append((test_x, test_y))
                # Improve AI model based on successful path finding
                self.improve_ai_model(True)
                break
        
        return waypoints
    
    def check_collision(self, x1, y1, x2, y2, environment):
        # Check if path between two points collides with any obstacle
        for obstacle in environment.obstacles:
            if self.line_rect_intersection(x1, y1, x2, y2, obstacle['rect']):
                return True
        
        # Check walls (but not doorways)
        for wall in environment.walls:
            if self.line_rect_intersection(x1, y1, x2, y2, pygame.Rect(wall)):
                # Check if this is actually a doorway
                is_doorway = False
                for doorway in environment.doorways:
                    if (wall[0] == doorway[0] and wall[1] == doorway[1] and 
                        wall[2] == doorway[2] and wall[3] == doorway[3]):
                        is_doorway = True
                        break
                
                if not is_doorway:
                    return True
                
        return False
    
    def move_toward_target(self):
        if not self.path or self.current_target >= len(self.path):
            self.status = "Idle"
            return
            
        target_x, target_y = self.path[self.current_target]
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < 10:  # Reached target
            self.current_target += 1
            if self.current_target >= len(self.path):
                self.completed_paths += 1
                self.status = "Idle"
                # Collect data at target
                self.data_collected += random.randint(1, 5) * self.skill_set["data_collection"]
            return
            
        # Normalize direction
        if distance > 0:
            dx /= distance
            dy /= distance
            
        # Update direction
        self.direction = math.atan2(dy, dx)
        
        # Move toward target with speed adjusted by AI model
        ai_adjusted_speed = self.speed * self.ai_model["path_optimization"]
        self.x += dx * ai_adjusted_speed
        self.y += dy * ai_adjusted_speed
        
        # Update battery with AI-optimized drain rate
        battery_drain = self.battery_drain_rate * (2 - self.ai_model["battery_management"])
        self.battery -= battery_drain
        if self.battery < 15:
            self.emergency_mode = True
        if self.battery < 0:
            self.battery = 0
            self.status = "Battery Dead"
        
        # Update distance traveled
        dist_moved = math.sqrt((self.x - self.last_position[0])**2 + 
                               (self.y - self.last_position[1])**2)
        self.total_distance += dist_moved
        self.last_position = (self.x, self.y)
        
        # Add to memory
        self.memory.append((self.x, self.y))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def charge_battery(self, environment):
        # Check if at a charging station
        for station in environment.charging_stations:
            dist = math.sqrt((self.x - station['pos'][0])**2 + (self.y - station['pos'][1])**2)
            if dist < 25 and not station['in_use']:
                self.charging = True
                station['in_use'] = True
                self.battery += self.charging_rate
                if self.battery >= 100:
                    self.battery = 100
                    self.charging = False
                    station['in_use'] = False
                    self.emergency_mode = False
                return True
        return False
    
    def avoid_obstacles(self, environment):
        if not self.collision_points:
            return
            
        # Calculate repulsion force from obstacles with AI-predicted avoidance
        repulsion_x, repulsion_y = 0, 0
        
        for point in self.collision_points:
            dx = self.x - point[0]
            dy = self.y - point[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < 1:  # Avoid division by zero
                distance = 1
                
            # Strength of repulsion is inverse to distance squared, adjusted by AI
            strength = 1.0 / (distance * distance) * self.ai_model["obstacle_prediction_accuracy"]
            repulsion_x += dx * strength
            repulsion_y += dy * strength
            
        # Normalize repulsion vector
        repulsion_mag = math.sqrt(repulsion_x**2 + repulsion_y**2)
        if repulsion_mag > 0:
            repulsion_x /= repulsion_mag
            repulsion_y /= repulsion_mag
            
            # Adjust direction based on repulsion
            self.direction = math.atan2(repulsion_y, repulsion_x)
            
            # Move away from obstacles
            avoidance_strength = 0.5 * self.ai_model["obstacle_prediction_accuracy"]
            self.x += repulsion_x * self.speed * avoidance_strength
            self.y += repulsion_y * self.speed * avoidance_strength
    
    def basic_navigation(self, environment):
        # Basic navigation: just follow the path
        self.move_toward_target()
    
    def advanced_navigation(self, environment):
        # Advanced navigation: follow path with obstacle avoidance
        self.move_toward_target()
        self.avoid_obstacles(environment)
    
    def smart_navigation(self, environment):
        # Smart navigation: adaptive path following with obstacle prediction
        self.move_toward_target()
        
        # Predict obstacle movement and adjust
        if self.collision_points and len(self.path) > self.current_target:
            # Check if we need to replan path due to moving obstacles
            target_x, target_y = self.path[self.current_target]
            if self.check_collision(self.x, self.y, target_x, target_y, environment):
                self.plan_path(self.target_x, self.target_y, environment)
        
        self.avoid_obstacles(environment)
    
    def adaptive_navigation(self, environment):
        # Adaptive navigation: machine learning-inspired approach
        self.move_toward_target()
        
        # Use cluster information for better obstacle avoidance
        if self.clusters:
            # Calculate center of mass of obstacles
            cluster_points = np.vstack(self.clusters)
            center_x, center_y = np.mean(cluster_points, axis=0)
            
            # Move away from obstacle clusters
            dx = self.x - center_x
            dy = self.y - center_y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance > 0:
                dx /= distance
                dy /= distance
                
                # Adjust path based on cluster position
                self.x += dx * self.speed * 0.3
                self.y += dy * self.speed * 0.3
        
        self.avoid_obstacles(environment)
    
    def ml_optimized_navigation(self, environment):
        # ML-optimized navigation with continuous learning
        if self.emergency_mode and not self.charging:
            # Find nearest charging station
            nearest_station = None
            min_dist = float('inf')
            for station in environment.charging_stations:
                if not station['in_use']:
                    dist = math.sqrt((self.x - station['pos'][0])**2 + (self.y - station['pos'][1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_station = station
            
            if nearest_station:
                self.plan_path(nearest_station['pos'][0], nearest_station['pos'][1], environment)
        
        self.move_toward_target()
        
        # Use AI model for improved obstacle avoidance
        if self.collision_points and len(self.path) > self.current_target:
            # Use AI to predict if we need to replan
            replan_probability = 1.0 - self.ai_model["obstacle_prediction_accuracy"]
            if random.random() < replan_probability:
                self.plan_path(self.target_x, self.target_y, environment)
        
        self.avoid_obstacles(environment)
        
        # Try to charge if battery is low
        if self.battery < 30 and not self.charging:
            self.charge_battery(environment)
    
    def swarm_intelligence_navigation(self, environment, agents):
        # Swarm intelligence approach that coordinates with other agents
        self.move_toward_target()
        
        # Share information with other agents
        self.update_communication(agents)
        
        # If other agents have found better paths, consider using them
        for partner in self.communication_partners:
            if partner.path and len(partner.path) > 1:
                # Check if partner's path is better
                partner_path_length = sum(math.sqrt((partner.path[i][0]-partner.path[i-1][0])**2 + 
                                          (partner.path[i][1]-partner.path[i-1][1])**2) 
                                 for i in range(1, len(partner.path)))
                
                our_path_length = sum(math.sqrt((self.path[i][0]-self.path[i-1][0])**2 + 
                                       (self.path[i][1]-self.path[i-1][1])**2) 
                              for i in range(1, len(self.path))) if self.path else float('inf')
                
                if partner_path_length < our_path_length * 0.8:  # Partner's path is at least 20% better
                    # Adopt part of partner's path
                    self.path = partner.path.copy()
                    self.current_target = min(self.current_target, len(self.path)-1)
                    break
        
        self.avoid_obstacles(environment)
    
    def update_communication(self, agents):
        # Find agents within communication range
        self.communication_partners = []
        for agent in agents:
            if agent != self:
                dist = math.sqrt((self.x - agent.x)**2 + (self.y - agent.y)**2)
                if dist < self.communication_range:
                    self.communication_partners.append(agent)
                    # Share data with nearby agents
                    if random.random() < 0.1:  # Occasional data transfer
                        transfer_amount = random.randint(1, 3)
                        if agent.data_collected >= transfer_amount:
                            agent.data_collected -= transfer_amount
                            self.data_collected += transfer_amount
                    # Share AI model improvements
                    if random.random() < 0.05:
                        # Learn from partner's AI model
                        for key in self.ai_model:
                            if key != "learning_rate":
                                self.ai_model[key] = (self.ai_model[key] + agent.ai_model[key]) / 2
    
    def add_task(self, task_type, target_x, target_y, priority=1):
        self.task_queue.append({
            'type': task_type,
            'target_x': target_x,
            'target_y': target_y,
            'priority': priority,
            'status': 'Queued'
        })
        # Sort tasks by priority
        self.task_queue.sort(key=lambda x: x['priority'], reverse=True)
    
    def process_tasks(self):
        if not self.current_task and self.task_queue:
            self.current_task = self.task_queue.pop(0)
            self.current_task['status'] = 'In Progress'
            self.plan_path(self.current_task['target_x'], self.current_task['target_y'], environment)
        
        if self.current_task and self.status == "Idle":
            self.current_task['status'] = 'Completed'
            self.current_task = None
    
    def update(self, environment, agents):
        if self.charging:
            self.charge_battery(environment)
            self.status = "Charging"
        else:
            self.update_sensors(environment)
            
            # Select appropriate algorithm based on agent type
            if self.type == "Swarm Intelligence":
                self.swarm_intelligence_navigation(environment, agents)
            else:
                self.algorithm(environment)
            
            environment.update_heatmap(self.x, self.y)
            self.update_communication(agents)
            self.process_tasks()
        
        # Record performance
        if len(self.performance_history) < 100:
            efficiency = self.total_distance / (self.completed_paths + 1) if self.completed_paths > 0 else 0
            self.performance_history.append({
                'time': time.time(),
                'efficiency': efficiency,
                'battery': self.battery,
                'data_collected': self.data_collected,
                'ai_obstacle_prediction': self.ai_model["obstacle_prediction_accuracy"],
                'ai_path_optimization': self.ai_model["path_optimization"]
            })
    
    def draw(self, screen):
        # Draw sensors
        for sensor in self.sensors:
            pygame.draw.line(screen, SENSOR_COLOR, (sensor[0], sensor[1]), (sensor[2], sensor[3]), 2)
        
        # Draw path
        for i in range(1, len(self.path)):
            pygame.draw.line(screen, self.path_color, self.path[i-1], self.path[i], 3)
        
        # Draw path points
        for point in self.path:
            pygame.draw.circle(screen, self.path_color, (int(point[0]), int(point[1])), 5)
        
        # Draw collision points
        for point in self.collision_points:
            pygame.draw.circle(screen, (231, 76, 60), (int(point[0]), int(point[1])), 4)
        
        # Draw clusters
        for cluster in self.clusters:
            if len(cluster) > 0:
                center = np.mean(cluster, axis=0)
                pygame.draw.circle(screen, (155, 89, 182), (int(center[0]), int(center[1])), 8, 2)
        
        # Draw memory trail
        for i in range(1, len(self.memory)):
            alpha = int(255 * i / len(self.memory))
            color = (self.color[0], self.color[1], self.color[2], alpha)
            pygame.draw.line(screen, color, self.memory[i-1], self.memory[i], 2)
        
        # Draw communication range
        pygame.draw.circle(screen, (200, 200, 200, 50), (int(self.x), int(self.y)), self.communication_range, 1)
        
        # Draw communication links
        for partner in self.communication_partners:
            pygame.draw.line(screen, (0, 255, 0, 100), (self.x, self.y), (partner.x, partner.y), 2)
        
        # Draw agent
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, AGENT_OUTLINE, (int(self.x), int(self.y)), self.radius, 3)
        
        # Draw direction indicator
        end_x = self.x + math.cos(self.direction) * self.radius
        end_y = self.y + math.sin(self.direction) * self.radius
        pygame.draw.line(screen, AGENT_OUTLINE, (self.x, self.y), (end_x, end_y), 3)
        
        # Draw agent type label
        label = small_font.render(self.type, True, TEXT_COLOR)
        screen.blit(label, (self.x - label.get_width()//2, self.y + self.radius + 5))
        
        # Draw battery indicator
        battery_width = 30
        battery_height = 10
        pygame.draw.rect(screen, (200, 200, 200), (self.x - battery_width//2, self.y - 30, battery_width, battery_height), 1)
        battery_color = SUCCESS_COLOR if self.battery > 30 else WARNING_COLOR if self.battery > 10 else ERROR_COLOR
        pygame.draw.rect(screen, battery_color, (self.x - battery_width//2, self.y - 30, int(battery_width * self.battery/100), battery_height))
        
        # Draw emergency mode indicator
        if self.emergency_mode:
            pygame.draw.circle(screen, ERROR_COLOR, (int(self.x), int(self.y - 45)), 5)

class Button:
    def __init__(self, x, y, width, height, text, icon=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.icon = icon
        self.hovered = False
        self.active = False
        
    def draw(self, screen):
        if self.active:
            color = (46, 204, 113)  # Green for active state
        else:
            color = BUTTON_HOVER_COLOR if self.hovered else BUTTON_COLOR
            
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, (30, 30, 30), self.rect, 2, border_radius=5)
        
        text_surface = font.render(self.text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
    def check_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)
        return self.hovered
        
    def check_click(self, pos, event):
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(pos):
                self.active = not self.active
                return True
        return False

class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, width, height)
        self.handle_rect = pygame.Rect(x, y, 20, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.dragging = False
        self.label = label
        self.update_handle_pos()
        
    def update_handle_pos(self):
        normalized = (self.value - self.min_val) / (self.max_val - self.min_val)
        self.handle_rect.centerx = self.rect.left + normalized * self.rect.width
        
    def draw(self, screen):
        # Draw slider track
        pygame.draw.rect(screen, SLIDER_COLOR, self.rect, border_radius=3)
        
        # Draw handle
        pygame.draw.rect(screen, SLIDER_HANDLE_COLOR, self.handle_rect, border_radius=5)
        
        # Draw label and value
        label_text = small_font.render(f"{self.label}: {self.value:.1f}", True, TEXT_COLOR)
        screen.blit(label_text, (self.rect.x, self.rect.y - 20))
        
    def check_click(self, pos, event):
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            if self.handle_rect.collidepoint(pos):
                self.dragging = True
                return True
        elif event.type == MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        return False
    
    def update(self, pos):
        if self.dragging:
            self.handle_rect.centerx = max(self.rect.left, min(pos[0], self.rect.right))
            normalized = (self.handle_rect.centerx - self.rect.left) / self.rect.width
            self.value = self.min_val + normalized * (self.max_val - self.min_val)
            return True
        return False

class Menu:
    def __init__(self):
        self.options = ["Home", "Simulation", "Analysis", "AI Models", "Settings", "Help"]
        self.selected = "Home"
        self.rects = []
        
        # Calculate button positions
        width = 150
        spacing = 10
        total_width = len(self.options) * width + (len(self.options) - 1) * spacing
        start_x = (WIDTH - total_width) // 2
        
        for i, option in enumerate(self.options):
            self.rects.append(pygame.Rect(start_x + i*(width + spacing), 10, width, 40))
    
    def draw(self, screen):
        # Draw menu background
        pygame.draw.rect(screen, MENU_COLOR, (0, 0, WIDTH, 60))
        
        # Draw menu options
        for i, option in enumerate(self.options):
            color = MENU_HIGHLIGHT if option == self.selected else (100, 100, 120)
            pygame.draw.rect(screen, color, self.rects[i], border_radius=5)
            
            text = font.render(option, True, (255, 255, 255))
            text_rect = text.get_rect(center=self.rects[i].center)
            screen.blit(text, text_rect)
    
    def check_click(self, pos):
        for i, rect in enumerate(self.rects):
            if rect.collidepoint(pos):
                self.selected = self.options[i]
                return self.options[i]
        return None

class MissionPlanner:
    def __init__(self):
        self.missions = []
        self.current_mission = None
        self.mission_time = 0
        self.mission_log = []
        
    def log_event(self, event):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.mission_log.append(f"{timestamp} - {event}")
        if len(self.mission_log) > 20:
            self.mission_log.pop(0)
    
    def create_mission(self, name, objectives, priority=1):
        mission = {
            'name': name,
            'objectives': objectives,
            'priority': priority,
            'status': 'Pending',
            'progress': 0,
            'start_time': None,
            'end_time': None
        }
        self.missions.append(mission)
        # Sort missions by priority
        self.missions.sort(key=lambda x: x['priority'], reverse=True)
        self.log_event(f"New mission created: {name}")
    
    def start_mission(self, mission_index):
        if 0 <= mission_index < len(self.missions):
            self.current_mission = self.missions[mission_index]
            self.current_mission['status'] = 'In Progress'
            self.current_mission['start_time'] = time.time()
            self.log_event(f"Mission started: {self.current_mission['name']}")
    
    def update_mission(self, agents):
        if self.current_mission and self.current_mission['status'] == 'In Progress':
            # Calculate progress based on agent objectives
            completed = 0
            for objective in self.current_mission['objectives']:
                if objective['status'] == 'Completed':
                    completed += 1
            
            self.current_mission['progress'] = (completed / len(self.current_mission['objectives'])) * 100
            
            if self.current_mission['progress'] >= 100:
                self.current_mission['status'] = 'Completed'
                self.current_mission['end_time'] = time.time()
                mission_time = self.current_mission['end_time'] - self.current_mission['start_time']
                self.log_event(f"Mission completed: {self.current_mission['name']} in {mission_time:.1f}s")
                self.current_mission = None
    
    def assign_objectives(self, agents):
        if self.current_mission:
            for objective in self.current_mission['objectives']:
                if objective['status'] == 'Pending':
                    # Find the best agent for this objective based on skills and proximity
                    best_agent = None
                    best_score = -1
                    
                    for agent in agents:
                        if agent.status != "Battery Dead" and not agent.charging:
                            # Score based on distance, battery, and skills
                            dist = math.sqrt((agent.x - objective['x'])**2 + (agent.y - objective['y'])**2)
                            skill_match = agent.skill_set.get(objective.get('required_skill', 'navigation'), 0.5)
                            score = (100 - dist/10) + agent.battery/2 + skill_match * 50
                            
                            if score > best_score:
                                best_score = score
                                best_agent = agent
                    
                    if best_agent:
                        best_agent.add_task(objective['type'], objective['x'], objective['y'], self.current_mission['priority'])
                        objective['status'] = 'Assigned'
                        objective['assigned_to'] = best_agent.type
                        self.log_event(f"Objective assigned to {best_agent.type}")

class AITrainingCenter:
    def __init__(self):
        self.training_programs = {
            "Obstacle Prediction": {"level": 0.7, "cost": 10, "progress": 0},
            "Path Optimization": {"level": 0.6, "cost": 15, "progress": 0},
            "Battery Management": {"level": 0.8, "cost": 8, "progress": 0},
            "Communication Efficiency": {"level": 0.5, "cost": 12, "progress": 0}
        }
        self.research_points = 0
    
    def update(self, agents):
        # Generate research points based on agent performance
        for agent in agents:
            self.research_points += agent.data_collected * 0.1
            agent.data_collected = 0  # Convert data to research points
        
        # Update training progress
        for program in self.training_programs.values():
            if program["progress"] > 0:
                program["progress"] -= 1
                if program["progress"] <= 0:
                    program["level"] = min(0.95, program["level"] + 0.05)
    
    def start_training(self, program_name):
        program = self.training_programs.get(program_name)
        if program and self.research_points >= program["cost"]:
            self.research_points -= program["cost"]
            program["progress"] = 100  # 100 updates to complete training
            return True
        return False

def draw_homepage(screen):
    # Draw background with gradient
    for y in range(HEIGHT):
        color_value = 30 + (y / HEIGHT) * 20
        pygame.draw.line(screen, (color_value, color_value + 10, color_value + 20), (0, y), (WIDTH, y))
    
    # Draw title with shadow
    title = large_font.render("Ultra-Advanced AI Indoor Obstacle Avoidance System", True, (255, 255, 255))
    shadow = large_font.render("Ultra-Advanced AI Indoor Obstacle Avoidance System", True, (20, 20, 20))
    screen.blit(shadow, (WIDTH//2 - title.get_width()//2 + 2, 102))
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 100))
    
    # Draw subtitle
    subtitle = title_font.render("Next-Generation Autonomous Navigation with Machine Learning", True, (200, 200, 200))
    screen.blit(subtitle, (WIDTH//2 - subtitle.get_width()//2, 160))
    
    # Draw feature boxes with icons
    features = [
        ("Multi-Algorithm Navigation", "Compare different AI approaches for path planning", "🧠", AGENT_COLORS[0]),
        ("Real-time Obstacle Avoidance", "Dynamic response to moving obstacles", "⚠️", AGENT_COLORS[1]),
        ("Performance Analytics", "Comprehensive metrics and visualization", "📊", AGENT_COLORS[2]),
        ("Mission Planning", "Complex task assignment and coordination", "🎯", AGENT_COLORS[3]),
        ("Multi-Agent Communication", "Agents share data and coordinate actions", "📡", AGENT_COLORS[4]),
        ("Adaptive Environments", "Multiple environment types with unique challenges", "🏢", AGENT_COLORS[5])
    ]
    
    for i, (title, desc, icon, color) in enumerate(features):
        row = i // 3
        col = i % 3
        x = 150 + col * 400
        y = 220 + row * 180
        
        # Draw feature box
        pygame.draw.rect(screen, (50, 60, 70, 200), (x, y, 350, 150), border_radius=10)
        pygame.draw.rect(screen, color, (x, y, 350, 150), 3, border_radius=10)
        
        # Draw icon
        icon_text = title_font.render(icon, True, (255, 255, 255))
        screen.blit(icon_text, (x + 25, y + 20))
        
        # Draw feature title
        title_text = font.render(title, True, (255, 255, 255))
        screen.blit(title_text, (x + 60, y + 25))
        
        # Draw feature description
        desc_text = small_font.render(desc, True, (180, 180, 180))
        screen.blit(desc_text, (x + 20, y + 70))
    
    # Draw start button with animation
    button_color = (41, 128, 185) if (time.time() % 1) > 0.5 else (52, 152, 219)
    pygame.draw.rect(screen, button_color, (WIDTH//2 - 100, 600, 200, 60), border_radius=10)
    pygame.draw.rect(screen, (30, 30, 30), (WIDTH//2 - 100, 600, 200, 60), 3, border_radius=10)
    start_text = title_font.render("Start Simulation", True, (255, 255, 255))
    screen.blit(start_text, (WIDTH//2 - start_text.get_width()//2, 610))
    
    # Draw footer
    footer = small_font.render("SLASH MARK IT Solutions | Advanced AI Robotics Division", True, (150, 150, 150))
    screen.blit(footer, (WIDTH//2 - footer.get_width()//2, HEIGHT - 30))
    
    # Draw version info
    version = small_font.render("v3.0.0 | © 2023 SLASH MARK IT Solutions", True, (120, 120, 120))
    screen.blit(version, (WIDTH - version.get_width() - 20, HEIGHT - 30))

def draw_analysis_screen(screen, agents, environment, mission_planner):
    screen.fill((40, 44, 52))
    
    # Draw title
    title = title_font.render("Advanced Performance Analytics Dashboard", True, (255, 255, 255))
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 20))
    
    # Draw agent performance panels
    panel_width = (WIDTH - 60) // 2
    panel_height = (HEIGHT - 100) // 3
    
    for i, agent in enumerate(agents):
        row = i // 2
        col = i % 2
        x = 20 + col * (panel_width + 20)
        y = 70 + row * (panel_height + 20)
        
        # Draw panel
        pygame.draw.rect(screen, (50, 54, 62), (x, y, panel_width, panel_height), border_radius=10)
        pygame.draw.rect(screen, agent.color, (x, y, panel_width, panel_height), 2, border_radius=10)
        
        # Draw agent title
        title_text = font.render(f"{agent.type} Performance", True, (255, 255, 255))
        screen.blit(title_text, (x + 20, y + 15))
        
        # Draw performance metrics
        metrics = [
            f"Paths: {agent.completed_paths} | Dist: {agent.total_distance:.1f}",
            f"Efficiency: {agent.total_distance/(agent.completed_paths+1):.1f}",
            f"Battery: {agent.battery:.1f}% | Data: {agent.data_collected}",
            f"AI Obstacle: {agent.ai_model['obstacle_prediction_accuracy']:.2f}",
            f"AI Path: {agent.ai_model['path_optimization']:.2f}",
            f"Status: {agent.status}"
        ]
        
        for j, metric in enumerate(metrics):
            metric_text = small_font.render(metric, True, (200, 200, 200))
            screen.blit(metric_text, (x + 20, y + 40 + j * 20))
        
        # Draw mini performance chart
        if len(agent.performance_history) > 1:
            chart_width = panel_width - 40
            chart_height = 40
            chart_x = x + 20
            chart_y = y + panel_height - chart_height - 10
            
            # Draw chart background
            pygame.draw.rect(screen, (40, 40, 40), (chart_x, chart_y, chart_width, chart_height))
            
            # Draw chart data
            max_efficiency = max([p['efficiency'] for p in agent.performance_history]) if agent.performance_history else 1
            if max_efficiency > 0:
                for j in range(1, len(agent.performance_history)):
                    if j < chart_width:
                        x1 = chart_x + j - 1
                        y1 = chart_y + chart_height - (agent.performance_history[j-1]['efficiency'] / max_efficiency) * chart_height
                        x2 = chart_x + j
                        y2 = chart_y + chart_height - (agent.performance_history[j]['efficiency'] / max_efficiency) * chart_height
                        pygame.draw.line(screen, agent.color, (x1, y1), (x2, y2), 2)
    
    # Draw environment stats
    env_x = 20
    env_y = 70 + 3 * (panel_height + 20)
    pygame.draw.rect(screen, (50, 54, 62), (env_x, env_y, WIDTH - 40, 100), border_radius=10)
    pygame.draw.rect(screen, NEUTRAL_COLOR, (env_x, env_y, WIDTH - 40, 100), 2, border_radius=10)
    
    env_title = font.render("Environment Statistics", True, (255, 255, 255))
    screen.blit(env_title, (env_x + 20, env_y + 15))
    
    env_stats = [
        f"Mission Time: {environment.mission_time:.1f} seconds",
        f"Obstacles: {len(environment.obstacles)}",
        f"Active Agents: {sum(1 for a in agents if a.status != 'Battery Dead')}/{len(agents)}",
        f"Total Data: {sum(a.data_collected for a in agents)}",
        f"Environment: {environment.env_type}"
    ]
    
    for i, stat in enumerate(env_stats):
        stat_text = small_font.render(stat, True, (200, 200, 200))
        screen.blit(stat_text, (env_x + 20 + i * 300, env_y + 50))
    
    # Draw mission log
    log_x = 20
    log_y = env_y + 120
    pygame.draw.rect(screen, (50, 54, 62), (log_x, log_y, WIDTH - 40, 120), border_radius=10)
    pygame.draw.rect(screen, NEUTRAL_COLOR, (log_x, log_y, WIDTH - 40, 120), 2, border_radius=10)
    
    log_title = font.render("Mission Log", True, (255, 255, 255))
    screen.blit(log_title, (log_x + 20, log_y + 15))
    
    for i, log_entry in enumerate(mission_planner.mission_log[-5:]):
        log_text = console_font.render(log_entry, True, (200, 200, 200))
        screen.blit(log_text, (log_x + 20, log_y + 40 + i * 20))

def draw_ai_models_screen(screen, agents, ai_training):
    screen.fill((40, 44, 52))
    
    # Draw title
    title = title_font.render("AI Model Training Center", True, (255, 255, 255))
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 20))
    
    # Draw research points
    research_text = title_font.render(f"Research Points: {ai_training.research_points:.1f}", True, (241, 196, 15))
    screen.blit(research_text, (WIDTH - 250, 20))
    
    # Draw AI training programs
    program_width = (WIDTH - 60) // 2
    program_height = (HEIGHT - 100) // 4
    
    programs = list(ai_training.training_programs.items())
    for i, (program_name, program_data) in enumerate(programs):
        row = i // 2
        col = i % 2
        x = 20 + col * (program_width + 20)
        y = 70 + row * (program_height + 20)
        
        # Draw program panel
        pygame.draw.rect(screen, (50, 54, 62), (x, y, program_width, program_height), border_radius=10)
        pygame.draw.rect(screen, AGENT_COLORS[i], (x, y, program_width, program_height), 2, border_radius=10)
        
        # Draw program title
        title_text = font.render(program_name, True, (255, 255, 255))
        screen.blit(title_text, (x + 20, y + 15))
        
        # Draw program level
        level_text = small_font.render(f"Level: {program_data['level']:.2f}", True, (200, 200, 200))
        screen.blit(level_text, (x + 20, y + 40))
        
        # Draw program cost
        cost_text = small_font.render(f"Cost: {program_data['cost']} research points", True, (200, 200, 200))
        screen.blit(cost_text, (x + 20, y + 60))
        
        # Draw progress bar
        if program_data['progress'] > 0:
            pygame.draw.rect(screen, (40, 40, 40), (x + 20, y + 80, program_width - 40, 15))
            pygame.draw.rect(screen, SUCCESS_COLOR, (x + 20, y + 80, (program_width - 40) * program_data['progress'] / 100, 15))
            progress_text = small_font.render(f"Training: {program_data['progress']}%", True, (255, 255, 255))
            screen.blit(progress_text, (x + 20, y + 100))
        else:
            # Draw train button
            button_color = BUTTON_HOVER_COLOR if ai_training.research_points >= program_data['cost'] else ERROR_COLOR
            pygame.draw.rect(screen, button_color, (x + 20, y + 80, 100, 30), border_radius=5)
            train_text = small_font.render("Train", True, (255, 255, 255))
            screen.blit(train_text, (x + 70 - train_text.get_width()//2, y + 95 - train_text.get_height()//2))
    
    # Draw AI model benefits
    benefits_x = 20
    benefits_y = 70 + 2 * (program_height + 20)
    pygame.draw.rect(screen, (50, 54, 62), (benefits_x, benefits_y, WIDTH - 40, 150), border_radius=10)
    pygame.draw.rect(screen, NEUTRAL_COLOR, (benefits_x, benefits_y, WIDTH - 40, 150), 2, border_radius=10)
    
    benefits_title = font.render("AI Model Benefits", True, (255, 255, 255))
    screen.blit(benefits_title, (benefits_x + 20, benefits_y + 15))
    
    benefits = [
        "Obstacle Prediction: Better avoidance of dynamic obstacles",
        "Path Optimization: More efficient route planning",
        "Battery Management: Longer operational time",
        "Communication Efficiency: Better multi-agent coordination"
    ]
    
    for i, benefit in enumerate(benefits):
        benefit_text = small_font.render(benefit, True, (200, 200, 200))
        screen.blit(benefit_text, (benefits_x + 20, benefits_y + 45 + i * 25))

# Create environment and agents
environment = IndoorEnvironment(WIDTH, HEIGHT, "Office")
agents = [
    Agent(200, 200, AGENT_COLORS[0], PATH_COLORS[0], "Basic AI"),
    Agent(200, 300, AGENT_COLORS[1], PATH_COLORS[1], "Advanced AI"),
    Agent(200, 400, AGENT_COLORS[2], PATH_COLORS[2], "Smart Navigator"),
    Agent(200, 500, AGENT_COLORS[3], PATH_COLORS[3], "Adaptive AI"),
    Agent(200, 600, AGENT_COLORS[4], PATH_COLORS[4], "ML Optimizer"),
    Agent(200, 700, AGENT_COLORS[5], PATH_COLORS[5], "Swarm Intelligence")
]

# Set different speeds for different agents
agents[0].speed = 2.0
agents[1].speed = 2.5
agents[2].speed = 3.0
agents[3].speed = 3.5
agents[4].speed = 3.2
agents[5].speed = 2.8

# Create mission planner
mission_planner = MissionPlanner()

# Create AI training center
ai_training = AITrainingCenter()

# Create UI buttons
buttons = [
    Button(50, 100, 160, 40, "Set Target All"),
    Button(50, 150, 160, 40, "Add Obstacle"),
    Button(50, 200, 160, 40, "Clear All Paths"),
    Button(50, 250, 160, 40, "Reset Agents"),
    Button(50, 300, 160, 40, "Toggle Heatmap"),
    Button(50, 350, 160, 40, "Pause/Resume"),
    Button(50, 400, 160, 40, "New Mission"),
    Button(50, 450, 160, 40, "Change Env"),
    Button(50, 500, 160, 40, "Train AI")
]

# Create sliders for agent parameters
sliders = [
    Slider(50, 560, 160, 15, 1.0, 5.0, 2.0, "Agent 1 Speed"),
    Slider(50, 590, 160, 15, 1.0, 5.0, 2.5, "Agent 2 Speed"),
    Slider(50, 620, 160, 15, 1.0, 5.0, 3.0, "Agent 3 Speed"),
    Slider(50, 650, 160, 15, 1.0, 5.0, 3.5, "Agent 4 Speed"),
    Slider(50, 680, 160, 15, 1.0, 5.0, 3.2, "Agent 5 Speed"),
    Slider(50, 710, 160, 15, 1.0, 5.0, 2.8, "Agent 6 Speed")
]

# Create menu
menu = Menu()

# Main game loop
clock = pygame.time.Clock()
running = True
target_mode = False
show_heatmap = False
selected_agent = None
last_update_time = time.time()
update_interval = 0.5  # Update stats every 0.5 seconds
paused = False
current_screen = "Home"  # Can be "Home", "Simulation", "Analysis", "AI Models", "Settings", "Help"
env_types = list(ENVIRONMENT_TEMPLATES.keys())
current_env_index = 0

while running:
    current_time = time.time()
    mouse_pos = pygame.mouse.get_pos()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        # Handle menu clicks
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            clicked_option = menu.check_click(mouse_pos)
            if clicked_option:
                current_screen = clicked_option
                
        # Handle button clicks
        for button in buttons:
            if button.check_click(mouse_pos, event):
                if button.text == "Set Target All":
                    target_mode = True
                elif button.text == "Add Obstacle":
                    # Add a new obstacle at a random position
                    x = random.randint(120, 1160)
                    y = random.randint(120, 760)
                    size = random.randint(20, 50)
                    environment.obstacles.append({
                        'type': 'static',
                        'rect': pygame.Rect(x, y, size, size),
                        'color': OBSTACLE_COLOR
                    })
                elif button.text == "Clear All Paths":
                    for agent in agents:
                        agent.path = []
                elif button.text == "Reset Agents":
                    for i, agent in enumerate(agents):
                        agent.x, agent.y = 200, 200 + i*100
                        agent.path = []
                        agent.memory = []
                        agent.battery = 100
                        agent.status = "Idle"
                        agent.data_collected = 0
                        agent.emergency_mode = False
                        agent.charging = False
                elif button.text == "Toggle Heatmap":
                    show_heatmap = not show_heatmap
                elif button.text == "Pause/Resume":
                    paused = not paused
                elif button.text == "New Mission":
                    # Create a new random mission
                    objectives = []
                    for _ in range(random.randint(3, 6)):
                        objectives.append({
                            'type': 'data_collection',
                            'x': random.randint(120, 1160),
                            'y': random.randint(120, 760),
                            'status': 'Pending',
                            'required_skill': random.choice(['navigation', 'data_collection', 'mapping'])
                        })
                    mission_planner.create_mission(f"Mission {len(mission_planner.missions)+1}", objectives, random.randint(1, 3))
                elif button.text == "Change Env":
                    current_env_index = (current_env_index + 1) % len(env_types)
                    environment = IndoorEnvironment(WIDTH, HEIGHT, env_types[current_env_index])
                    for i, agent in enumerate(agents):
                        agent.x, agent.y = 200, 200 + i*100
                        agent.path = []
                        agent.memory = []
                elif button.text == "Train AI" and current_screen == "AI Models":
                    # Check if clicked on a training program
                    program_width = (WIDTH - 60) // 2
                    program_height = (HEIGHT - 100) // 4
                    
                    programs = list(ai_training.training_programs.items())
                    for i, (program_name, program_data) in enumerate(programs):
                        row = i // 2
                        col = i % 2
                        x = 20 + col * (program_width + 20)
                        y = 70 + row * (program_height + 20)
                        
                        if x + 20 <= mouse_pos[0] <= x + 120 and y + 80 <= mouse_pos[1] <= y + 110:
                            if ai_training.start_training(program_name):
                                mission_planner.log_event(f"Started training: {program_name}")
                    
        # Handle slider interactions
        for slider in sliders:
            slider.check_click(mouse_pos, event)
            
        # Set target on mouse click
        if event.type == pygame.MOUSEBUTTONDOWN and target_mode and current_screen == "Simulation":
            x, y = mouse_pos
            if 100 < x < 1180 and 100 < y < 780:  # Within indoor area
                for agent in agents:
                    agent.plan_path(x, y, environment)
            target_mode = False
            
        # Handle homepage start button
        if current_screen == "Home" and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if WIDTH//2 - 100 <= mouse_pos[0] <= WIDTH//2 + 100 and 600 <= mouse_pos[1] <= 660:
                current_screen = "Simulation"
            
        # Update slider values if dragging
        for i, slider in enumerate(sliders):
            if slider.dragging:
                slider.update(mouse_pos)
                agents[i].speed = slider.value
    
    # Update sliders
    for slider in sliders:
        if slider.dragging:
            slider.update(mouse_pos)
    
    # Draw the appropriate screen
    if current_screen == "Home":
        draw_homepage(screen)
    elif current_screen == "Analysis":
        draw_analysis_screen(screen, agents, environment, mission_planner)
    elif current_screen == "AI Models":
        draw_ai_models_screen(screen, agents, ai_training)
    else:
        # Draw simulation environment
        screen.fill(BACKGROUND)
        
        # Draw heatmap if enabled
        if show_heatmap:
            environment.draw_heatmap(screen)
        
        # Draw title and info
        title_text = title_font.render("Ultra-Advanced AI Indoor Obstacle Avoidance System", True, TEXT_COLOR)
        screen.blit(title_text, (WIDTH//2 - title_text.get_width()//2, 70))
        
        # Draw environment and agents
        environment.draw(screen)
        
        # Update environment and agents if not paused
        if not paused:
            environment.update_dynamic_obstacles()
            environment.update_mission_time()
            for agent in agents:
                agent.update(environment, agents)
            
            # Update missions
            mission_planner.update_mission(agents)
            if mission_planner.current_mission:
                mission_planner.assign_objectives(agents)
            elif mission_planner.missions and not mission_planner.current_mission:
                mission_planner.start_mission(0)
            
            # Update AI training
            ai_training.update(agents)
            
            # Apply AI training benefits to agents
            for agent in agents:
                for program_name, program_data in ai_training.training_programs.items():
                    if program_name == "Obstacle Prediction":
                        agent.ai_model["obstacle_prediction_accuracy"] = max(
                            agent.ai_model["obstacle_prediction_accuracy"],
                            program_data["level"]
                        )
                    elif program_name == "Path Optimization":
                        agent.ai_model["path_optimization"] = max(
                            agent.ai_model["path_optimization"],
                            program_data["level"]
                        )
                    elif program_name == "Battery Management":
                        agent.ai_model["battery_management"] = max(
                            agent.ai_model["battery_management"],
                            program_data["level"]
                        )
        
        for agent in agents:
            agent.draw(screen)
        
        # Draw UI panel
        pygame.draw.rect(screen, PANEL_COLOR, (10, 80, 240, 700), border_radius=10)
        pygame.draw.rect(screen, (180, 180, 190), (10, 80, 240, 700), 3, border_radius=10)
        
        # Draw buttons
        for button in buttons:
            button.check_hover(mouse_pos)
            button.draw(screen)
        
        # Draw sliders
        for slider in sliders:
            slider.draw(screen)
        
        # Draw mission info
        mission_y = 750
        mission_text = font.render("Missions:", True, TEXT_COLOR)
        screen.blit(mission_text, (20, mission_y))
        
        mission_y += 25
        if mission_planner.current_mission:
            mission = mission_planner.current_mission
            mission_name = small_font.render(f"{mission['name']}: {mission['progress']:.1f}%", True, TEXT_COLOR)
            screen.blit(mission_name, (30, mission_y))
        
        # Draw environment info
        env_text = small_font.render(f"Environment: {environment.env_type}", True, TEXT_COLOR)
        screen.blit(env_text, (20, 820))
        
        time_text = small_font.render(f"Mission Time: {environment.mission_time:.1f}s", True, TEXT_COLOR)
        screen.blit(time_text, (20, 840))
        
        # Draw research points
        research_text = small_font.render(f"Research: {ai_training.research_points:.1f}", True, TEXT_COLOR)
        screen.blit(research_text, (20, 860))
        
        # Draw help text
        help_text = small_font.render("Click 'Set Target All' then click anywhere to set navigation target", True, TEXT_COLOR)
        screen.blit(help_text, (WIDTH//2 - help_text.get_width()//2, HEIGHT - 30))
        
        if target_mode:
            help_text = font.render("Click anywhere to set target", True, (231, 76, 60))
            screen.blit(help_text, (WIDTH//2 - help_text.get_width()//2, HEIGHT - 60))
        
        if show_heatmap:
            heatmap_text = font.render("Heatmap: Red areas show frequent agent positions", True, (231, 76, 60))
            screen.blit(heatmap_text, (WIDTH//2 - heatmap_text.get_width()//2, HEIGHT - 90))
        
        if paused:
            pause_text = large_font.render("PAUSED", True, (231, 76, 60))
            screen.blit(pause_text, (WIDTH//2 - pause_text.get_width()//2, HEIGHT//2 - 50))
    
    # Draw menu
    menu.draw(screen)
    
    # Update display
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()