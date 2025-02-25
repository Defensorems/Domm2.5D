import pygame
import math
import random
import heapq

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

GREEN = (0, 150, 0)
DARK_GREEN = (0, 50, 0)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

BACKGROUND_COLOR = (30, 30, 30)
WALL_COLOR = (50, 50, 150)
EMPTY_COLOR = (50, 50, 50)
PLAYER_COLOR = (100, 150, 255)
ENEMY_COLOR = (120, 180, 120)

FACE_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255)
]

MINI_MAP_SIZE = 200
MINI_MAP_SCALE = 5
TILE_SIZE = 32
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
DRAW_RANGE = 10

player_x, player_y = 100, 100
player_health = 100
player_score = 0

# MAP_WIDTH, MAP_HEIGHT = 20, 20
GENERATION_BUFFER = 2
# MAP = [[0 for _ in range(MAP_WIDTH)] for _ in range(MAP_HEIGHT)]
# MAP[5][5] = 1

# MAP = [
#     [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
#     [0, 0, 0, 0, 1, 1, 0, 2, 0, 0],
#     [1, 1, 1, 0, 1, 1, 0, 1, 0, 0],
#     [1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
#     [1, 0, 1, 1, 0, 0, 2, 1, 0, 0],
#     [1, 0, 1, 0, 0, 1, 1, 1, 0, 0],
#     [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 0, 0, 1, 0, 0, 3, 0, 0],
#     [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
# ]

MAP = []
MAP_WIDTH = 0
MAP_HEIGHT = 0

TILE_SIZE = 64

player_x = TILE_SIZE + TILE_SIZE // 2
player_y = TILE_SIZE + TILE_SIZE // 2
player_angle = 0
FOV = math.pi / 3

NUM_RAYS = 120
MAX_DEPTH = 800

MINI_MAP_SCALE = 4
MINI_MAP_SIZE = 150
MINI_MAP_BORDER = 4

class Sphere:
    def __init__(self, x, y, z, direction_x, direction_y, speed=4, radius=5):
        self.x = x
        self.y = y
        self.z = z
        self.direction_x = direction_x
        self.direction_y = direction_y
        self.speed = speed
        self.radius = radius
        self.active = True

    def move(self):
        self.x += self.direction_x * self.speed
        self.y += self.direction_y * self.speed
    
        cell_x = int(self.x // TILE_SIZE)
        cell_y = int(self.y // TILE_SIZE)
        #print('comb', cell_x, cell_y)
        if 0 <= cell_x < MAP_WIDTH and 0 <= cell_y < MAP_HEIGHT:
            if MAP[cell_y][cell_x] == 1 or depth_buffer[int(self.x // TILE_SIZE)] < math.hypot(self.x - player_x, self.y - player_y):
                self.active = False
            else:
                self.active = True
        else:
            self.active = False

    def check_collision_with_player(self, player_x, player_y):
        distance = math.hypot(self.x - player_x, self.y - player_y)
        if distance < self.radius + TILE_SIZE // 4:
            self.active = False
            return True
        return False

class ExperienceSphere:
    def __init__(self, x, y, z, radius=5):
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.color = random.choice(FACE_COLORS)  # Используем случайный цвет из FACE_COLORS
        self.active = True
        
    def update(self):
        cell_x = int(self.x // TILE_SIZE)
        cell_y = int(self.y // TILE_SIZE)
        #print('exp', cell_x, cell_y)
        if 0 <= cell_x < MAP_WIDTH and 0 <= cell_y < MAP_HEIGHT:
            if MAP[cell_y][cell_x] == 1 or depth_buffer[int(self.x // TILE_SIZE)] < math.hypot(self.x - player_x, self.y - player_y):
                self.active = False
            else:
                self.active = True
        else:
            self.active = False
            
    def check_collision_with_player(self, player_x, player_y):
        distance = math.hypot(self.x - player_x, self.y - player_y)
        if distance < self.radius + TILE_SIZE // 4:
            self.active = False
            return True
        return False

class Enemy:
    def __init__(self, x, y, speed=1.5):
        self.x = x
        self.y = y
        self.speed = speed
        self.path = []
        self.fire_cooldown = 50
        self.fire_timer = 0

    def move_towards_player(self, player_x, player_y, map_grid):
        start = (int(self.x) // TILE_SIZE, int(self.y) // TILE_SIZE)
        goal = (int(player_x) // TILE_SIZE, int(player_y) // TILE_SIZE)
        
        if not self.path:
            self.path = astar(start, goal, map_grid)
        
        if self.path:
            next_tile = self.path[0]
            next_x = next_tile[0] * TILE_SIZE + TILE_SIZE // 2
            next_y = next_tile[1] * TILE_SIZE + TILE_SIZE // 2
            
            dx = next_x - self.x
            dy = next_y - self.y
            distance = math.hypot(dx, dy)
            
            if distance > 1:
                move_x = self.speed * dx / distance
                move_y = self.speed * dy / distance
                self.x += move_x
                self.y += move_y
            else:
                self.path.pop(0)

    def fire_at_player(self, player_x, player_y):
        if self.fire_timer == 0:
            dx = player_x - self.x
            dy = player_y - self.y
            distance = math.hypot(dx, dy)
            if distance > 0:
                direction_x = dx / distance
                direction_y = dy / distance
                sphere = Sphere(self.x, self.y, 0, direction_x, direction_y)
                spheres.append(sphere)
            self.fire_timer = self.fire_cooldown
        else:
            self.fire_timer -= 1

    def get_position(self):
        return self.x, self.y

    def check_collision(self, new_x, new_y, map_grid):
        cell_x = int(new_x // TILE_SIZE)
        cell_y = int(new_y // TILE_SIZE)
        if 0 <= cell_x < MAP_WIDTH and 0 <= cell_y < MAP_HEIGHT:
            return map_grid[cell_y][cell_x] == 1
        return True

pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

def show_start_screen():
    screen.fill(BLACK)
    font = pygame.font.SysFont('Arial', 60)
    title_text = font.render('Matrix Doom', True, (255, 255, 255))
    screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, SCREEN_HEIGHT // 3))

    font = pygame.font.SysFont('Arial', 30)
    start_text = font.render('Press any key to start', True, (255, 255, 255))
    screen.blit(start_text, (SCREEN_WIDTH // 2 - start_text.get_width() // 2, SCREEN_HEIGHT // 2))

    pygame.display.flip()
    wait_for_key()

def show_victory_screen():
    screen.fill(BLACK)
    font = pygame.font.SysFont('Arial', 60)
    victory_text = font.render('You Win!', True, (255, 255, 0))
    screen.blit(victory_text, (SCREEN_WIDTH // 2 - victory_text.get_width() // 2, SCREEN_HEIGHT // 3))

    font = pygame.font.SysFont('Arial', 30)
    restart_text = font.render('Press any key to restart', True, (255, 255, 255))
    screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2))

    pygame.display.flip()
    wait_for_key()
    
def show_gameover_screen():
    screen.fill(BLACK)
    font = pygame.font.SysFont('Arial', 60)
    victory_text = font.render('You are looser!', True, RED)
    screen.blit(victory_text, (SCREEN_WIDTH // 2 - victory_text.get_width() // 2, SCREEN_HEIGHT // 3))

    font = pygame.font.SysFont('Arial', 30)
    restart_text = font.render('Press any key to restart', True, (255, 255, 255))
    screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2))

    pygame.display.flip()
    wait_for_key()

def wait_for_key():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                return

def initialize_map():
    global MAP, MAP_HEIGHT, MAP_WIDTH, enemies, experience_spheres
    
    enemies = []
    experience_spheres = []
    
    MAP = generate_map_chunk(10, 10)  # Стартовый чанк 10x10
    
    MAP_WIDTH = len(MAP[0])
    MAP_HEIGHT = len(MAP)
    
    enemies = create_enemies(MAP)
    experience_spheres = create_experience_spheres(MAP)
    
    for _ in range(2):  # Генерируем чанки во всех направлениях
        expand_map('right')
        expand_map('left')
        expand_map('up')
        expand_map('down')

def generate_map_chunk(width, height):
    new_chunk = []
    for _ in range(height):
        row = []
        for _ in range(width):
            cell = random.choices(
                [0, 1, 2, 3], 
                weights=[59, 35, 3, 3],  # Вероятности: пусто, стена, сфера опыта, враг
                k=1
            )[0]
            row.append(cell)
        new_chunk.append(row)
    return new_chunk

def expand_map(direction):
    global MAP
    chunk_size = 10  # Размер чанка
    print(MAP_HEIGHT)
    if direction == 'right':
        new_chunk = generate_map_chunk(chunk_size, MAP_HEIGHT)
        for y in range(MAP_HEIGHT):
            MAP[y].extend(new_chunk[y])
            
    elif direction == 'left':
        new_chunk = generate_map_chunk(chunk_size, MAP_HEIGHT)
        for y in range(MAP_HEIGHT):
            MAP[y] = new_chunk[y] + MAP[y]
        # Корректируем позицию игрока
        global player_x
        player_x += chunk_size * TILE_SIZE
            
    elif direction == 'down':
        new_chunk = generate_map_chunk(MAP_WIDTH, chunk_size)
        MAP.extend(new_chunk)
            
    elif direction == 'up':
        new_chunk = generate_map_chunk(MAP_WIDTH, chunk_size)
        MAP = new_chunk + MAP
        # Корректируем позицию игрока
        global player_y
        player_y += chunk_size * TILE_SIZE
    
    update_objects_from_map()

def update_objects_from_map():
    global enemies, experience_spheres
    for y, row in enumerate(MAP):
        for x, cell in enumerate(row):
            if cell == 2:
                sphere_x = x * TILE_SIZE + TILE_SIZE // 2
                sphere_y = y * TILE_SIZE + TILE_SIZE // 2
                if not any(sphere.x == sphere_x and sphere.y == sphere_y for sphere in experience_spheres):
                    experience_spheres.append(ExperienceSphere(sphere_x, sphere_y, 0))
            elif cell == 3:
                enemy_x = x * TILE_SIZE + TILE_SIZE // 2
                enemy_y = y * TILE_SIZE + TILE_SIZE // 2
                if not any(enemy.x == enemy_x and enemy.y == enemy_y for enemy in enemies):
                    enemies.append(Enemy(enemy_x, enemy_y))

def check_player_position():
    global player_x, player_y, MAP_WIDTH, MAP_HEIGHT
    current_col = player_x // TILE_SIZE
    if current_col >= MAP_WIDTH - GENERATION_BUFFER:
        expand_map('right')
    elif current_col < GENERATION_BUFFER:
        expand_map('left')
    current_row = player_y // TILE_SIZE
    if current_row >= MAP_HEIGHT - GENERATION_BUFFER:
        expand_map('down')
    elif current_row < GENERATION_BUFFER:
        expand_map('up')
    update_map_dimensions()

def update_map_dimensions():
    global MAP_WIDTH, MAP_HEIGHT
    MAP_WIDTH = len(MAP[0])
    MAP_HEIGHT = len(MAP)

def is_within_distance(x, y, radius):
    dx = x - player_x
    dy = y - player_y
    return math.hypot(dx, dy) <= radius * TILE_SIZE

def handle_spheres():
    global player_health
    visible_spheres = [sphere for sphere in spheres if is_within_distance(sphere.x, sphere.y, DRAW_RANGE)]
    for sphere in visible_spheres:
        sphere.move()
        if sphere.active and sphere.check_collision_with_player(player_x, player_y):
            player_health -= 10  # Урон от сферы
            spheres.remove(sphere)
        elif not sphere.active:
            spheres.remove(sphere)
            
def handle_experience_spheres():
    #print('handle_experience_spheres')
    global player_score
    visible_spheres = [sphere for sphere in experience_spheres if is_within_distance(sphere.x, sphere.y, DRAW_RANGE)]
    for sphere in visible_spheres:
        sphere.update()
        if sphere.active and sphere.check_collision_with_player(player_x, player_y):
            player_score += 2  # Добавляем 2 очка
            experience_spheres.remove(sphere)
            
def create_experience_spheres(map_data):
    spheres = []
    for y, row in enumerate(map_data):
        for x, cell in enumerate(row):
            if cell == 2:
                sphere_x = x * TILE_SIZE + TILE_SIZE // 2
                sphere_y = y * TILE_SIZE + TILE_SIZE // 2
                spheres.append(ExperienceSphere(sphere_x, sphere_y, 0))
    return spheres

def create_enemies(map_data):
    enemies = []
    for y, row in enumerate(map_data):
        for x, cell in enumerate(row):
            if cell == 3:
                enemy_x = x * TILE_SIZE + TILE_SIZE // 2
                enemy_y = y * TILE_SIZE + TILE_SIZE // 2
                enemies.append(Enemy(enemy_x, enemy_y))
    return enemies

def astar(start, goal, map_grid):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current_g, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)

            # Проверка, находится ли соседняя ячейка внутри границ карты
            if (0 <= neighbor[0] < len(map_grid[0]) and 0 <= neighbor[1] < len(map_grid)):
                if map_grid[neighbor[1]][neighbor[0]] == 0:
                    tentative_g_score = current_g + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, neighbor))
            else:
                continue  # Пропустить соседние ячейки вне карты

    return []


def matrix_effect(surface, width, height):
    for _ in range(100):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        pygame.draw.line(surface, DARK_GREEN, (x, y), (x, y), 1)

def draw_wall_contour(x1, y1, x2, y2, height):
    color_intensity = max(50, min(150, 150 - height // 5))
    glow_color = (0, color_intensity, 0)
    pygame.draw.line(screen, glow_color, (x1, y1 - height // 2), (x1, y1 + height // 2), 2)
    pygame.draw.line(screen, glow_color, (x2, y2 - height // 2), (x2, y2 + height // 2), 2)
    pygame.draw.line(screen, glow_color, (x1, y1 - height // 2), (x2, y2 - height // 2), 2)
    pygame.draw.line(screen, glow_color, (x1, y1 + height // 2), (x2, y2 + height // 2), 2)
    matrix_effect(screen, x2 - x1, height)

def cast_rays():
    global depth_buffer
    depth_buffer = [float('inf')] * SCREEN_WIDTH
    start_angle = player_angle - FOV / 2
    for ray in range(NUM_RAYS):
        ray_angle = start_angle + ray * FOV / NUM_RAYS
        for depth in range(MAX_DEPTH):
            target_x = player_x + depth * math.cos(ray_angle)
            target_y = player_y + depth * math.sin(ray_angle)
            if 0 <= int(target_x // TILE_SIZE) < MAP_WIDTH and 0 <= int(target_y // TILE_SIZE) < MAP_HEIGHT:
                if MAP[int(target_y // TILE_SIZE)][int(target_x // TILE_SIZE)] == 1:
                    distance = depth * math.cos(ray_angle - player_angle)
                    wall_height = TILE_SIZE * SCREEN_HEIGHT / (distance + 0.0001)
                    x1 = ray * (SCREEN_WIDTH // NUM_RAYS)
                    x2 = (ray + 1) * (SCREEN_WIDTH // NUM_RAYS)
                    y1 = SCREEN_HEIGHT // 2
                    y2 = SCREEN_HEIGHT // 2
                    draw_wall_contour(x1, y1, x2, y2, int(wall_height))
                    depth_buffer[ray] = distance
                    break
            else:
                break

def average_without_inf(numbers):
    valid_numbers = [num for num in numbers if num != float('inf') and num != float('-inf')]
    if not valid_numbers:
        return 0  # Возвращаем 0 вместо None, если нет валидных чисел
    return sum(valid_numbers) / len(valid_numbers)

def move_player():
    global player_x, player_y, player_angle
    keys = pygame.key.get_pressed()
    move_speed = 3
    rot_speed = 0.03
    new_x = player_x
    new_y = player_y
    if keys[pygame.K_w]:
        new_x += move_speed * math.cos(player_angle)
        new_y += move_speed * math.sin(player_angle)
    if keys[pygame.K_s]:
        new_x -= move_speed * math.cos(player_angle)
        new_y -= move_speed * math.sin(player_angle)
    if not check_collision(new_x, new_y):
        player_x = new_x
        player_y = new_y

def check_collision(new_x, new_y):
    cell_x = int(new_x // TILE_SIZE)
    cell_y = int(new_y // TILE_SIZE)
    if 0 <= cell_x < MAP_WIDTH and 0 <= cell_y < MAP_HEIGHT:
        return MAP[cell_y][cell_x] == 1
    return True

def rotate_player_with_mouse():
    global player_angle
    mouse_x, _ = pygame.mouse.get_pos()
    center_x = SCREEN_WIDTH // 2
    mouse_delta_x = mouse_x - center_x
    mouse_sensitivity = 0.002
    player_angle += mouse_delta_x * mouse_sensitivity
    pygame.mouse.set_pos((center_x, SCREEN_HEIGHT // 2))

def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def project_to_screen(world_x, world_y, world_z, player_x, player_y, player_angle):
    dx = world_x - player_x
    dy = world_y - player_y
    distance = math.hypot(dx, dy)
    angle_to_point = math.atan2(dy, dx)
    relative_angle = normalize_angle(angle_to_point - player_angle)
    if abs(relative_angle) > FOV / 2:
        return None, None, None
    screen_x = SCREEN_WIDTH // 2 + (relative_angle / (FOV / 2)) * (SCREEN_WIDTH // 2)
    if distance > 0:
        screen_y = SCREEN_HEIGHT // 2 - (world_z / distance) * (SCREEN_HEIGHT // 2)
    else:
        screen_y = SCREEN_HEIGHT // 2
    return screen_x, screen_y, distance

def draw_3d_cube(surface, cube_x, cube_y, cube_z, cube_size, player_x, player_y, player_angle):
    half_size = cube_size / 2
    corners = {
        "front_bottom_left": (cube_x - half_size, cube_y - half_size, cube_z),
        "front_bottom_right": (cube_x + half_size, cube_y - half_size, cube_z),
        "front_top_left": (cube_x - half_size, cube_y - half_size, cube_z + cube_size),
        "front_top_right": (cube_x + half_size, cube_y - half_size, cube_z + cube_size),
        "back_bottom_left": (cube_x - half_size, cube_y + half_size, cube_z),
        "back_bottom_right": (cube_x + half_size, cube_y + half_size, cube_z),
        "back_top_left": (cube_x - half_size, cube_y + half_size, cube_z + cube_size),
        "back_top_right": (cube_x + half_size, cube_y + half_size, cube_z + cube_size),
    }
    faces = [
        ("front_bottom_left", "front_bottom_right", "front_top_right", "front_top_left"),
        ("back_bottom_left", "back_bottom_right", "back_top_right", "back_top_left"),
        ("front_bottom_left", "front_top_left", "back_top_left", "back_bottom_left"),
        ("front_bottom_right", "front_top_right", "back_top_right", "back_bottom_right"),
        ("front_top_left", "front_top_right", "back_top_right", "back_top_left"),
        ("front_bottom_left", "front_bottom_right", "back_bottom_right", "back_bottom_left"),
    ]
    projected_corners = {}
    for corner_name, (corner_x, corner_y, corner_z) in corners.items():
        screen_x, screen_y, distance = project_to_screen(corner_x, corner_y, corner_z, player_x, player_y, player_angle)
        if screen_x is not None and 0 <= int(screen_x) < SCREEN_WIDTH and depth_buffer[int(screen_x)] > distance:
            projected_corners[corner_name] = (screen_x, screen_y, distance)

    if len(projected_corners) < 8:
        return

    sorted_faces = sorted(
        enumerate(faces),
        key=lambda face: sum(projected_corners[corner][2] for corner in face[1]) / 4,
        reverse=True,
    )
    for face_index, face in sorted_faces:
        face_color = FACE_COLORS[face_index]
        screen_points = []
        face_visible = False
        average_depth = average_without_inf(depth_buffer)  # Получаем среднюю глубину

        for corner in face:
            if corner in projected_corners:
                screen_x, screen_y, distance = projected_corners[corner]
                screen_points.append((screen_x, screen_y))
                
                if 0 <= int(screen_x) < SCREEN_WIDTH and average_depth > distance:
                    face_visible = True

        if face_visible and len(screen_points) == 4:
            pygame.draw.polygon(surface, face_color, screen_points)
            
def draw_3d_sphere(surface, sphere_x, sphere_y, sphere_z, sphere_radius, player_x, player_y, player_angle, resolution=20):
    points = []
    for i in range(resolution + 1):
        theta = math.pi * i / resolution
        for j in range(resolution + 1):
            phi = 2 * math.pi * j / resolution
            x = sphere_x + sphere_radius * math.sin(theta) * math.cos(phi)
            y = sphere_y + sphere_radius * math.sin(theta) * math.sin(phi)
            z = sphere_z + sphere_radius * math.cos(theta)
            points.append((x, y, z))

    projected_points = []
    valid_indices = []  # Хранит индексы успешно спроецированных точек
    for index, (x, y, z) in enumerate(points):
        screen_x, screen_y, distance = project_to_screen(x, y, z, player_x, player_y, player_angle)
        if screen_x is not None and 0 <= int(screen_x) < SCREEN_WIDTH:
            # Проверяем, находится ли точка за стеной
            if depth_buffer[int(screen_x)] > distance:
                projected_points.append((screen_x, screen_y, distance))
                valid_indices.append(index)

    if len(projected_points) < len(points) * 0.8:
        return

    for i in range(len(points)):
        if i not in valid_indices:  # Пропуск точек, которые не прошли проекцию
            continue
        if (i % (resolution + 1)) != resolution and (i + 1) in valid_indices:
            pygame.draw.line(
                surface,
                (255, 255, 255),
                projected_points[valid_indices.index(i)][:2],
                projected_points[valid_indices.index(i + 1)][:2],
            )
        if (i + resolution + 1) in valid_indices:
            pygame.draw.line(
                surface,
                (255, 255, 255),
                projected_points[valid_indices.index(i)][:2],
                projected_points[valid_indices.index(i + resolution + 1)][:2],
            )

def draw_3d_tetrahedron(surface, tetra_x, tetra_y, tetra_z, size, player_x, player_y, player_angle):
    # Определение вершин тетраэдра
    half_size = size / 2
    height_multiplier = 1.5  # Коэффициент увеличения высоты
    vertices = [
        (tetra_x, tetra_y, tetra_z + size * height_multiplier),  # Вершина сверху
        (tetra_x - half_size, tetra_y - half_size, tetra_z),  # Нижний левый угол
        (tetra_x + half_size, tetra_y - half_size, tetra_z),  # Нижний правый угол
        (tetra_x, tetra_y + half_size, tetra_z),  # Нижний задний угол
    ]

    # Определение граней тетраэдра (индексы вершин)
    faces = [
        (0, 1, 2),  # Верхняя грань
        (0, 2, 3),  # Правая грань
        (0, 3, 1),  # Левая грань
        (1, 2, 3),  # Основание
    ]

    # Цвета для граней
    face_colors = [
        (255, 0, 0),  # Красный
        (0, 255, 0),  # Зелёный
        (0, 0, 255),  # Синий
        (255, 255, 0),  # Жёлтый
    ]

    # Проецируем вершины на экран и проверяем видимость
    projected_vertices = []
    visible_faces = []
    for vertex in vertices:
        screen_x, screen_y, distance = project_to_screen(*vertex, player_x, player_y, player_angle)
        if screen_x is not None and 0 <= int(screen_x) < SCREEN_WIDTH:
            # Сохраняем вершину только если она видима
            projected_vertices.append((screen_x, screen_y, distance))
        else:
            projected_vertices.append(None)

    # Если хотя бы одна вершина тетраэдра невидима, пропускаем его
    if all(v is None for v in projected_vertices):
        return

    # Сортировка граней по средней глубине для корректного наложения
    face_depths = []
    for face in faces:
        avg_depth = sum(
            projected_vertices[v][2] if projected_vertices[v] else float('inf')
            for v in face
        ) / 3
        face_depths.append((avg_depth, face))

    face_depths.sort(reverse=True, key=lambda item: item[0])

    # Отрисовка граней
    for depth, face in face_depths:
        points = []
        face_visible = True
        for v in face:
            vertex = projected_vertices[v]
            if vertex is None or depth_buffer[int(vertex[0])] <= vertex[2]:
                face_visible = False
                break
            points.append(vertex[:2])

        if face_visible:
            pygame.draw.polygon(surface, face_colors[faces.index(face)], points)

def draw_mini_map(screen):
    mini_map_x = SCREEN_WIDTH - MINI_MAP_SIZE - 20
    mini_map_y = 20

    pygame.draw.rect(screen, GREEN, (mini_map_x - MINI_MAP_BORDER, mini_map_y - MINI_MAP_BORDER,
                                     MINI_MAP_SIZE + 2 * MINI_MAP_BORDER, MINI_MAP_SIZE + 2 * MINI_MAP_BORDER), border_radius=15)
    pygame.draw.rect(screen, DARK_GREEN, (mini_map_x, mini_map_y, MINI_MAP_SIZE, MINI_MAP_SIZE), border_radius=15)

    mini_map_center_x = mini_map_x + MINI_MAP_SIZE // 2
    mini_map_center_y = mini_map_y + MINI_MAP_SIZE // 2

    visible_radius = MINI_MAP_SIZE // 2 // MINI_MAP_SCALE

    for row in range(MAP_HEIGHT):
        for col in range(MAP_WIDTH):
            cell_x = col * TILE_SIZE
            cell_y = row * TILE_SIZE

            dx = cell_x - player_x
            dy = cell_y - player_y

            # Проверяем, находятся ли координаты в пределах видимости
            if abs(dx) <= visible_radius * TILE_SIZE and abs(dy) <= visible_radius * TILE_SIZE:
                map_x = mini_map_center_x + dx // MINI_MAP_SCALE
                map_y = mini_map_center_y + dy // MINI_MAP_SCALE

                # Проверяем, чтобы координаты находились внутри миникарты
                if mini_map_x <= map_x < mini_map_x + MINI_MAP_SIZE and mini_map_y <= map_y < mini_map_y + MINI_MAP_SIZE:
                    color = GREEN if MAP[row][col] == 1 else DARK_GREEN
                    pygame.draw.rect(screen, color, 
                                     (map_x, map_y, TILE_SIZE // MINI_MAP_SCALE, TILE_SIZE // MINI_MAP_SCALE))

    # Отображение игрока
    player_direction_x = mini_map_center_x + 10 * math.cos(player_angle)
    player_direction_y = mini_map_center_y + 10 * math.sin(player_angle)
    pygame.draw.polygon(screen, (0, 255, 0), [
        (mini_map_center_x, mini_map_center_y),
        (mini_map_center_x + 8 * math.cos(player_angle - math.pi / 3), mini_map_center_y + 8 * math.sin(player_angle - math.pi / 3)),
        (mini_map_center_x + 8 * math.cos(player_angle + math.pi / 3), mini_map_center_y + 8 * math.sin(player_angle + math.pi / 3))
    ])

    # Отображение врагов
    for enemy in enemies:
        ex, ey = enemy.get_position()
        dx = ex - player_x
        dy = ey - player_y
        dist = math.hypot(dx, dy)

        # Проверяем, находится ли враг в пределах видимости миникарты
        if dist < visible_radius * TILE_SIZE:
            map_ex = mini_map_center_x + dx // MINI_MAP_SCALE
            map_ey = mini_map_center_y + dy // MINI_MAP_SCALE
            if mini_map_x <= map_ex < mini_map_x + MINI_MAP_SIZE and mini_map_y <= map_ey < mini_map_y + MINI_MAP_SIZE:
                pygame.draw.circle(screen, (255, 0, 0), (map_ex, map_ey), 4)

def draw_health_bar():
    health_bar_width = 200
    health_bar_height = 20
    health_percentage = player_health / 100
    pygame.draw.rect(screen, RED, (10, 10, health_bar_width, health_bar_height))
    pygame.draw.rect(screen, GREEN, (10, 10, health_bar_width * health_percentage, health_bar_height))

def draw_score():
    font = pygame.font.SysFont('Arial', 30)
    score_text = font.render(f'Score: {player_score}', True, (255, 255, 255))
    screen.blit(score_text, (10, 40))

clock = pygame.time.Clock()
running = True

spheres = []

# MAP_WIDTH = len(MAP[0])
# MAP_HEIGHT = len(MAP)

initialize_map()
update_map_dimensions()

show_start_screen()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BACKGROUND_COLOR)

    if player_score >= 20:
        show_victory_screen()
        player_score = 0 
        show_start_screen()
        
    if player_health <= 0:
        show_gameover_screen()
        player_score = 0 
        show_start_screen()

    move_player()
    rotate_player_with_mouse()
    cast_rays()
    draw_mini_map(screen)

    visible_enemies = [enemy for enemy in enemies if is_within_distance(enemy.x, enemy.y, DRAW_RANGE)]
    for enemy in visible_enemies:
        enemy.move_towards_player(player_x, player_y, MAP)
        enemy.fire_at_player(player_x, player_y)
        enemy_x, enemy_y = enemy.get_position()
        draw_3d_cube(screen, enemy_x, enemy_y, 0, 10, player_x, player_y, player_angle)
 
    handle_spheres()
    handle_experience_spheres()
    check_player_position()
    update_map_dimensions()

    for sphere in spheres:
        if sphere.active and is_within_distance(sphere.x, sphere.y, DRAW_RANGE):
            draw_3d_sphere(screen, sphere.x, sphere.y, sphere.z, sphere.radius, player_x, player_y, player_angle)

    for sphere in experience_spheres:
        if sphere.active and is_within_distance(sphere.x, sphere.y, DRAW_RANGE):
            draw_3d_sphere(screen, sphere.x, sphere.y, sphere.z, sphere.radius, player_x, player_y, player_angle)


    draw_score()
    draw_health_bar()
        
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
