import pygame
import numpy as np
import random
import math
import time
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RYGate
from qiskit.primitives import StatevectorSampler

# Initialize Pygame
#(C)2025/8/15 Tsubasa Kato - Inspire Search Corporation - Created using Perplexity Pro.
pygame.init()

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
CAR_WIDTH = 30
CAR_HEIGHT = 15
PYLON_SIZE = 20
SENSOR_RANGE = 150
LOG_INTERVAL = 2.0  # Log data every 2 seconds

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)
DARK_GREEN = (0, 100, 0)
PURPLE = (128, 0, 128)

class QuantumSpikingNeuron:
    """Quantum Spiking Neuron for car control"""
    def __init__(self, num_inputs=3, threshold=0.4):
        self.num_inputs = num_inputs
        self.threshold = threshold
        self.weights = ParameterVector('w', num_inputs)
        self.input_params = ParameterVector('x', num_inputs)
        self.theta_param = Parameter('theta')
        self.bias_param = Parameter('bias')
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        qreg = QuantumRegister(self.num_inputs + 1, 'q')
        creg = ClassicalRegister(1, 'spike')
        qc = QuantumCircuit(qreg, creg)

        # Input encoding
        for i in range(self.num_inputs):
            qc.ry(self.input_params[i] * np.pi / 2, qreg[i])

        # Bias rotation
        qc.ry(self.bias_param, qreg[self.num_inputs])

        # Controlled rotations
        for i in range(self.num_inputs):
            qc.append(RYGate(self.weights[i]).control(1), [qreg[i], qreg[self.num_inputs]])

        # Final rotation
        qc.ry(self.theta_param, qreg[self.num_inputs])
        qc.measure(qreg[self.num_inputs], creg[0])
        
        return qc

    def process(self, inputs, weights, theta=np.pi/4, bias=np.pi/8):
        sampler = StatevectorSampler()
        
        # Bind parameters
        params = {self.theta_param: theta, self.bias_param: bias}
        for i in range(self.num_inputs):
            params[self.input_params[i]] = float(inputs[i])
            params[self.weights[i]] = float(weights[i])

        bound = self.circuit.assign_parameters(params)
        
        # Execute
        job = sampler.run([bound], shots=256)
        pub_res = job.result()[0]
        counts = pub_res.data.spike.get_counts()
        
        # Calculate probability
        total_shots = sum(counts.values())
        p1 = counts.get("1", 0) / total_shots if total_shots > 0 else 0.0
        
        return 1 if p1 > self.threshold else 0, p1

class Car:
    """Car controlled by quantum spiking neural network with collision detection"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0  # Facing right
        self.speed = 0
        self.max_speed = 3
        self.acceleration = 0.1
        self.turn_speed = 2
        
        # Collision properties
        self.collision_radius = max(CAR_WIDTH, CAR_HEIGHT) // 2
        self.collision_cooldown = 0
        self.bounce_strength = 2.0
        self.is_colliding = False
        
        # Sensors (3 distance sensors: left, center, right)
        self.sensor_angles = [-30, 0, 30]  # Relative to car direction
        self.sensor_distances = [1.0, 1.0, 1.0]  # Initialize with safe values
        
        # Quantum brain
        self.brain_steering = QuantumSpikingNeuron(num_inputs=3, threshold=0.3)
        self.brain_speed = QuantumSpikingNeuron(num_inputs=3, threshold=0.5)
        
        # Neural network weights (tunable parameters)
        self.steering_weights = [0.8, -0.4, -0.8]  # left sensor positive, right negative
        self.speed_weights = [-0.6, -0.9, -0.6]    # all sensors negative (brake on obstacles)
        
        # Car history for trail
        self.trail = []
        self.trail_max_length = 50
        
        # Logging system
        self.last_log_time = time.time()
        self.log_counter = 0
        self.total_distance = 0
        self.last_position = (x, y)
        
        # Decision tracking for logging
        self.last_steering_spike = 0
        self.last_speed_spike = 0
        self.last_steering_prob = 0.0
        self.last_speed_prob = 0.0

    def get_car_rect(self):
        """Get collision rectangle for the car"""
        return pygame.Rect(
            self.x - CAR_WIDTH // 2,
            self.y - CAR_HEIGHT // 2,
            CAR_WIDTH,
            CAR_HEIGHT
        )

    def check_collision(self, obstacles):
        """Check collision with obstacles and handle bounce"""
        car_rect = self.get_car_rect()
        collision_detected = False
        
        # Check collision with obstacles
        for obs_x, obs_y in obstacles:
            obs_rect = pygame.Rect(
                obs_x - PYLON_SIZE // 2,
                obs_y - PYLON_SIZE // 2,
                PYLON_SIZE,
                PYLON_SIZE
            )
            
            if car_rect.colliderect(obs_rect):
                collision_detected = True
                
                # Calculate bounce direction
                dx = self.x - obs_x
                dy = self.y - obs_y
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance > 0:
                    # Normalize direction
                    dx /= distance
                    dy /= distance
                    
                    # Apply bounce
                    bounce_distance = self.bounce_strength * 10
                    self.x += dx * bounce_distance
                    self.y += dy * bounce_distance
                    
                    # Reverse and reduce speed
                    self.speed = -abs(self.speed) * 0.5
                    
                    # Adjust angle away from obstacle
                    obstacle_angle = math.degrees(math.atan2(dy, dx))
                    self.angle = obstacle_angle
                
                break
        
        # Check collision with screen boundaries
        if self.x <= CAR_WIDTH // 2:
            self.x = CAR_WIDTH // 2
            self.speed = -self.speed * 0.5
            collision_detected = True
        elif self.x >= SCREEN_WIDTH - CAR_WIDTH // 2:
            self.x = SCREEN_WIDTH - CAR_WIDTH // 2
            self.speed = -self.speed * 0.5
            collision_detected = True
            
        if self.y <= CAR_HEIGHT // 2:
            self.y = CAR_HEIGHT // 2
            self.speed = -self.speed * 0.5
            collision_detected = True
        elif self.y >= SCREEN_HEIGHT - CAR_HEIGHT // 2:
            self.y = SCREEN_HEIGHT - CAR_HEIGHT // 2
            self.speed = -self.speed * 0.5
            collision_detected = True
        
        # Set collision state
        if collision_detected:
            self.is_colliding = True
            self.collision_cooldown = 30  # 30 frames cooldown
        else:
            if self.collision_cooldown > 0:
                self.collision_cooldown -= 1
            else:
                self.is_colliding = False
        
        return collision_detected

    def cast_sensors(self, obstacles, screen_width, screen_height):
        """Cast distance sensors to detect obstacles"""
        new_sensor_distances = []
        
        for sensor_angle in self.sensor_angles:
            # Calculate sensor direction
            angle_rad = math.radians(self.angle + sensor_angle)
            
            # Cast ray from car position
            distance = 0
            step_size = 2
            max_distance = SENSOR_RANGE
            
            while distance < max_distance:
                # Calculate sensor point
                sensor_x = self.x + distance * math.cos(angle_rad)
                sensor_y = self.y + distance * math.sin(angle_rad)
                
                # Check boundaries
                if sensor_x < 0 or sensor_x >= screen_width or sensor_y < 0 or sensor_y >= screen_height:
                    break
                
                # Check collision with obstacles
                collision_found = False
                for obstacle in obstacles:
                    obs_rect = pygame.Rect(obstacle[0] - PYLON_SIZE//2, obstacle[1] - PYLON_SIZE//2, 
                                         PYLON_SIZE, PYLON_SIZE)
                    if obs_rect.collidepoint(sensor_x, sensor_y):
                        new_sensor_distances.append(distance / SENSOR_RANGE)  # Normalize
                        collision_found = True
                        break
                
                if collision_found:
                    break
                    
                distance += step_size
            else:
                new_sensor_distances.append(1.0)  # No obstacle detected
        
        # Only update if we have valid sensor data
        if len(new_sensor_distances) == 3:
            self.sensor_distances = new_sensor_distances
        # If sensor casting fails, keep previous values (initialized to safe defaults)

    def quantum_decision(self):
        """Use quantum spiking neural network to make decisions"""
        if len(self.sensor_distances) == 3:
            # Steering decision (left/right)
            steering_spike, steering_prob = self.brain_steering.process(
                self.sensor_distances, self.steering_weights
            )
            
            # Speed decision (accelerate/brake)
            speed_spike, speed_prob = self.brain_speed.process(
                self.sensor_distances, self.speed_weights
            )
            
            # Store for logging
            self.last_steering_spike = steering_spike
            self.last_speed_spike = speed_spike
            self.last_steering_prob = steering_prob
            self.last_speed_prob = speed_prob
            
            return steering_spike, speed_spike
        
        return 0, 1  # Default: no turn, accelerate

    def log_data(self):
        """Log sensor and car data to terminal"""
        current_time = time.time()
        if current_time - self.last_log_time >= LOG_INTERVAL:
            # Calculate distance traveled
            distance_step = math.sqrt((self.x - self.last_position[0])**2 + 
                                    (self.y - self.last_position[1])**2)
            self.total_distance += distance_step
            self.last_position = (self.x, self.y)
            
            self.log_counter += 1
            
            # Print detailed log
            print(f"\n{'='*80}")
            print(f"🧠 QUANTUM CAR LOG #{self.log_counter:03d} | Time: {current_time:.1f}s")
            print(f"{'='*80}")
            
            # Position and movement
            print(f"📍 Position: ({self.x:.1f}, {self.y:.1f}) | Angle: {self.angle:.1f}°")
            print(f"🏎️  Speed: {self.speed:.2f} | Distance: {self.total_distance:.1f}px")
            
            # Collision status
            collision_status = "🔴 COLLISION!" if self.is_colliding else "🟢 CLEAR"
            print(f"💥 Collision: {collision_status}")
            
            # Sensor data
            if len(self.sensor_distances) == 3:
                left, center, right = self.sensor_distances
                print(f"🔍 Sensors [L|C|R]: [{left:.3f}|{center:.3f}|{right:.3f}]")
                
                # Sensor status
                sensor_status = []
                for i, (name, dist) in enumerate(zip(['LEFT', 'CENTER', 'RIGHT'], self.sensor_distances)):
                    if dist < 0.3:
                        status = f"{name}: 🔴 DANGER"
                    elif dist < 0.6:
                        status = f"{name}: 🟡 CAUTION"
                    else:
                        status = f"{name}: 🟢 CLEAR"
                    sensor_status.append(status)
                
                for status in sensor_status:
                    print(f"   {status}")
            
            # Quantum brain decisions
            print(f"🧠 Quantum Decisions:")
            print(f"   Steering: Spike={self.last_steering_spike} | Prob={self.last_steering_prob:.3f}")
            print(f"   Speed:    Spike={self.last_speed_spike} | Prob={self.last_speed_prob:.3f}")
            
            # Driving behavior analysis
            avg_sensor_distance = np.mean(self.sensor_distances) if self.sensor_distances else 0
            if self.is_colliding:
                behavior = "🚨 COLLISION RECOVERY"
            elif avg_sensor_distance < 0.4:
                behavior = "🚨 OBSTACLE AVOIDANCE"
            elif avg_sensor_distance < 0.7:
                behavior = "⚠️  CAUTIOUS DRIVING"
            else:
                behavior = "🏁 OPEN ROAD CRUISING"
            
            print(f"🎯 Behavior: {behavior}")
            print(f"{'='*80}")
            
            self.last_log_time = current_time

    def update(self, obstacles, screen_width, screen_height):
        """Update car position based on quantum neural network decisions"""
        # Cast sensors first
        self.cast_sensors(obstacles, screen_width, screen_height)
        
        # Check for collisions and handle bouncing
        collision_occurred = self.check_collision(obstacles)
        
        # Get quantum decisions (only if not in collision recovery)
        if not self.is_colliding:
            steering_spike, speed_spike = self.quantum_decision()
            
            # Apply steering based on sensor data and quantum decision
            if len(self.sensor_distances) == 3:
                left_distance, center_distance, right_distance = self.sensor_distances
                
                # Smart steering based on sensor data and quantum decision
                if center_distance < 0.3:  # Obstacle ahead
                    if left_distance > right_distance:
                        self.angle -= self.turn_speed * 2  # Turn left more aggressively
                    else:
                        self.angle += self.turn_speed * 2  # Turn right more aggressively
                elif left_distance < 0.5:  # Obstacle on left
                    self.angle += self.turn_speed
                elif right_distance < 0.5:  # Obstacle on right
                    self.angle -= self.turn_speed
                
                # Fine tuning with quantum decision
                if steering_spike == 1:
                    self.angle += self.turn_speed * 0.5
                else:
                    self.angle -= self.turn_speed * 0.5
            
            # Apply speed control - FIXED: Check if sensor_distances is not empty
            min_sensor_distance = min(self.sensor_distances) if self.sensor_distances else 1.0
            
            if speed_spike == 1 and min_sensor_distance > 0.4:
                self.speed = min(self.speed + self.acceleration, self.max_speed)
            else:
                self.speed = max(self.speed - self.acceleration * 2, 0.5)  # Minimum speed
        
        # Update position
        angle_rad = math.radians(self.angle)
        self.x += self.speed * math.cos(angle_rad)
        self.y += self.speed * math.sin(angle_rad)
        
        # Update trail
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.trail_max_length:
            self.trail.pop(0)
        
        # Log data at regular intervals
        self.log_data()

    def draw(self, screen):
        """Draw car, sensors, and collision effects"""
        # Draw trail with collision indicator
        if len(self.trail) > 1:
            for i in range(1, len(self.trail)):
                alpha = i / len(self.trail)
                if self.is_colliding:
                    color = (int(255 * alpha), int(0 * alpha), int(0 * alpha))  # Red trail during collision
                else:
                    color = (int(0 * alpha), int(255 * alpha), int(0 * alpha))  # Green trail normally
                pygame.draw.circle(screen, color, (int(self.trail[i][0]), int(self.trail[i][1])), 2)
        
        # Draw car body
        car_points = [
            (-CAR_WIDTH//2, -CAR_HEIGHT//2),
            (CAR_WIDTH//2, -CAR_HEIGHT//2),
            (CAR_WIDTH//2, CAR_HEIGHT//2),
            (-CAR_WIDTH//2, CAR_HEIGHT//2)
        ]
        
        # Rotate car points
        angle_rad = math.radians(self.angle)
        rotated_points = []
        for px, py in car_points:
            rx = px * math.cos(angle_rad) - py * math.sin(angle_rad)
            ry = px * math.sin(angle_rad) + py * math.cos(angle_rad)
            rotated_points.append((self.x + rx, self.y + ry))
        
        # Change car color during collision
        car_color = RED if self.is_colliding else BLUE
        pygame.draw.polygon(screen, car_color, rotated_points)
        
        # Draw collision radius (debug visualization)
        if self.is_colliding:
            pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), self.collision_radius, 2)
        
        # Draw direction indicator
        front_x = self.x + CAR_WIDTH//2 * math.cos(angle_rad)
        front_y = self.y + CAR_WIDTH//2 * math.sin(angle_rad)
        pygame.draw.circle(screen, WHITE, (int(front_x), int(front_y)), 3)
        
        # Draw sensors
        for i, (sensor_angle, distance) in enumerate(zip(self.sensor_angles, self.sensor_distances)):
            angle_rad = math.radians(self.angle + sensor_angle)
            sensor_length = distance * SENSOR_RANGE
            
            end_x = self.x + sensor_length * math.cos(angle_rad)
            end_y = self.y + sensor_length * math.sin(angle_rad)
            
            # Color sensor based on distance
            if distance < 0.3:
                color = RED
            elif distance < 0.6:
                color = YELLOW
            else:
                color = GREEN
                
            pygame.draw.line(screen, color, (int(self.x), int(self.y)), 
                           (int(end_x), int(end_y)), 2)

class ObstacleField:
    """Manages obstacles (pylons) in the environment"""
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.obstacles = []
        self.generate_course()
    
    def generate_course(self):
        """Generate a challenging obstacle course"""
        # Create a slalom course
        num_gates = 8
        for i in range(num_gates):
            y = 100 + i * 80
            
            if i % 2 == 0:
                # Gates alternating left-right
                left_x = 200 + i * 20
                right_x = 600 - i * 20
            else:
                left_x = 300 - i * 15
                right_x = 500 + i * 15
            
            self.obstacles.append((left_x, y))
            self.obstacles.append((right_x, y))
        
        # Add some random obstacles
        for _ in range(15):
            x = random.randint(100, self.screen_width - 100)
            y = random.randint(100, self.screen_height - 100)
            
            # Ensure obstacles aren't too close to start position
            if math.sqrt((x - 150)**2 + (y - 350)**2) > 80:
                self.obstacles.append((x, y))
    
    def draw(self, screen):
        """Draw all obstacles"""
        for obs_x, obs_y in self.obstacles:
            # Draw pylon base
            pygame.draw.circle(screen, ORANGE, (int(obs_x), int(obs_y)), PYLON_SIZE//2)
            pygame.draw.circle(screen, RED, (int(obs_x), int(obs_y)), PYLON_SIZE//2 - 3)
            
            # Draw pylon top
            pygame.draw.circle(screen, WHITE, (int(obs_x), int(obs_y) - 5), 3)

def draw_ui(screen, car):
    """Draw user interface information"""
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    # Title
    title_text = font.render("Quantum Spiking Neural Network Car", True, WHITE)
    screen.blit(title_text, (10, 10))
    
    # Collision status
    collision_color = RED if car.is_colliding else GREEN
    collision_text = small_font.render(f"Collision: {'YES' if car.is_colliding else 'NO'}", True, collision_color)
    screen.blit(collision_text, (10, 40))
    
    # Sensor data
    sensor_text = small_font.render("Sensors (L/C/R):", True, WHITE)
    screen.blit(sensor_text, (10, 65))
    
    if len(car.sensor_distances) == 3:
        for i, distance in enumerate(car.sensor_distances):
            color = RED if distance < 0.3 else YELLOW if distance < 0.6 else GREEN
            sensor_info = small_font.render(f"{distance:.2f}", True, color)
            screen.blit(sensor_info, (10 + i * 50, 90))
    
    # Speed and quantum data
    speed_text = small_font.render(f"Speed: {car.speed:.1f}", True, WHITE)
    screen.blit(speed_text, (10, 115))
    
    # Quantum brain activity
    brain_text = small_font.render("Quantum Brain:", True, WHITE)
    screen.blit(brain_text, (10, 140))
    
    steering_text = small_font.render(f"Steering: {car.last_steering_spike} ({car.last_steering_prob:.2f})", True, WHITE)
    screen.blit(steering_text, (10, 165))
    
    speed_brain_text = small_font.render(f"Speed: {car.last_speed_spike} ({car.last_speed_prob:.2f})", True, WHITE)
    screen.blit(speed_brain_text, (10, 190))
    
    # Instructions
    instructions = [
        "🧠 Quantum Brain Controls:",
        "• Blue sensors detect obstacles", 
        "• Red = Close, Yellow = Medium, Green = Far",
        "• Neural spikes control steering & speed",
        "• 💥 Car bounces back from collisions!",
        "• Check terminal for detailed logs!",
        "• Press R to reset, Q to quit"
    ]
    
    for i, instruction in enumerate(instructions):
        inst_text = small_font.render(instruction, True, WHITE)
        screen.blit(inst_text, (SCREEN_WIDTH - 400, 10 + i * 25))

def main():
    """Main simulation loop"""
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Quantum Spiking Neural Network - Autonomous Car with Collision Detection")
    clock = pygame.time.Clock()
    
    # Print initial info
    print("🚗 Starting Quantum Spiking Neural Network Car Simulation...")
    print("🧠 The car's brain uses quantum spikes to navigate through obstacles!")
    print("💥 Added collision detection and bounce-back physics!")
    print("📊 Watch the terminal for detailed sensor data and neural activity logs.")
    print(f"⏱️  Logging interval: {LOG_INTERVAL} seconds")
    print("=" * 80)
    
    # Create car and obstacles
    car = Car(150, 350)
    obstacle_field = ObstacleField(SCREEN_WIDTH, SCREEN_HEIGHT)
    
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset car position
                    print("\n🔄 Resetting car position...")
                    car = Car(150, 350)
        
        # Update
        car.update(obstacle_field.obstacles, SCREEN_WIDTH, SCREEN_HEIGHT)
        
        # Draw everything
        screen.fill(DARK_GREEN)  # Grass background
        
        # Draw road/track
        pygame.draw.rect(screen, GRAY, (100, 0, SCREEN_WIDTH - 200, SCREEN_HEIGHT))
        
        # Draw lane markings
        for y in range(0, SCREEN_HEIGHT, 40):
            pygame.draw.rect(screen, WHITE, (SCREEN_WIDTH//2 - 2, y, 4, 20))
        
        obstacle_field.draw(screen)
        car.draw(screen)
        draw_ui(screen, car)
        
        pygame.display.flip()
        clock.tick(60)  # 60 FPS
    
    pygame.quit()

if __name__ == "__main__":
    main()
