import math
from enum import Enum, auto
from controller import Robot
from base import Base
from arm import Arm
from gripper import Gripper
from fuzzy_logic import trapezoidal, triangular, AND, defuzzify, OR
from ultralytics import YOLO
import numpy as np

class Estados(Enum):
    INICIAL = 0
    PROCURA_CUBO = 1
    APROXIMA_CUBO = 2
    ALINHA_CUBO = 3
    AGARRA_CUBO = 4
    PROCURA_CAIXA = 5
    APROXIMA_CAIXA = 6
    SOLTA_CUBO = 7

class CLASSES(Enum):
    RED_CUBE = 0
    GREEN_CUBE = 1
    BLUE_CUBE = 2
    RED_BASKET = 3
    GREEN_BASKET = 4
    BLUE_BASKET = 5
    OBSTACLE = 6


TOTAL_CUBOS = 15
TEMPO_PASSEIO = 6.0  # segundos

class DetectedObject:
    def __init__(self, d_class, distance, angle, xy_inicial, xy_final, x_normalized, conf):
        self.d_class = d_class
        self.distance = distance
        self.angle = angle
        self.xy_inicial = xy_inicial
        self.xy_final = xy_final
        self.x_normalized = x_normalized
        self.conf = conf
    @property
    def wh(self):
        return [self.xy_final[0] - self.xy_inicial[0], self.xy_final[1] - self.xy_inicial[1]]

class YouBotController:
    def __init__(self):
        # --- Inicialização ---
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())

        self.base = Base(self.robot)
        self.arm = Arm(self.robot)
        self.gripper = Gripper(self.robot)
        
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.time_step)

        self.display = self.robot.getDevice("display")

        self.display.attachCamera(self.camera)

        self.yolo_model = YOLO("best.pt")

        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.time_step)
        self.lidar.enablePointCloud()

        self.estado_atual = Estados.INICIAL
        self.contagem_inicial = 0

        self.tempo_passeio = -1.0
        self.tempo_rotacao = -1.0

        self.colors = [
            0xF28D94,  # Light Red
            0x94F2B8,  # Light Green
            0x94C2F2,  # Light Blue
            0xFF0000,  # Red
            0x00FF00,  # Green
            0x0000FF,  # Blue
            0x8A6C46,  # Brown
        ]
        
        self.detected_objects = []
        self.lidar_points = []

    # --- Funções de Pertinência ---
    def fuzzy_dist_cube(self, d):
        return {
            "perto": trapezoidal(d, 0.0, 0.0, 0.25, 0.35),
            "medio": triangular(d, 0.30, 0.55, 0.80),
            "longe": trapezoidal(d, 0.70, 0.90, 1.0, 1.0)
        }

    def fuzzy_ang_cube(self, a):
        return {
            "direita": trapezoidal(a, -math.pi, -math.pi, -0.6, -0.10),
            "centro": triangular(a, -0.20, 0.0, 0.20),
            "esquerda": trapezoidal(a, 0.10, 0.6, math.pi, math.pi)
        }

    def fuzzy_dist_obs(self, d):
        return {
            "perto": trapezoidal(d, 0.0, 0.0, 0.25, 0.45),
            "medio": triangular(d, 0.40, 0.65, 0.90),
            "longe": trapezoidal(d, 0.80, 1.0, 1.0, 1.0)
        }
    
    def fuzzy_dist_wall(self, d):
        return {
            "perto": trapezoidal(d, 0.0, 0.0, 0.15, 0.30),
            "medio": triangular(d, 0.25, 0.45, 0.65),
            "longe": trapezoidal(d, 0.55, 0.75, 1.0, 1.0)
        }
    def fuzzy_ang_wall(self, a):
        return {
            "direita": trapezoidal(a, -math.pi, -math.pi, -0.6, -0.10),
            "centro": triangular(a, -0.20, 0.0, 0.20),
            "esquerda": trapezoidal(a, 0.10, 0.6, math.pi, math.pi)
        }
    def get_nearest_lidar_point(self):
        nearest_point = min(self.lidar_points, key=lambda p: p[0])
        return nearest_point  # (distance, angle)

    def fuzzy_dist_nearest(self):
        dist = self.get_nearest_lidar_point()[0]
        if dist > 1.0:
            dist = 1.0
        
        return {
            "perto": trapezoidal(dist, 0.0, 0.0, 0.25, 0.45),
            "medio": triangular(dist, 0.40, 0.65, 0.90),
            "longe": trapezoidal(dist, 0.80, 1.0, 1.0, 1.0)
        }
    def fuzzy_ang_nearest(self):
        omega = self.get_nearest_lidar_point()[1]

        return {
            "negativa": trapezoidal(omega, -math.pi, -math.pi, -0.6, -0.10),
            "zero": triangular(omega, -0.20, 0.0, 0.20),
            "positiva": trapezoidal(omega, 0.10, 0.6, math.pi, math.pi)
        }
    
    def inferencia_vx_passeio(self):
        # Regras de inferência para vx durante o passeio
        vx_baixa = AND(self.fuzzy_dist_nearest()["perto"], self.fuzzy_ang_nearest()["zero"])
        vx_media = AND(self.fuzzy_dist_nearest()["medio"], self.fuzzy_ang_nearest()["zero"])
        vx_alta = AND(self.fuzzy_dist_nearest()["longe"], self.fuzzy_ang_nearest()["zero"])

        return {
            "baixa": trapezoidal(vx_baixa, 0.0, 0.0, 0.25, 0.45),
            "media": triangular(vx_media, 0.40, 0.65, 0.90),
            "alta": trapezoidal(vx_alta, 0.80, 1.0, 1.0, 1.0)
        }
    
    def inferencia_vy_passeio(self):
        return {
            "negativa": trapezoidal(self.vy, -1.0, -1.0, -0.6, -0.10),
            "zero": triangular(self.vy, -0.20, 0.0, 0.20),
            "positiva": trapezoidal(self.vy, 0.10, 0.6, 1.0, 1.0)
        }

    def inferencia_omega_passeio(self):
        return {
            "negativa": trapezoidal(self.omega, -math.pi, -math.pi, -0.6, -0.10),
            "zero": triangular(self.omega, -0.20, 0.0, 0.20),
            "positiva": trapezoidal(self.omega, 0.10, 0.6, math.pi, math.pi)
        }
    


    def bgra_bytes_to_bgr(self, bgra_bytes, width, height):
        # BGRA = 4 channels
        bgra = np.frombuffer(bgra_bytes, dtype=np.uint8)
        bgra = bgra.reshape((height, width, 4))

        # Drop alpha channel → BGR
        bgr = bgra[:, :, :3]

        return bgr
    def get_pictures(self):
        generations = 20
        samples = 20
        for gen in range(generations):
            self.robot.step(200)  # Wait for 200 ms at start of generation
            for samp in range(samples):
                self.robot.step(200)
                filename = f"pictures/pic_gen{gen}_sample{samp}.png"
                self.camera.saveImage(filename, 100)
                print(f"Saved image: {filename}")

    def measure_lidar(self):
        ranges = self.lidar.getRangeImage()
        fov = self.lidar.getFov()
        n = len(ranges)
        self.lidar_points = []
        for i in range(n):
            angle = -fov/2 + fov * i / (n - 1)
            distance = ranges[i]
            self.lidar_points.append((distance, angle))

    def detect(self):
        imageBuffer = self.camera.getImage()
        img = self.bgra_bytes_to_bgr(imageBuffer, self.camera.getWidth(), self.camera.getHeight())
        results = self.yolo_model(img, verbose=False, imgsz=128, conf=0.3)

        self.detected_objects = []
        for r in results[0].boxes:
            box = (r.xyxy[0]).tolist()

            x_normalized = r.xywhn[0].tolist()[0]

            d_class = CLASSES(r.cls.item())
            distance = 1.0
            angle = self.calcular_angulo_graus(x_normalized, self.camera.getFov())
            xy_inicial = [box[0], box[1]]
            xy_final = [box[2], box[3]]
            conf = r.conf.item()
            o = DetectedObject(d_class, distance, angle, xy_inicial, xy_final, x_normalized, conf)
            self.detected_objects.append(o)


    def display_overlays(self):
        self.display.setAlpha(0.0)
        self.display.fillRectangle(0, 0, self.display.getWidth(), self.display.getHeight())
        self.display.setAlpha(1.0)

        for o in self.detected_objects:
            angulo = self.calcular_angulo_graus(o.x_normalized, self.camera.getFov())
            self.display.setColor(self.colors[o.d_class.value])
            self.display.drawRectangle(o.xy_inicial[0]*2, o.xy_inicial[1]*2, o.wh[0]*2, o.wh[1]*2)
            self.display.drawText(f"{o.d_class} {o.conf:.2f} {angulo:.2f}°", o.xy_inicial[0]*2, o.xy_inicial[1]*2-10)

    def calcular_angulo_rad(self, x, fov):
        return math.atan(math.tan(fov / 2) * (2 * x - 1))
    
    def calcular_angulo_graus(self, x, fov):
        return math.degrees(self.calcular_angulo_rad(x, fov))
        
    def passeio(self):
        if self.tempo_passeio < 0.0 and self.tempo_rotacao < 0.0:
            self.tempo_passeio = TEMPO_PASSEIO
            self.tempo_rotacao = np.random.uniform(0.5, 4.0)
            self.direction = np.random.choice([-1, 1])

        vx = defuzzify(self.inferencia_vx_passeio())

        if self.tempo_passeio > 0:
            self.base.move(.3*vx, 0.0, 0.0)
            self.tempo_passeio -= self.time_step / 1000.0
        elif self.tempo_rotacao > 0:
            self.tempo_rotacao -= self.time_step / 1000.0
            self.base.move(0.0, 0.0, math.pi/4*self.direction)



    # --- Loop Principal ---
    def run(self):
        # get_pictures() # Usar em conjunto com supervisor.py para capturar imagens

        while self.robot.step(self.time_step) != -1:
            
            self.detect()
            self.measure_lidar()

            self.display_overlays()

            # --- ESTADO: PROCURANDO CUBO ---
            if self.estado_atual == Estados.INICIAL:
                self.estado_atual = Estados.PROCURA_CUBO
            elif self.estado_atual == Estados.PROCURA_CUBO:
                self.passeio()
            elif self.estado_atual == Estados.APROXIMA_CUBO:
                pass


if __name__ == "__main__":
    c = YouBotController()
    c.run()