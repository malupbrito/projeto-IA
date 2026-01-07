import math
from enum import Enum, auto
from controller import Robot
from base import Base
from arm import Arm, ArmHeight
from gripper import Gripper
from fuzzy_logic import FuzzySet, FuzzySystem, LinguisticVariable, Rule, defuzzify, triangular, trapezoidal
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
TEMPO_PASSEIO = 12  # segundos

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
        self.passeio_rtime = 0

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
        self.lidar_fov = self.lidar.getFov()

        self.init_fuzzy_variables()
        self.rules_sistema_passeio()
        self.rules_sistema_aprox_cube()
        self.rules_sistema_alinha_cubo()

        # Memory
        # Passeio
        self.direction = 1
        self.passeio_timeout = TEMPO_PASSEIO
        # Aprox Cubo
        self.ato = None
        self.target_cube = None

        self.grab_state = 0
        self.grab_counter = 0
        self.drop_state = 0
        self.drop_counter = 0

        self.target_b_class = None
        self.target_basket = None
    
    def get_nearest_lidar_point(self):
        nearest_point = min(self.lidar_points, key=lambda p: p[0])
        return nearest_point  # (distance, angle)
    
    def get_nearest_obs(self):
        obs = None
        for o in self.detected_objects:
            if abs(o.angle - self.target_cube.angle) > 0.17:
                if obs is None or obs.distance > o.distance:
                    obs = o
        if obs is None:
            return (1.0, -0.786)

        return (min(obs.distance, 1.0), obs.angle)
    
    def init_fuzzy_variables(self):
        self.infPasseio = None
        self.infAproxCube = None
        self.infAlignCube = None

        self.distTgt = LinguisticVariable("DistanciaTgt", np.arange(0, 1.01, 0.01))
        self.distTgt.add_set(FuzzySet("PERTO", trapezoidal(0.0, 0.0, 0.05, 0.2)))
        self.distTgt.add_set(FuzzySet("MEDIO", triangular(0.17, 0.2, 0.4)))
        self.distTgt.add_set(FuzzySet("LONGE", trapezoidal(0.35, 0.4, 1.0, 1.1)))

        self.angleTgt = LinguisticVariable("AnguloTgt", np.arange(-0.786, 0.786, 0.05))
        self.angleTgt.add_set(FuzzySet("ESQUERDA", trapezoidal( 0.0, 0.393, 0.786, 0.786)))
        self.angleTgt.add_set(FuzzySet("CENTRO", triangular(-0.393, 0.0, .393)))
        self.angleTgt.add_set(FuzzySet("DIREITA", trapezoidal(-.786, -.786, -.383, 0.0 )))

        self.distObs = LinguisticVariable("DistanciaObs", np.arange(0, 1.01, 0.01))
        self.distObs.add_set(FuzzySet("PERTO", trapezoidal(0.0, 0.0, 0.05, 0.2)))
        self.distObs.add_set(FuzzySet("MEDIO", triangular(0.17, 0.2, 0.4)))
        self.distObs.add_set(FuzzySet("LONGE", trapezoidal(0.35, 0.4, 1.0, 1.1)))

        self.angleObs = LinguisticVariable("AnguloObs", np.arange(-0.786, 0.786, 0.05))
        self.angleObs.add_set(FuzzySet("ESQUERDA", trapezoidal( 0.0, 0.393, 0.786, 0.786)))
        self.angleObs.add_set(FuzzySet("CENTRO", triangular(-0.393, 0.0, .393)))
        self.angleObs.add_set(FuzzySet("DIREITA", trapezoidal(-.786, -.786, -.383, 0.0 )))

        self.vxPasseio = LinguisticVariable("Vx", np.arange(-1.0, 1.01, 0.01))
        self.vxPasseio.add_set(FuzzySet("NEGATIVO", trapezoidal(-1.0, -1.0, -0.5, 0.0)))
        self.vxPasseio.add_set(FuzzySet("ZERO", triangular(-0.5, 0.0, 0.5)))
        self.vxPasseio.add_set(FuzzySet("POSITIVO", trapezoidal(0.0, 0.5, 1.0, 1.0)))

        self.omegaPasseio = LinguisticVariable("Omega", np.arange(-1.0, 1.01, 0.01))
        self.omegaPasseio.add_set(FuzzySet("DIREITA", trapezoidal(-1.0, -1.0, -0.5, 0.0)))
        self.omegaPasseio.add_set(FuzzySet("CENTRO", triangular(-0.5, 0.0, 0.5)))
        self.omegaPasseio.add_set(FuzzySet("ESQUERDA", trapezoidal(0.0, 0.5, 1.0, 1.0)))

        self.vxAproxCube = LinguisticVariable("Vx", np.arange(-1.0, 1.01, 0.01))
        self.vxAproxCube.add_set(FuzzySet("NEGATIVO", trapezoidal(-1.0, -1.0, -0.5, 0.0)))
        self.vxAproxCube.add_set(FuzzySet("ZERO", triangular(-0.5, 0.0, 0.5)))
        self.vxAproxCube.add_set(FuzzySet("POSITIVO", trapezoidal(0.0, 0.5, 1.0, 1.0)))

        self.omegaAproxCube = LinguisticVariable("Omega", np.arange(-1.0, 1.01, 0.01))
        self.omegaAproxCube.add_set(FuzzySet("DIREITA", trapezoidal(-1.0, -1.0, -0.5, 0.0)))
        self.omegaAproxCube.add_set(FuzzySet("CENTRO", triangular(-0.5, 0.0, 0.5)))
        self.omegaAproxCube.add_set(FuzzySet("ESQUERDA", trapezoidal(0.0, 0.5, 1.0, 1.0)))

        self.vxAlignCube = LinguisticVariable("Vx", np.arange(-1.0, 1.01, 0.01))
        self.vxAlignCube.add_set(FuzzySet("NEGATIVO", trapezoidal(-1.0, -1.0, -0.5, 0.0)))
        self.vxAlignCube.add_set(FuzzySet("ZERO", triangular(-0.5, 0.0, 0.5)))
        self.vxAlignCube.add_set(FuzzySet("POSITIVO", trapezoidal(0.0, 0.5, 1.0, 1.0)))

        self.vyAlignCube = LinguisticVariable("Vy", np.arange(-1.0, 1.01, 0.01))
        self.vyAlignCube.add_set(FuzzySet("ESQUERDA", trapezoidal(-1.0, -1.0, -0.5, 0.0)))
        self.vyAlignCube.add_set(FuzzySet("PARA", triangular(-0.5, 0.0, 0.5)))
        self.vyAlignCube.add_set(FuzzySet("DIREITA", trapezoidal(0.0, 0.5, 1.0, 1.0)))

        self.sistemaPasseio = FuzzySystem()
        self.sistemaPasseio.add_variable(self.distObs)
        self.sistemaPasseio.add_variable(self.angleObs)
        self.sistemaPasseio.add_variable(self.vxPasseio)
        self.sistemaPasseio.add_variable(self.omegaPasseio)

        self.sistemaAproxCube = FuzzySystem()
        self.sistemaAproxCube.add_variable(self.distTgt)
        self.sistemaAproxCube.add_variable(self.angleTgt)
        self.sistemaAproxCube.add_variable(self.distObs)
        self.sistemaAproxCube.add_variable(self.angleObs)
        self.sistemaAproxCube.add_variable(self.vxAproxCube)
        self.sistemaAproxCube.add_variable(self.omegaAproxCube)

        self.sistemaAlignCube = FuzzySystem()
        self.sistemaAlignCube.add_variable(self.distTgt)
        self.sistemaAlignCube.add_variable(self.angleTgt)
        self.sistemaAlignCube.add_variable(self.vxAlignCube)
        self.sistemaAlignCube.add_variable(self.vyAlignCube)


    def rules_sistema_passeio(self):
        self.sistemaPasseio.add_rule(Rule({"DistanciaObs": "PERTO", "AnguloObs": "CENTRO"}, { "Vx": "NEGATIVO"}, operator='AND'))
        self.sistemaPasseio.add_rule(Rule({"DistanciaObs": "MEDIO", "AnguloObs": "CENTRO"}, { "Vx": "NEGATIVO"}, operator='AND'))
        self.sistemaPasseio.add_rule(Rule({"DistanciaObs": "LONGE"}, { "Vx": "POSITIVO"}))
        self.sistemaPasseio.add_rule(Rule({"DistanciaObs": "PERTO","AnguloObs": "ESQUERDA"}, { "Vx": "NEGATIVO"}, operator='AND'))
        self.sistemaPasseio.add_rule(Rule({"DistanciaObs": "PERTO","AnguloObs": "DIREITA"}, { "Vx": "NEGATIVO"}, operator='AND'))
        self.sistemaPasseio.add_rule(Rule({"DistanciaObs": "MEDIO","AnguloObs": "ESQUERDA"}, { "Vx": "ZERO"}, operator='AND'))
        self.sistemaPasseio.add_rule(Rule({"DistanciaObs": "MEDIO","AnguloObs": "DIREITA"}, { "Vx": "ZERO"}, operator='AND'))
        self.sistemaPasseio.add_rule(Rule({"AnguloObs": "DIREITA"}, { "Vx": "POSITIVO"}))

        self.sistemaPasseio.add_rule(Rule({"DistanciaObs": "PERTO", "AnguloObs": "ESQUERDA"}, {"Omega": "DIREITA"}))
        self.sistemaPasseio.add_rule(Rule({"DistanciaObs": "PERTO", "AnguloObs": "CENTRO"}, {"Omega": "CENTRO"}))
        self.sistemaPasseio.add_rule(Rule({"DistanciaObs": "PERTO", "AnguloObs": "DIREITA"}, {"Omega": "ESQUERDA"}))
        self.sistemaPasseio.add_rule(Rule({"DistanciaObs": "MEDIO", "AnguloObs": "ESQUERDA"}, {"Omega": "DIREITA"}))
        self.sistemaPasseio.add_rule(Rule({"DistanciaObs": "MEDIO", "AnguloObs": "CENTRO"}, {"Omega": "CENTRO"}))
        self.sistemaPasseio.add_rule(Rule({"DistanciaObs": "MEDIO", "AnguloObs": "DIREITA"}, {"Omega": "ESQUERDA"}))
        self.sistemaPasseio.add_rule(Rule({"DistanciaObs": "LONGE", "AnguloObs": "ESQUERDA"}, {"Omega": "CENTRO"}))
        self.sistemaPasseio.add_rule(Rule({"DistanciaObs": "LONGE", "AnguloObs": "CENTRO"}, {"Omega": "CENTRO"}))
        self.sistemaPasseio.add_rule(Rule({"DistanciaObs": "LONGE", "AnguloObs": "DIREITA"}, {"Omega": "CENTRO"}))

    def rules_sistema_aprox_cube(self):
        self.sistemaAproxCube.add_rule(Rule({"DistanciaTgt": "PERTO", "AnguloTgt": "CENTRO"}, { "Vx": "ZERO"}, operator='AND'))
        self.sistemaAproxCube.add_rule(Rule({"DistanciaTgt": "MEDIO", "AnguloTgt": "CENTRO"}, { "Vx": "POSITIVO"}, operator='AND'))
        self.sistemaAproxCube.add_rule(Rule({"DistanciaTgt": "LONGE", "AnguloTgt": "CENTRO"}, { "Vx": "POSITIVO"}))
        self.sistemaAproxCube.add_rule(Rule({"DistanciaTgt": "PERTO","AnguloTgt": "ESQUERDA"}, { "Vx": "ZERO"}, operator='AND'))
        self.sistemaAproxCube.add_rule(Rule({"DistanciaTgt": "PERTO","AnguloTgt": "DIREITA"}, { "Vx": "ZERO"}, operator='AND'))
        self.sistemaAproxCube.add_rule(Rule({"DistanciaTgt": "MEDIO","AnguloTgt": "ESQUERDA"}, { "Vx": "ZERO"}, operator='AND'))
        self.sistemaAproxCube.add_rule(Rule({"DistanciaTgt": "MEDIO","AnguloTgt": "DIREITA"}, { "Vx": "ZERO"}, operator='AND'))
        self.sistemaAproxCube.add_rule(Rule({"DistanciaObs": "PERTO", "AnguloObs": "CENTRO"}, { "Vx": "NEGATIVO"}, operator='AND'))
        self.sistemaAproxCube.add_rule(Rule({"DistanciaObs": "MEDIO", "AnguloObs": "CENTRO"}, { "Vx": "ZERO"}, operator='AND'))
        self.sistemaAproxCube.add_rule(Rule({"DistanciaObs": "LONGE"}, { "Vx": "POSITIVO"}))
        self.sistemaAproxCube.add_rule(Rule({"DistanciaObs": "PERTO","AnguloObs": "ESQUERDA"}, { "Vx": "NEGATIVO"}, operator='AND'))
        self.sistemaAproxCube.add_rule(Rule({"DistanciaObs": "PERTO","AnguloObs": "DIREITA"}, { "Vx": "NEGATIVO"}, operator='AND'))
        self.sistemaAproxCube.add_rule(Rule({"DistanciaObs": "MEDIO","AnguloObs": "ESQUERDA"}, { "Vx": "ZERO"}, operator='AND'))
        self.sistemaAproxCube.add_rule(Rule({"DistanciaObs": "MEDIO","AnguloObs": "DIREITA"}, { "Vx": "ZERO"}, operator='AND'))

        self.sistemaAproxCube.add_rule(Rule({"DistanciaTgt": "PERTO", "AnguloTgt": "CENTRO"}, { "Omega": "CENTRO"}, operator='AND'))
        self.sistemaAproxCube.add_rule(Rule({"DistanciaTgt": "MEDIO", "AnguloTgt": "CENTRO"}, { "Omega": "CENTRO"}, operator='AND'))
        self.sistemaAproxCube.add_rule(Rule({"DistanciaTgt": "LONGE", "AnguloTgt": "CENTRO"}, { "Omega": "CENTRO"}))
        self.sistemaAproxCube.add_rule(Rule({"DistanciaTgt": "PERTO","AnguloTgt": "ESQUERDA"}, { "Omega": "ESQUERDA"}, operator='AND'))
        self.sistemaAproxCube.add_rule(Rule({"DistanciaTgt": "PERTO","AnguloTgt": "DIREITA"}, { "Omega": "DIREITA"}, operator='AND'))
        self.sistemaAproxCube.add_rule(Rule({"DistanciaTgt": "MEDIO","AnguloTgt": "ESQUERDA"}, { "Omega": "ESQUERDA"}, operator='AND'))
        self.sistemaAproxCube.add_rule(Rule({"DistanciaTgt": "MEDIO","AnguloTgt": "DIREITA"}, { "Omega": "DIREITA"}, operator='AND'))
        self.sistemaAproxCube.add_rule(Rule({"DistanciaObs": "PERTO", "AnguloTgt": "ESQUERDA"}, { "Omega": "ESQUERDA"}, operator='AND'))
        self.sistemaAproxCube.add_rule(Rule({"DistanciaObs": "PERTO", "AnguloTgt": "DIREITA"}, { "Omega": "DIREITA"}, operator='AND'))

    def rules_sistema_alinha_cubo(self):
        self.sistemaAlignCube.add_rule(Rule({"AnguloTgt": "CENTRO"}, {"Vy": "PARA"}))
        self.sistemaAlignCube.add_rule(Rule({"AnguloTgt": "ESQUERDA"}, {"Vy": "ESQUERDA"}))
        self.sistemaAlignCube.add_rule(Rule({"AnguloTgt": "DIREITA"}, {"Vy": "DIREITA"}))
        self.sistemaAlignCube.add_rule(Rule({"DistanciaTgt": "MEDIO"}, {"Vx": "POSITIVO"}))
        self.sistemaAlignCube.add_rule(Rule({"DistanciaTgt": "PERTO"}, {"Vx": "ZERO"}))

    def process_passeio(self):
        p = self.get_nearest_lidar_point()
        entrada = {"DistanciaObs": min(p[0], 1.0), "AnguloObs": p[1]}

        self.infPasseio = self.sistemaPasseio.infer(entrada)
        combined_membership_vx = np.zeros_like(self.vxPasseio.universe, dtype=float)
        combined_membership_omega = np.zeros_like(self.omegaPasseio.universe, dtype=float)
        for res in self.infPasseio:
            for var, (set_name, truth_value) in res.items():
                if var == 'Vx':
                    set_func = self.vxPasseio.sets[set_name].func
                    combined_membership_vx = np.maximum(
                        combined_membership_vx,
                        [min(set_func(x), truth_value) for x in self.vxPasseio.universe]
                    )
                else:
                    set_func = self.omegaPasseio.sets[set_name].func
                    combined_membership_omega = np.maximum(
                        combined_membership_omega,
                        [min(set_func(x), truth_value) for x in self.omegaPasseio.universe]
                    )

        return (
            defuzzify(self.vxPasseio.universe, combined_membership_vx, method="centroid"),
            0.0,
            defuzzify(self.omegaPasseio.universe, combined_membership_omega, method="centroid")
        )
    
    def process_aprox_cube(self):
        entrada = {"DistanciaTgt": self.target_cube.distance, "AnguloTgt": self.target_cube.angle, "DistanciaObs": 1.0, "AnguloObs": -0.786}

        self.infAproxCube = self.sistemaAproxCube.infer(entrada)

        combined_membership_vx = np.zeros_like(self.vxAproxCube.universe, dtype=float)
        combined_membership_omega = np.zeros_like(self.omegaAproxCube.universe, dtype=float)
        for res in self.infAproxCube:
            for var, (set_name, truth_value) in res.items():
                if var == 'Vx':
                    set_func = self.vxAproxCube.sets[set_name].func
                    combined_membership_vx = np.maximum(
                        combined_membership_vx,
                        [min(set_func(x), truth_value) for x in self.vxAproxCube.universe]
                    )
                else:
                    set_func = self.omegaAproxCube.sets[set_name].func
                    combined_membership_omega = np.maximum(
                        combined_membership_omega,
                        [min(set_func(x), truth_value) for x in self.omegaAproxCube.universe]
                    )

        return (
            defuzzify(self.vxAproxCube.universe, combined_membership_vx, method="centroid"),
            0.0,
            defuzzify(self.omegaAproxCube.universe, combined_membership_omega, method="centroid")
        )
    
    def process_aprox_basket(self):
        entrada = {"DistanciaTgt": self.target_basket.distance, "AnguloTgt": self.target_basket.angle, "DistanciaObs": 1.0, "AnguloObs": -0.786}

        self.infAproxCube = self.sistemaAproxCube.infer(entrada)

        combined_membership_vx = np.zeros_like(self.vxAproxCube.universe, dtype=float)
        combined_membership_omega = np.zeros_like(self.omegaAproxCube.universe, dtype=float)
        for res in self.infAproxCube:
            for var, (set_name, truth_value) in res.items():
                if var == 'Vx':
                    set_func = self.vxAproxCube.sets[set_name].func
                    combined_membership_vx = np.maximum(
                        combined_membership_vx,
                        [min(set_func(x), truth_value) for x in self.vxAproxCube.universe]
                    )
                else:
                    set_func = self.omegaAproxCube.sets[set_name].func
                    combined_membership_omega = np.maximum(
                        combined_membership_omega,
                        [min(set_func(x), truth_value) for x in self.omegaAproxCube.universe]
                    )

        return (
            defuzzify(self.vxAproxCube.universe, combined_membership_vx, method="centroid"),
            0.0,
            defuzzify(self.omegaAproxCube.universe, combined_membership_omega, method="centroid")
        )

    def processa_alinha_cubo(self):
        entrada = {"DistanciaTgt": self.target_cube.distance, "AnguloTgt": self.target_cube.angle}

        self.infAlignCube = self.sistemaAlignCube.infer(entrada)

        combined_membership_vx = np.zeros_like(self.vxAlignCube.universe, dtype=float)
        combined_membership_vy = np.zeros_like(self.vyAlignCube.universe, dtype=float)
        for res in self.infAlignCube:
            for var, (set_name, truth_value) in res.items():
                if var == 'Vx':
                    set_func = self.vxAlignCube.sets[set_name].func
                    combined_membership_vx = np.maximum(
                        combined_membership_vx,
                        [min(set_func(x), truth_value) for x in self.vxAlignCube.universe]
                    )
                else:
                    set_func = self.vyAlignCube.sets[set_name].func
                    combined_membership_vy = np.maximum(
                        combined_membership_vy,
                        [min(set_func(x), truth_value) for x in self.vyAlignCube.universe]
                    )

        return (
            defuzzify(self.vxAlignCube.universe, combined_membership_vx, method="centroid"),
            defuzzify(self.vyAlignCube.universe, combined_membership_vy, method="centroid"),
            0.0
        )

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
            self.lidar_points.append((distance, -angle))

    def detect(self):
        imageBuffer = self.camera.getImage()
        img = self.bgra_bytes_to_bgr(imageBuffer, self.camera.getWidth(), self.camera.getHeight())
        results = self.yolo_model(img, verbose=False, imgsz=128, conf=0.3)

        self.detected_objects = []
        for r in results[0].boxes:
            box = (r.xyxy[0]).tolist()

            x_normalized = r.xywhn[0].tolist()[0]

            d_class = CLASSES(r.cls.item())
            angle = -self.calcular_angulo_rad(x_normalized, self.camera.getFov())
            distance = min(self.lidar_points, key=lambda x:abs(x[1] - angle))[0]
            xy_inicial = [box[0], box[1]]
            xy_final = [box[2], box[3]]
            conf = r.conf.item()
            o = DetectedObject(d_class, distance, angle, xy_inicial, xy_final, x_normalized, conf)
            self.detected_objects.append(o)

    def state_info(self):
        match(self.estado_atual):
            case Estados.PROCURA_CUBO:
                if self.infPasseio:
                    return "\n".join( [str(v) for v in self.infPasseio])
            case Estados.APROXIMA_CUBO:
                if self.infAproxCube:
                    return  f"Dist: {self.target_cube.distance:.2f} Ang: {self.target_cube.angle:.2f} Cor:{self.target_cube.d_class} \n"\
                        + "\n".join( [str(v) for v in self.infAproxCube])
            case Estados.ALINHA_CUBO:
                if self.infAlignCube:
                    return  f"Dist: {self.target_cube.distance:.2f} Ang: {self.target_cube.angle:.2f} Cor:{self.target_cube.d_class} \n"\
                        + "\n".join( [str(v) for v in self.infAlignCube])
            case Estados.AGARRA_CUBO:
                pass
            case Estados.PROCURA_CAIXA:
                return str(self.target_b_class)
        return " "
    
    def display_overlays(self):
        self.display.setAlpha(0.0)
        self.display.fillRectangle(0, 0, self.display.getWidth(), self.display.getHeight())
        self.display.setAlpha(1.0)

        for o in self.detected_objects:
            angulo = self.calcular_angulo_graus(o.x_normalized, self.camera.getFov())
            self.display.setColor(self.colors[o.d_class.value])
            self.display.drawRectangle(o.xy_inicial[0]*2, o.xy_inicial[1]*2, o.wh[0]*2, o.wh[1]*2)
            self.display.drawText(f"{o.d_class} {o.conf:.2f} {angulo:.2f}°", o.xy_inicial[0]*2, o.xy_inicial[1]*2-10)
        
        if self.estado_atual == Estados.PROCURA_CUBO:
            self.display.setColor(0x000000)
            self.display.drawText("Estado: PROCURANDO CUBO", 0, 0)
        elif self.estado_atual == Estados.APROXIMA_CUBO:
            self.display.setColor(0x000000)
            self.display.drawText("Estado: APROXIMANDO CUBO", 0, 0)
        elif self.estado_atual == Estados.ALINHA_CUBO:
            self.display.setColor(0x000000)
            self.display.drawText("Estado: ALINHANDO CUBO", 0, 0)
        elif self.estado_atual == Estados.AGARRA_CUBO:
            self.display.setColor(0x000000)
            self.display.drawText("Estado: AGARRANDO CUBO", 0, 0)
        elif self.estado_atual == Estados.PROCURA_CAIXA:
            self.display.setColor(0x000000)
            self.display.drawText("Estado: PROCURANDO CAIXA", 0, 0)
        elif self.estado_atual == Estados.APROXIMA_CAIXA:
            self.display.setColor(0x000000)
            self.display.drawText("Estado: APROXIMANDO CAIXA", 0, 0)
        elif self.estado_atual == Estados.SOLTA_CUBO:
            self.display.setColor(0x000000)
            self.display.drawText("Estado: SOLTANDO CUBO", 0, 0)
        self.display.drawText(self.state_info(), 0, 10)
            

    def calcular_angulo_rad(self, x, fov):
        return math.atan(math.tan(fov / 2) * (2 * x - 1))
    
    def calcular_angulo_graus(self, x, fov):
        return math.degrees(self.calcular_angulo_rad(x, fov))
        
    def passeio(self):
        if self.passeio_timeout < 0:
            self.passeio_timeout = TEMPO_PASSEIO
            self.passeio_rtime = np.random.random() * 3
            self.direction = np.random.choice([-1, 1])

        if self.passeio_rtime > 0.0:
            self.passeio_rtime -= self.time_step / 1000.0
            self.base.move(0.0, 0.0, math.pi/4*self.direction)
        else:
            self.passeio_timeout -= self.time_step / 1000.0
            vx, vy, omega = self.process_passeio()
            self.base.move(vx * 0.3, vy, omega)
            self.passeio_timeout -= self.time_step / 1000.0
            

    def aprox(self):
        self.nearest_cube()
        
        vx, vy, omega = self.process_aprox_cube()

        self.base.move(vx * 0.5, vy, omega * 0.7)

    def aprox_basket(self):
        vx, vy, omega = self.process_aprox_basket()

        self.base.move(vx * 0.5, vy, omega * 0.7)

    def cube_timeout(self):
        
        if self.ato is None:
            self.ato = 0.5
        self.ato -= self.time_step / 1000.0
        if self.ato <= 0.0:
            self.ato = 0.5
            return True
        return False
    
    def nearest_cube(self):
        alvos = [CLASSES.RED_CUBE, CLASSES.GREEN_CUBE, CLASSES.BLUE_CUBE]
        nearest = None
        for ob in self.detected_objects:
            if ob.d_class in alvos and ob.distance < 1.0:
                if not nearest or nearest.distance > ob.distance:
                    nearest = ob
        if nearest is not None and self.cube_match(nearest):
            self.target_cube = nearest
            self.ato = 0.5
            return True
        else:
            return False
        
    def cube_match(self, cube):
        if self.target_cube is None:
            return True
        if np.abs(self.target_cube.distance - cube.distance) < 0.05 and \
            np.abs(self.target_cube.angle - cube.angle) < 0.03:
            return True
        return False

    def find_basket(self, cube):
        if self.target_b_class is None:
            self.target_b_class = CLASSES(self.target_cube.d_class.value + 3)
        for ob in self.detected_objects:
            if ob.d_class == self.target_b_class:
                self.target_basket = ob
                return True
            else:
                return False
    
    def alinha_cubo(self):
        self.nearest_cube()
        vx, vy, omega = self.processa_alinha_cubo()

        self.base.move(vx * 0.02, vy*0.01, omega)
        pass

    def cubo_alinhado(self):
        if np.abs(self.target_cube.distance - 0.1) < 0.1 and np.abs(self.target_cube.angle) < 0.03:
            return True
        return False

    def agarra_cubo(self):
        delay = 150 
        self.grab_counter += 1

        if self.grab_state == 0:
            if self.grab_counter == 1:
                self.gripper.release() 
                self.arm.set_height(ArmHeight.HANOI_PREPARE)
            
            if self.grab_counter > delay:
                self.grab_state = 1
                self.grab_counter = 0

        elif self.grab_state == 1:
            if self.grab_counter == 1:
                self.arm.set_height(ArmHeight.FRONT_FLOOR)
            
            if self.grab_counter > delay:
                self.grab_state = 2
                self.grab_counter = 0

        elif self.grab_state == 2:
            if self.grab_counter == 1:
                self.gripper.grip() 
            
            if self.grab_counter > delay:
                self.grab_state = 3
                self.grab_counter = 0

        elif self.grab_state == 3:
            if self.grab_counter == 1:
                self.arm.set_height(ArmHeight.HANOI_PREPARE)
                self.arm.set_height(ArmHeight.RESET)
            
            if self.grab_counter > delay:
                return True
        
        return False
    
    def soltar_cubo(self):
        delay = 100
        self.drop_counter += 1
        if self.drop_state == 0:
            if self.drop_counter == 1:
                self.arm.set_height(ArmHeight.HANOI_PREPARE) 
            if self.drop_counter > delay:
                self.drop_state = 1
                self.drop_counter = 0

        elif self.drop_state == 1:
            if self.drop_counter == 1:
                self.arm.set_sub_arm_rotation(1, 0.0) 
                self.arm.set_sub_arm_rotation(2, -0.5)
                self.arm.set_sub_arm_rotation(3, -1.0)
            if self.drop_counter > delay:
                self.drop_state = 2
                self.drop_counter = 0

        elif self.drop_state == 2:
            if self.drop_counter == 1:
                self.arm.set_height(ArmHeight.FRONT_CARDBOARD_BOX) 
            if self.drop_counter > delay:
                self.drop_state = 3
                self.drop_counter = 0

        elif self.drop_state == 3:
            if self.drop_counter == 1:
                self.gripper.release()
            if self.drop_counter > delay:
                self.drop_state = 4
                self.drop_counter = 0

        elif self.drop_state == 4:
            if self.drop_counter == 1:
                self.arm.set_height(ArmHeight.RESET) 
            if self.drop_counter > delay:
                return True

        return False
            
    def change_state(self, state):
        self.estado_atual = state
        print(self.estado_atual)
        self.base.reset()
    
    def couter(self):
        self.contagem_inicial +=1
        if self.contagem_inicial >= 15:
            return True
        else:
            return False


    # --- Loop Principal ---
    def run(self):
        # get_pictures() # Usar em conjunto com supervisor.py para capturar imagens

        while self.robot.step(self.time_step) != -1:
            self.measure_lidar()
            self.detect()

            self.display_overlays()

            # --- ESTADO: PROCURANDO CUBO ---
            if self.estado_atual == Estados.INICIAL:
                self.change_state(Estados.PROCURA_CUBO)
            elif self.estado_atual == Estados.PROCURA_CUBO:
                self.passeio()
                if self.nearest_cube():
                    self.change_state(Estados.APROXIMA_CUBO)
                    
            elif self.estado_atual == Estados.APROXIMA_CUBO:
                if not self.cube_timeout():
                    self.aprox()
                    if self.target_cube.distance <= 0.1:
                        self.change_state(Estados.ALINHA_CUBO)
                else:
                    self.target_cube = None
                    self.change_state(Estados.PROCURA_CUBO)
            elif self.estado_atual == Estados.ALINHA_CUBO:
                if not self.cube_timeout():
                    self.alinha_cubo()
                    if self.cubo_alinhado():
                        self.change_state(Estados.AGARRA_CUBO)
                else:
                    self.change_state(Estados.PROCURA_CUBO)
            elif self.estado_atual == Estados.AGARRA_CUBO:
                if self.agarra_cubo():
                    self.change_state(Estados.PROCURA_CAIXA)
            elif self.estado_atual == Estados.PROCURA_CAIXA:
                self.passeio()
                if self.find_basket(self.target_cube):
                    self.change_state(Estados.APROXIMA_CAIXA)
            elif self.estado_atual == Estados.APROXIMA_CAIXA:
                self.aprox()
                self.change_state = Estados.SOLTA_CUBO
            elif self.estado_atual == Estados.SOLTA_CUBO:
                self.soltar_cubo()
                if self.counter():
                    self.change_state(Estados.FINAL)
                else:
                    self.change_state(Estados.PROCURA_CUBO)
            elif self.estado_atual == Estados.FINAL:
                print("Missão Cumprida! O robô venceu.")

            
if __name__ == "__main__":
    c = YouBotController()
    c.run()