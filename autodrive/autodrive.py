import time
import math
from control.State import State
from control.Motor import Motor
from control.Gamepad import Gamepad
from .LidarParserUDP import LidarParserUDP

# =============================================================================
# Follow-the-Gap AutoDrive
#
# Principe : on discrétise le champ de vision avant (-90° à +90°) en N bins
# angulaires.  Pour chaque bin on retient la distance minimale.  On applique
# une "bulle de sécurité" autour des obstacles proches, puis on cherche la
# séquence consécutive de bins libres la plus large (= le plus grand gap).
# On braque vers le centre de ce gap.  Le tout est lissé pour être fluide.
# =============================================================================

# -- Scan --
NUM_BINS = 72              # Résolution angulaire (180° / 72 = 2.5° par bin)
SCAN_HALF_FOV = 90.0       # Demi-champ de vision avant (°)
BUBBLE_RADIUS = 0.45       # Rayon de la bulle de sécurité autour des obstacles (m)
BUBBLE_ANGLE_DEG = 15.0    # Demi-angle effacé autour d'un obstacle proche (°)

# -- Distances --
SAFE_DISTANCE = 3.5        # Au-delà = pleine vitesse
SLOW_DISTANCE = 1.5        # En-dessous = vitesse réduite
STOP_DISTANCE = 0.55       # En-dessous = marche arrière
MAX_RANGE = 10.0           # Distance max considérée (clip)

# -- Vitesses --
FORWARD_SPEED = 0.06
SLOW_SPEED = 0.04
BACKWARD_SPEED = -0.04

# -- Steering --
STEER_SMOOTHING = 0.35     # 0 = pas de lissage, 1 = figé (exponentiel)
STRAIGHT_BIAS = 0.15       # Bonus de score pour les gaps proches de 0° (favorise tout droit)

# -- Reverse --
REVERSE_CYCLES = 25


class AutoDriveState(State):
    def __init__(self, use_gps=False, gps_host='localhost'):
        print("Initializing AutoDrive (Follow-the-Gap)...")
        self.lidar = LidarParserUDP(host='127.0.0.1', port=8888)

        self.last_steering = 0.0
        self.reverse_timer = 0

        # GPS (conservé pour compatibilité)
        self.gps = None
        self.use_gps = use_gps

        # Pré-calcul des angles centraux de chaque bin
        self.bin_angles = [
            -SCAN_HALF_FOV + (i + 0.5) * (2 * SCAN_HALF_FOV / NUM_BINS)
            for i in range(NUM_BINS)
        ]
        self.bin_width = 2 * SCAN_HALF_FOV / NUM_BINS

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------
    def stop(self):
        self.lidar.stop()
        if self.gps:
            self.gps.stop()

    # -----------------------------------------------------------------
    # Main loop (appelé par le Manager)
    # -----------------------------------------------------------------
    def run_single(self, motor: Motor, gamepad: Gamepad):
        points = self.lidar.get_points()
        if not points:
            time.sleep(0.05)
            return

        # 1. Construire le tableau de distances par bin
        bins = self._build_distance_bins(points)

        # 2. Marche arrière en cours ?
        if self.reverse_timer > 0:
            self._handle_reverse(motor, bins)
            self.reverse_timer -= 1
            return

        # 3. Obstacle critique devant → marche arrière
        front_min = self._front_min_distance(bins)
        if front_min < STOP_DISTANCE:
            self._initiate_reverse(motor, bins)
            return

        # 4. Appliquer la bulle de sécurité
        safe_bins = self._apply_safety_bubble(list(bins))

        # 5. Trouver le meilleur gap
        target_angle = self._find_best_gap(safe_bins)

        # 6. Convertir en steering [-1, 1] avec lissage
        raw_steering = max(-1.0, min(1.0, target_angle / SCAN_HALF_FOV))
        smoothed = (STEER_SMOOTHING * self.last_steering
                    + (1 - STEER_SMOOTHING) * raw_steering)
        smoothed = max(-1.0, min(1.0, smoothed))
        self.last_steering = smoothed

        # 7. Vitesse adaptative (basée sur la distance min dans la direction choisie)
        look_min = self._distance_in_direction(bins, target_angle, 20)
        speed = self._calculate_speed(look_min)

        motor.set_steering_objective(smoothed)
        motor.set_speed_objective(speed)

    # =================================================================
    #  BINNING  – on projette les points lidar dans des bins angulaires
    # =================================================================
    def _build_distance_bins(self, points):
        """Retourne un tableau de NUM_BINS distances (la plus proche par bin)."""
        bins = [MAX_RANGE] * NUM_BINS
        for p in points:
            angle = self._normalize_angle(p['angle'])
            if -SCAN_HALF_FOV <= angle <= SCAN_HALF_FOV:
                idx = int((angle + SCAN_HALF_FOV) / (2 * SCAN_HALF_FOV) * NUM_BINS)
                idx = max(0, min(NUM_BINS - 1, idx))
                dist = min(p['distance'], MAX_RANGE)
                if dist < bins[idx]:
                    bins[idx] = dist
        return bins

    # =================================================================
    #  SAFETY BUBBLE  – on efface les bins autour des obstacles proches
    # =================================================================
    def _apply_safety_bubble(self, bins):
        """Met à 0 les bins dans un rayon angulaire autour de chaque obstacle
        plus proche que BUBBLE_RADIUS."""
        bubble_half_bins = max(1, int(BUBBLE_ANGLE_DEG / self.bin_width))
        for i, d in enumerate(bins):
            if d < BUBBLE_RADIUS:
                lo = max(0, i - bubble_half_bins)
                hi = min(NUM_BINS - 1, i + bubble_half_bins)
                for j in range(lo, hi + 1):
                    bins[j] = 0.0
        return bins

    # =================================================================
    #  FIND BEST GAP  – plus grande séquence consécutive de bins libres
    # =================================================================
    def _find_best_gap(self, bins):
        """Retourne l'angle (°) vers le centre du meilleur gap."""
        free_threshold = SLOW_DISTANCE * 0.8  # bin considéré libre au-dessus de ça

        # Trouver tous les runs consécutifs de bins libres
        gaps = []
        start = None
        for i in range(NUM_BINS):
            if bins[i] > free_threshold:
                if start is None:
                    start = i
            else:
                if start is not None:
                    gaps.append((start, i - 1))
                    start = None
        if start is not None:
            gaps.append((start, NUM_BINS - 1))

        if not gaps:
            # Aucun espace libre → viser le bin le plus lointain
            best_idx = max(range(NUM_BINS), key=lambda i: bins[i])
            return self.bin_angles[best_idx]

        # Scorer chaque gap : largeur × profondeur moyenne + bonus tout-droit
        best_score = -1
        best_angle = 0.0
        for (s, e) in gaps:
            width = e - s + 1
            avg_depth = sum(bins[s:e + 1]) / width
            center_idx = (s + e) // 2
            center_angle = self.bin_angles[center_idx]

            # Le score favorise les gaps larges, profonds, et proches de 0°
            straightness = 1.0 - abs(center_angle) / SCAN_HALF_FOV  # 1 si droit, 0 si ±90°
            score = width * avg_depth + STRAIGHT_BIAS * straightness * avg_depth

            if score > best_score:
                best_score = score
                best_angle = center_angle

        # Dans le gap choisi, on peut affiner en visant le point le plus profond
        # plutôt que le centre géométrique (meilleur résultat dans les couloirs)
        for (s, e) in gaps:
            center_idx = (s + e) // 2
            if abs(self.bin_angles[center_idx] - best_angle) < self.bin_width:
                deepest_idx = max(range(s, e + 1), key=lambda i: bins[i])
                # Pondérer entre centre géométrique et point le plus profond
                geo_angle = self.bin_angles[center_idx]
                deep_angle = self.bin_angles[deepest_idx]
                best_angle = 0.6 * geo_angle + 0.4 * deep_angle
                break

        return best_angle

    # =================================================================
    #  SPEED
    # =================================================================
    def _calculate_speed(self, min_distance):
        if min_distance >= SAFE_DISTANCE:
            return FORWARD_SPEED
        elif min_distance <= STOP_DISTANCE:
            return 0.0
        else:
            t = (min_distance - STOP_DISTANCE) / (SAFE_DISTANCE - STOP_DISTANCE)
            return SLOW_SPEED + (FORWARD_SPEED - SLOW_SPEED) * t

    def _distance_in_direction(self, bins, angle_deg, half_window_deg=15):
        """Distance minimale dans un cône autour de angle_deg."""
        lo = angle_deg - half_window_deg
        hi = angle_deg + half_window_deg
        min_d = MAX_RANGE
        for i, a in enumerate(self.bin_angles):
            if lo <= a <= hi:
                if bins[i] < min_d:
                    min_d = bins[i]
        return min_d

    def _front_min_distance(self, bins, half_angle=20):
        """Distance minimale dans le secteur avant ±half_angle°."""
        return self._distance_in_direction(bins, 0, half_angle)

    # =================================================================
    #  REVERSE
    # =================================================================
    def _initiate_reverse(self, motor: Motor, bins):
        self.reverse_timer = REVERSE_CYCLES

        # Trouver de quel côté il y a plus d'espace pour braquer en reculant
        mid = NUM_BINS // 2
        left_space = sum(bins[:mid]) / mid if mid > 0 else 0
        right_space = sum(bins[mid:]) / (NUM_BINS - mid) if (NUM_BINS - mid) > 0 else 0

        if left_space > right_space:
            steering = -0.8   # braquer à gauche en reculant → le nez ira à gauche
        elif right_space > left_space:
            steering = 0.8
        else:
            steering = 0.0

        motor.set_steering_objective(steering)
        motor.set_speed_objective(BACKWARD_SPEED)

    def _handle_reverse(self, motor: Motor, bins):
        """Continue la marche arrière, redresse vers la fin."""
        if self.reverse_timer < 8:
            motor.set_steering_objective(0.0)

    # =================================================================
    #  UTILS
    # =================================================================
    @staticmethod
    def _normalize_angle(angle):
        """Normalise un angle entre -180 et 180."""
        angle = angle % 360
        if angle > 180:
            angle -= 360
        return angle
