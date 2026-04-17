# Robocar Architecture Overview

Ce document présente l'architecture matérielle et logicielle du projet Robocar. L'objectif de la refonte récente était de séparer l'algorithme d'évitement d'obstacles (Core Logic) de l'implémentation matérielle (ESP32) et logicielle (Simulation ROS 2 / Gazebo), en utilisant des interfaces communes.

## 1. Architecture de la Simulation (ROS 2 + Gazebo)

Dans cet environnement, le code tourne nativement sur l'ordinateur de développement (Linux). Les périphériques physiques sont remplacés par des nœuds ROS et des plugins Gazebo.

```mermaid
graph TD
    subgraph ROS2_Environment ["ROS 2 Node (RobocarSimController)"]
        AppNode[robocar_sim_controller_node.cpp]
        
        subgraph Virtual_Hardware_Abstraction
            SimLidar[SimLidarSensor]
            SimVesc[SimVescController]
        end
        
        subgraph Core_Algorithm
            Algo[LidarObstacleAvoidanceAlgorithm]
        end
        
        AppNode --> SimLidar
        AppNode --> SimVesc
        AppNode --> Algo
        
        SimLidar -- "std::vector<Point>" --> Algo
        Algo -- "Speed, Steering angle" --> SimVesc
        
        SimVesc -- "geometry_msgs::msg::Twist" --> TopicCmdVel[(/cmd_vel)]
        TopicScan[(/scan)] -- "sensor_msgs::msg::LaserScan" --> SimLidar
    end

    subgraph Middleware
        Bridge[ros_gz_bridge]
    end

    subgraph Gazebo_Environment ["Gazebo Simulation (robocar_empty.sdf)"]
        GzDiffDrive[gz::sim::systems::DiffDrive]
        GzLidar[GPU Lidar Sensor]
        
        GzTopicCmdVel[(/cmd_vel)]
        GzTopicScan[(/scan)]
        
        GzTopicCmdVel -- "gz.msgs.Twist" --> GzDiffDrive
        GzLidar -- "gz.msgs.LaserScan" --> GzTopicScan
    end
    
    TopicCmdVel <--> Bridge
    Bridge <--> GzTopicCmdVel
    
    TopicScan <--> Bridge
    Bridge <--> GzTopicScan
```

---

## 2. Architecture Réelle (Embarquée sur ESP32)

Sur l'appareil réel, le firmware s'exécute sur l'ESP32 via FreeRTOS. Le code interagit directement avec les périphériques I2C, SPI, UART, PWM, et BLE.

```mermaid
graph TD
    subgraph ESP32_Firmware ["ESP32 Firmware (main.cpp)"]
        MainTask[Main FreeRTOS Task]
        
        subgraph Hardware_Drivers
            LidarDriver[LidarReader : UART]
            VescDriver[VescController : UART/PWM]
            BTDriver[BluetoothReceiver : BLE]
        end
        
        subgraph Core_Algorithm
            Algo[LidarObstacleAvoidanceAlgorithm]
        end
        
        MainTask --> LidarDriver
        MainTask --> VescDriver
        MainTask --> BTDriver
        MainTask --> Algo
        
        LidarDriver -- "std::vector<Point>" --> Algo
        BTDriver -- "User commands" --> MainTask
        Algo -- "Duty Cycle / Servo PWM" --> VescDriver
    end

    subgraph Physical_Components ["Physical Hardware"]
        PhysicalLidar[Physical Lidar - LD06/RPLidar]
        PhysicalVesc[Physical VESC Motor Controller]
        MobileApp[Mobile Phone App]
        
        PhysicalLidar -- UART RX --> LidarDriver
        VescDriver -- UART TX / PWM --> PhysicalVesc
        MobileApp -- Bluetooth LE --> BTDriver
    end
```

---

## 3. Architecture Unifiée (Dépendances et Interfaces)

Au cœur du système se trouve le principe d'Inversion de Dépendance (DIP - SOLID). Le `LidarObstacleAvoidanceAlgorithm` ne connaît pas l'ESP32 ni ROS2. Il s'adresse uniquement à des interfaces neutres abstraites (API).

```mermaid
classDiagram
    %% Core Interfaces (main/api/)
    class DrivingAlgorithmInterface {
        <<interface>>
        +compute_driving_command()
    }
    class LidarSensorAPI {
        <<interface>>
        +get_scan_data() vector~LidarPoint~
    }
    class VescControllerAPI {
        <<interface>>
        +set_speed(speed)
        +set_steering(angle)
    }

    %% The single source of truth for intelligence
    class LidarObstacleAvoidanceAlgorithm {
        -LidarSensorAPI* lidar
        -VescControllerAPI* vesc
        +compute_driving_command()
    }
    
    DrivingAlgorithmInterface <|.. LidarObstacleAvoidanceAlgorithm

    LidarObstacleAvoidanceAlgorithm --> LidarSensorAPI : Demande les points
    LidarObstacleAvoidanceAlgorithm --> VescControllerAPI : Envoie les commandes
    
    %% Real Implementations (ESP32)
    namespace ESP32_Hardware {
        class LidarReader {
            +UART setup()
        }
        class VescController {
            +PWM/UART setup()
        }
    }
    
    LidarSensorAPI <|.. LidarReader
    VescControllerAPI <|.. VescController

    %% Simulated Implementations (ROS 2)
    namespace ROS2_Gazebo {
        class SimLidarSensor {
            +ROS Subscription()
        }
        class SimVescController {
            +ROS Publisher()
        }
    }
    
    LidarSensorAPI <|.. SimLidarSensor
    VescControllerAPI <|.. SimVescController
```

### Bénéfices de ce modèle commun
- **Portabilité :** Le même algorithme (`LidarObstacleAvoidanceAlgorithm`) est cloné et testé en simulation sans modification de son code source.
- **Mocking pour les tests :** Possibilité de créer des faux Lidars (des Mocks) qui injectent des obstacles virtuels pour valider unitairement l'algorithme.
- **Séparation des préoccupations :** Les bugs de ROS 2 ou de pont Gazebo n'impacteront jamais le code qui tourne physiquement sur l'ESP32.
