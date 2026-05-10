# Structure Des Fichiers

## Arborescence recommandee

```text
RobocarLineDetection/
  app/
    src/main/
      java/com/robocar/linedetection/
        MainActivity.kt
        RobocarViewModel.kt

        camera/
          CameraManager.kt
          FrameAnalyzer.kt

        vision/
          OpenCVProcessor.kt
          LineDetector.kt
          DetectionResult.kt
          DetectedLine.kt
          TrackState.kt

        control/
          Direction.kt
          DirectionController.kt
          DirectionSmoother.kt
          CommandFormatter.kt

        communication/
          CommandTransport.kt
          UsbCommunicationManager.kt
          BleCommunicationManager.kt
          ConnectionState.kt
          TransportMode.kt

        ui/
          RobocarScreen.kt
          CameraPreview.kt
          DetectionOverlay.kt
          ControlPanel.kt
          UiState.kt

        config/
          DetectionConfig.kt
          AppConfig.kt

        utils/
          PermissionManager.kt
          Logger.kt

      res/
        xml/
          device_filter.xml

      AndroidManifest.xml
```

## Detail par package

### Racine applicative

#### `MainActivity.kt`

Point d'entree Android:

- initialise l'application
- verifie OpenCV
- demande permissions
- affiche l'UI

#### `RobocarViewModel.kt`

Coordonne les couches internes:

- recoit les resultats du pipeline camera/vision
- calcule la direction
- met a jour l'etat UI
- declenche l'envoi des commandes

### `camera/`

#### `CameraManager.kt`

Responsable de:

- configurer CameraX
- lancer `Preview`
- lancer `ImageAnalysis`
- selectionner la camera arriere
- choisir resolution et rotation

#### `FrameAnalyzer.kt`

Responsable de:

- lire `ImageProxy`
- transmettre les frames a `OpenCVProcessor`
- exposer le resultat au `ViewModel`

### `vision/`

#### `OpenCVProcessor.kt`

Pipeline haut niveau:

- conversion image CameraX -> `Mat`
- preparation des donnees
- appel de `LineDetector`
- construction du `DetectionResult`

#### `LineDetector.kt`

Detection pure:

- blur
- canny
- hough
- selection de lignes gauche / droite
- calcul du centre et de la confiance

#### `DetectionResult.kt`

Structure de sortie principale:

```kotlin
data class DetectionResult(
    val leftLine: DetectedLine?,
    val rightLine: DetectedLine?,
    val trackCenterX: Float?,
    val imageCenterX: Float,
    val offset: Float?,
    val confidence: Float,
    val state: TrackState
)
```

#### `DetectedLine.kt`

Represente une ligne retenue:

```kotlin
data class DetectedLine(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val confidence: Float
)
```

#### `TrackState.kt`

```kotlin
enum class TrackState {
    TWO_LINES_DETECTED,
    ONE_LINE_ESTIMATED,
    LOST
}
```

### `control/`

#### `Direction.kt`

```kotlin
enum class Direction {
    FORWARD,
    LEFT,
    RIGHT,
    SLIGHT_LEFT,
    SLIGHT_RIGHT,
    LOST
}
```

#### `DirectionController.kt`

Convertit `DetectionResult` en `Direction`.

#### `DirectionSmoother.kt`

Applique:

- moyenne glissante
- hysteresis
- validation sur plusieurs frames

#### `CommandFormatter.kt`

Convertit une direction ou une commande riche en texte:

- `CMD:LEFT\n`
- `DIR:LEFT;OFFSET:-0.24;CONF:0.76\n`

### `communication/`

#### `CommandTransport.kt`

Interface commune:

```kotlin
interface CommandTransport {
    fun connect()
    fun disconnect()
    fun send(command: String)
    val isConnected: Boolean
}
```

#### `UsbCommunicationManager.kt`

Responsable de:

- detection du peripherique
- demande de permission USB
- ouverture du port serie
- envoi des commandes

#### `BleCommunicationManager.kt`

Responsable de:

- scan BLE
- connexion GATT
- resolution du service et de la characteristic
- ecriture de commandes

#### `ConnectionState.kt`

Etat de connexion exploitable par l'UI.

#### `TransportMode.kt`

Selectionne le canal actif:

- USB
- BLE
- NONE

### `ui/`

#### `RobocarScreen.kt`

Assemble l'ecran principal:

- preview camera
- overlay
- panneau de controle
- indicateurs d'etat

#### `CameraPreview.kt`

Pont entre CameraX `PreviewView` et Compose.

#### `DetectionOverlay.kt`

Dessine:

- ROI
- lignes detectees
- centre image
- centre piste
- texte debug

#### `ControlPanel.kt`

Boutons et indicateurs:

- start / stop
- usb / ble
- etat connexion
- direction

#### `UiState.kt`

Contient les donnees observees par l'ecran.

### `config/`

#### `DetectionConfig.kt`

Centralise les seuils de vision:

- position ROI
- seuil Canny
- longueur minimale de ligne
- plage de pente
- largeur piste attendue
- seuils de confiance

#### `AppConfig.kt`

Centralise:

- mode debug
- frequence d'envoi
- format de commande
- resolution camera souhaitee

### `utils/`

#### `PermissionManager.kt`

Regroupe la logique runtime pour:

- camera
- bluetooth
- eventuels acces USB selon parcours UX choisi

#### `Logger.kt`

Wrapper leger autour des logs Android pour garder une sortie propre.

## Fichiers Android de support

### `AndroidManifest.xml`

Contient:

- permissions camera
- permissions bluetooth
- feature USB host
- declaration de l'activite principale

### `res/xml/device_filter.xml`

Permet de filtrer les peripheriques USB supportes si necessaire.

## Convention de dependances

Pour le MVP, il est sain de garder:

- peu de packages
- peu d'abstractions
- des classes petites et lisibles
- des DTO explicites pour les echanges inter-couches
