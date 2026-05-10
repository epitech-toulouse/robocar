# Architecture Android

## Objectif global

L'application Android doit effectuer une boucle de perception et de commande en temps reel:

```text
Camera
  -> acquisition image
  -> traitement OpenCV
  -> detection des lignes
  -> calcul centre piste
  -> calcul offset
  -> determination direction
  -> envoi commande robot
  -> affichage overlay et etat
```

## Choix techniques

- **Langage**: Kotlin
- **UI**: Jetpack Compose recommande pour la rapidite d'iteration
- **Camera**: CameraX
- **Vision**: OpenCV Android SDK
- **Communication USB**: Android USB Host API + driver serie
- **Communication BLE**: Android Bluetooth LE API
- **Architecture logicielle**: MVVM simple
- **Concurrence**: Kotlin Coroutines
- **Version Android cible**: Android 8+ minimum, Android 10+ recommande

## Vision d'architecture

Le projet peut etre decoupe en couches bien separees:

1. **UI**
   - affiche la camera
   - dessine l'overlay
   - montre la direction, l'offset, la confiance et l'etat de connexion

2. **Camera**
   - ouvre la camera
   - gere Preview et ImageAnalysis
   - transmet les frames a l'analyseur

3. **Vision**
   - convertit les images CameraX en `Mat`
   - applique le pipeline OpenCV
   - estime les lignes, le centre de piste et la confiance

4. **Control**
   - transforme le resultat de vision en direction robot
   - applique le lissage temporel
   - formate les commandes

5. **Communication**
   - encapsule USB et BLE derriere une meme interface
   - gere etat de connexion, envoi, reprise et erreurs

6. **Config**
   - centralise les seuils, ROI, commandes, debugs et parametres camera

## Architecture MVVM simple

### View

La couche View correspond aux composants Compose ou XML:

- `MainActivity`
- `RobocarScreen`
- `CameraPreview`
- `DetectionOverlay`
- `ControlPanel`

Elle observe un `ViewModel` contenant l'etat courant:

- direction
- offset
- confiance
- etat de piste
- etat de connexion
- mode de transport actif

### ViewModel

Le `ViewModel` orchestre:

- le demarrage et l'arret de la camera
- le routage des resultats de vision
- le calcul de la direction
- l'envoi des commandes
- l'exposition d'un `UiState`

Il ne contient pas les details OpenCV ni les details USB/BLE.

### Domain / Processing

La logique metier reside dans:

- `OpenCVProcessor`
- `LineDetector`
- `DirectionController`
- `DirectionSmoother`
- `CommandFormatter`

Cette couche doit rester testable sans UI.

### Data / I/O

La couche I/O regroupe:

- `CameraManager`
- `UsbCommunicationManager`
- `BleCommunicationManager`

Elle interagit avec les API Android et les bibliotheques externes.

## Flux principal

```text
MainActivity
  -> demande permissions
  -> initialise UI + ViewModel
  -> lance CameraManager

CameraManager
  -> ouvre CameraX
  -> attache Preview
  -> attache ImageAnalysis

FrameAnalyzer
  -> recoit ImageProxy
  -> appelle OpenCVProcessor

OpenCVProcessor
  -> convertit en Mat
  -> applique ROI + Canny + Hough
  -> retourne DetectionResult

DirectionController
  -> convertit DetectionResult en Direction

CommandTransport
  -> envoie commande en USB ou BLE

UI
  -> affiche preview, overlay, direction et connexions
```

## Responsabilites par composant

### MainActivity

- initialise OpenCV
- demande les permissions runtime
- heberge l'ecran principal
- branche le cycle de vie Android

### CameraManager

- configure `Preview`
- configure `ImageAnalysis`
- regle la resolution et la strategie de backpressure
- attache la camera au lifecycle

### FrameAnalyzer

- recoit chaque `ImageProxy`
- evite d'accumuler les frames
- appelle le processeur OpenCV sur thread dedie
- referme correctement `ImageProxy`

### OpenCVProcessor

- convertit YUV en RGB ou Gray
- applique rotation si necessaire
- extrait la ROI
- lance detection de lignes
- produit un `DetectionResult`

### DirectionController

- applique les seuils sur l'offset normalise
- produit une direction discrete
- degrade vers `LOST` si la confiance est insuffisante

### DirectionSmoother

- lisse les variations d'offset
- limite les changements trop rapides de direction

### CommandTransport

- expose une interface unique pour USB et BLE
- permet de basculer de transport sans changer le coeur de l'application

## Etat UI conseille

Un `UiState` simple peut contenir:

```kotlin
data class RobocarUiState(
    val isRunning: Boolean = false,
    val direction: Direction = Direction.LOST,
    val offset: Float? = null,
    val confidence: Float = 0f,
    val trackState: TrackState = TrackState.LOST,
    val connectionState: ConnectionState = ConnectionState.Disconnected,
    val transportMode: TransportMode = TransportMode.USB,
    val lastDetection: DetectionResult? = null,
    val debugEnabled: Boolean = true
)
```

## Regles de conception

- isoler les appels Android dans les managers
- garder la logique de vision et de controle testable
- eviter le couplage direct UI <-> transport
- garder des objets de configuration centraux
- prioriser des structures simples avant une architecture plus lourde

## Priorites de la premiere iteration

1. obtenir un flux CameraX stable
2. faire tourner OpenCV sur ROI basse
3. afficher un overlay fiable
4. produire une direction lisible
5. simuler l'envoi de commande
6. brancher USB serie
