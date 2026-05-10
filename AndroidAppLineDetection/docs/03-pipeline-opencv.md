# Pipeline OpenCV

## But du pipeline

Le pipeline vision doit detecter les deux lignes de la piste dans la partie basse de l'image, puis calculer le centre de piste et l'offset par rapport au centre de l'image.

## Pipeline MVP recommande

1. reception d'une frame CameraX
2. conversion `YUV_420_888` vers `Mat`
3. rotation si necessaire
4. decoupage de la ROI basse
5. conversion en gris ou HSV
6. `GaussianBlur`
7. `Canny`
8. `HoughLinesP`
9. filtrage des segments
10. separation gauche / droite
11. selection des meilleurs candidats
12. calcul du centre de piste
13. calcul de l'offset
14. calcul de la confiance
15. emission du `DetectionResult`

## ROI

Pour limiter le bruit et accelerer le traitement, on ignore la partie haute de l'image.

```text
Image complete
+----------------------+
|                      |
|      ignore          |
|                      |
|----------------------|
|                      |
|   zone analysee      |
|   lignes au sol      |
+----------------------+
```

Exemple:

```kotlin
val roiStartY = (imageHeight * 0.55f).toInt()
val roiHeight = imageHeight - roiStartY
```

## Conversion CameraX vers OpenCV

La camera Android fournit souvent des images au format `YUV_420_888`.

Le processeur doit:

1. lire les plans Y, U et V
2. reconstruire l'image dans un format compatible OpenCV
3. convertir vers RGB ou Gray

Pour le MVP, travailler directement en niveaux de gris peut simplifier la pipeline.

## Pre-traitement

### Gris ou HSV

- **Gris**: simple et rapide pour une premiere version
- **HSV**: utile si les lignes ont une couleur differenciee du fond

### Flou gaussien

Le flou gaussien reduit le bruit avant Canny:

```kotlin
Imgproc.GaussianBlur(input, blurred, Size(5.0, 5.0), 0.0)
```

### Canny

Permet d'extraire les contours:

```kotlin
Imgproc.Canny(blurred, edges, lowThreshold, highThreshold)
```

## Detection par HoughLinesP

`HoughLinesP` renvoie plusieurs segments candidats:

```kotlin
Imgproc.HoughLinesP(
    edges,
    lines,
    1.0,
    Math.PI / 180,
    threshold,
    minLineLength,
    maxLineGap
)
```

## Filtrage des segments

Pour chaque segment:

1. calculer la longueur
2. calculer la pente
3. ignorer les segments trop courts
4. ignorer les segments trop horizontaux
5. evaluer la position dans la ROI

Exemple de regles:

- longueur minimale
- pente absolue > seuil minimal
- point bas suffisamment proche du bas de la ROI

## Separation gauche / droite

Approche MVP:

- pente negative -> candidat gauche
- pente positive -> candidat droite

Attention: cela depend de l'orientation finale de l'image apres rotation. Il faut donc calibrer ce point sur appareil reel.

## Choix des meilleures lignes

Parmi les candidats, garder ceux qui maximisent un score combinant:

- longueur
- pente utile
- proximite avec le bas de l'image
- stabilite par rapport a la frame precedente

Un score simple peut suffire au debut:

```text
score = longueur * 0.5
      + proximite_bas * 0.3
      + stabilite_temporelle * 0.2
```

## Calcul du centre de piste

### Cas nominal: deux lignes detectees

On projette chaque ligne sur le bas de la ROI pour obtenir:

- `leftBottomX`
- `rightBottomX`

Puis:

```kotlin
val trackCenterX = (leftBottomX + rightBottomX) / 2f
val imageCenterX = imageWidth / 2f
val offset = trackCenterX - imageCenterX
val normalizedOffset = offset / imageWidth
```

### Cas degrade: une seule ligne detectee

Si une seule ligne est fiable:

- ligne gauche detectee -> estimer la ligne droite avec une largeur de piste moyenne
- ligne droite detectee -> estimer la ligne gauche

L'etat remonte alors en `ONE_LINE_ESTIMATED`.

### Cas perdu

Si aucune ligne fiable n'est disponible:

- `trackCenterX = null`
- `offset = null`
- `state = LOST`

## Calcul de confiance

La confiance peut combiner:

- presence de deux lignes
- longueur moyenne des segments
- coherence geometrique entre les deux lignes
- proximite du bas de la ROI
- stabilite inter-frame

Exemple d'heuristique:

- deux lignes solides -> `0.75` a `1.0`
- une ligne estimee -> `0.45` a `0.7`
- detection fragile -> `< 0.45`

## Resultat de sortie

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

## Debug visuel recommande

Le mode debug doit permettre d'afficher:

- ROI
- segments bruts Hough
- ligne gauche retenue
- ligne droite retenue
- centre image
- centre piste
- texte offset / confiance / etat

## Ameliorations futures

1. calibration HSV
2. transformation de perspective
3. suivi temporel des lignes
4. approche par contours
5. sliding windows
6. moyenne glissante ou Kalman
7. ajustement auto des seuils
