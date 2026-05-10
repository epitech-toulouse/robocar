# MVP Et Roadmap

## Objectif MVP

Le MVP doit prouver que l'application Android sait:

1. afficher la camera
2. analyser les frames en temps reel
3. detecter la piste
4. calculer une direction exploitable
5. afficher un overlay de debug
6. produire une commande commandeable par le robot

## Contenu du MVP

### Etape 1 - socle Android

- creer le projet Kotlin Android
- configurer le package
- ajouter CameraX
- preparer l'UI principale

### Etape 2 - preview camera

- afficher la camera arriere
- verifier la rotation
- verifier les performances temps reel

### Etape 3 - analyse CameraX

- ajouter `ImageAnalysis`
- traiter les frames sur thread dedie
- fermer chaque `ImageProxy`

### Etape 4 - integration OpenCV

- charger OpenCV au demarrage
- convertir les frames en `Mat`
- valider une frame de debug

### Etape 5 - pipeline vision

- ROI basse
- gris ou HSV
- blur
- canny
- hough
- filtrage des lignes

### Etape 6 - logique direction

- calculer le centre de piste
- calculer l'offset
- transformer l'offset en direction

### Etape 7 - overlay debug

- dessiner ROI
- dessiner lignes
- dessiner centres
- afficher direction / offset / confiance

### Etape 8 - commandes

- commencer par logguer les commandes
- ajouter un format texte stable
- brancher USB serie ensuite

## Seuils de direction proposes

```kotlin
class DirectionController {

    fun computeDirection(result: DetectionResult): Direction {
        if (result.confidence < 0.45f || result.offset == null) {
            return Direction.LOST
        }

        return when {
            result.offset < -0.15f -> Direction.LEFT
            result.offset < -0.05f -> Direction.SLIGHT_LEFT
            result.offset > 0.15f -> Direction.RIGHT
            result.offset > 0.05f -> Direction.SLIGHT_RIGHT
            else -> Direction.FORWARD
        }
    }
}
```

## Lissage recommande

Pour eviter les oscillations:

```text
offset_lisse = 0.7 * ancien_offset + 0.3 * nouvel_offset
```

Et cote direction:

- n'accepter un changement qu'apres 2 ou 3 frames coherentes
- ou maintenir la derniere bonne valeur en cas de perte courte

## Plan de developpement concret

1. creer le projet Android Kotlin
2. ajouter CameraX
3. afficher l'apercu camera
4. ajouter OpenCV Android
5. convertir les frames CameraX en `Mat`
6. implementer ROI + Canny + Hough
7. dessiner l'overlay
8. calculer centre et direction
9. ajouter le lissage
10. logger les commandes
11. ajouter USB serie
12. tester sur piste reelle
13. ajuster les seuils

## Version robuste apres MVP

Une fois le MVP valide, les ameliorations prioritaires sont:

1. calibration HSV ou seuils adaptatifs
2. transformation de perspective vue du dessus
3. estimation plus robuste d'une ligne manquante
4. suivi temporel de ligne
5. score de confiance avance
6. mode debug avec vue intermediaire Canny / ROI
7. sauvegarde des parametres
8. BLE comme canal secondaire
9. commande continue type `STEER` et `SPEED`

## Evolution controle

Au lieu de rester sur:

```text
LEFT
RIGHT
FORWARD
```

on pourra evoluer vers:

```text
STEER:-0.35;SPEED:0.40\n
```

Cela permettra:

- un pilotage plus fin
- moins d'oscillations
- une meilleure compatibilite avec une boucle PID cote robot

## Risques techniques a surveiller

- latence trop forte sur certains telephones
- rotation camera / orientation mal calibree
- conversion YUV -> Mat couteuse
- forte sensibilite a la lumiere
- faux positifs Hough sur textures du sol
- saturation d'envoi des commandes

## Strategie de validation terrain

1. valider le pipeline en interieur avec affichage debug
2. enregistrer les offsets et directions
3. verifier la stabilite avec une piste simple
4. brancher le robot a vitesse faible
5. ajuster les seuils de confiance
6. introduire progressivement le controle continu
