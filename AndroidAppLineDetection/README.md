# Robocar Line Detection - Documentation Technique

Ce depot contient la base documentaire du projet Android de suivi de ligne pour robot mobile.

## Objectif

Construire une application Android en Kotlin capable de:

1. lire un flux camera en temps reel
2. detecter les lignes de piste avec OpenCV
3. calculer la direction optimale
4. afficher un overlay de debug
5. envoyer les commandes au robot via USB serie ou BLE

## Documentation

- [Architecture Android](./docs/01-architecture-android.md)
- [Structure des fichiers](./docs/02-structure-fichiers.md)
- [Pipeline OpenCV](./docs/03-pipeline-opencv.md)
- [Communication USB et BLE](./docs/04-communication-usb-ble.md)
- [MVP et roadmap](./docs/05-mvp-et-roadmap.md)

## Stack recommandee

- Kotlin
- CameraX
- OpenCV Android SDK
- Jetpack Compose ou UI XML
- Coroutines
- MVVM simple
- USB Host API Android
- Bluetooth LE API Android

## Boucle temps reel cible

```text
Camera
  -> frame image
  -> OpenCV
  -> detection des deux lignes
  -> calcul centre piste
  -> calcul direction
  -> envoi commande USB ou BLE
  -> affichage overlay
```

## Priorite de prototypage

Le premier prototype recommande est:

1. CameraX Preview
2. CameraX ImageAnalysis
3. OpenCV avec ROI basse
4. Detection HoughLinesP
5. Overlay visuel
6. Direction locale
7. Simulation de commandes
8. USB serie reel

## Suite

Une fois la documentation en place, l'etape naturelle est de scaffold le projet Android Kotlin puis d'implementer le MVP module par module.
