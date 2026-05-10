# Communication USB Et BLE

## Objectif

L'application doit pouvoir envoyer des commandes de direction au robot en utilisant:

- USB serie pour le premier prototype
- BLE comme alternative sans fil

L'architecture recommande d'abstraire le transport pour que le reste du code ne depenne pas du canal choisi.

## Interface commune

```kotlin
interface CommandTransport {
    fun connect()
    fun disconnect()
    fun send(command: String)
    val isConnected: Boolean
}
```

Cette interface permet au `ViewModel` ou au `DirectionController` de rester independants du transport reel.

## Format de commande

### Format minimal

```text
FORWARD\n
LEFT\n
RIGHT\n
SLIGHT_LEFT\n
SLIGHT_RIGHT\n
LOST\n
```

### Format debug recommande

```text
CMD:FORWARD\n
CMD:LEFT\n
CMD:RIGHT\n
CMD:SLIGHT_LEFT\n
CMD:SLIGHT_RIGHT\n
CMD:LOST\n
```

### Format avance

```text
DIR:LEFT;OFFSET:-0.24;CONF:0.76\n
```

Le format avance est tres utile pour tracer le comportement du robot et enrichir le controle plus tard.

## USB serie sur Android

## Pourquoi USB

Android est tres adapte au prototypage USB serie:

- USB Host API accessible
- adaptateur USB-C OTG courant
- compatibilite avec Arduino, ESP32, STM32, Pico
- faible latence
- debuggage simple

## Bibliotheque conseillee

La bibliotheque `usb-serial-for-android` est pratique pour gerer:

- CH340
- CP210x
- FTDI
- CDC ACM

## Responsabilites de `UsbCommunicationManager`

- detecter les peripheriques USB compatibles
- demander la permission utilisateur USB
- ouvrir le port serie
- configurer le baud rate
- envoyer les commandes texte
- fermer proprement la connexion

## Permissions et manifest

Le manifest doit contenir:

```xml
<uses-feature
    android:name="android.hardware.usb.host"
    android:required="false" />
```

Un filtre USB peut etre ajoute dans `res/xml/device_filter.xml` selon le materiel cible.

## Parcours typique USB

1. detection du peripherique branche
2. demande de permission Android USB
3. ouverture du driver
4. ouverture du port
5. configuration serie
6. envoi de commande a chaque mise a jour utile

## BLE sur Android

## Pourquoi BLE

BLE est interessant si:

- on veut supprimer le cable
- le robot utilise un ESP32
- on veut un prototype mobile et compact

## Architecture BLE recommandee

- Android agit comme **central**
- ESP32 agit comme **peripheral**
- service UUID custom
- characteristic de type write

## Responsabilites de `BleCommunicationManager`

- verifier Bluetooth actif
- scanner les peripheriques
- reconnaitre le bon robot
- se connecter en GATT
- resoudre service et characteristic
- envoyer les commandes
- exposer l'etat de connexion a l'UI

## Permissions BLE

Pour Android moderne:

```xml
<uses-permission android:name="android.permission.BLUETOOTH_SCAN" />
<uses-permission android:name="android.permission.BLUETOOTH_CONNECT" />
```

Il faudra gerer les permissions runtime selon la version Android cible.

## Etat de connexion

Un modele d'etat simple peut suffire:

```kotlin
sealed class ConnectionState {
    data object Disconnected : ConnectionState()
    data object Connecting : ConnectionState()
    data object Connected : ConnectionState()
    data class Error(val message: String) : ConnectionState()
}
```

## Strategie d'envoi des commandes

Pour eviter de saturer le transport:

- ne pas envoyer a chaque frame si la direction n'a pas change
- ou limiter la frequence d'envoi
- ou envoyer uniquement si l'offset varie au-dela d'un seuil

Recommandation MVP:

- calcul de direction a chaque frame analysee
- envoi si direction differente de la precedente
- envoi periodique de garde si necessaire

## Interface utilisateur

L'UI doit rendre visibles:

- le mode de transport selectionne
- l'etat de connexion
- le dernier message envoye
- les erreurs de connexion

## Ordre recommande de mise en oeuvre

1. mock local sans transport reel
2. transport USB reel
3. BLE optionnel
4. format de commande enrichi
5. reprise de connexion et robustesse
