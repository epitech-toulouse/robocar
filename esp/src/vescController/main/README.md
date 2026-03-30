# VESC Controller + Wi-Fi (ESP32-S3)

Ce composant supporte:
- Pilotage automatique via LiDAR
- Override manuel via Wi-Fi (TCP)

## Transport Wi-Fi

- Mode Wi-Fi: SoftAP (l'ESP32 cree son propre reseau)
- SSID: `ROBOCAR_CTRL`
- Mot de passe: `robocar123`
- Port TCP: `3333`
- Payload: 1 caractere ASCII par evenement

Le protocole de commande est strictement identique a l'ancien BLE (meme chars, meme semantique):
- `F` / `f` : avancer press / release
- `B` / `b` : reculer press / release
- `L` / `l` : gauche press / release
- `R` / `r` : droite press / release
- `S` : stop urgence

Exemples payload:
- `F` puis `R` pour avancer + tourner a droite
- `r` pour arreter de tourner a droite
- `f` pour arreter d'avancer
- `S` pour arret immediat

Mapping interne actuel:
- `F` => duty `+0.05`
- `B` => duty `-0.05`
- `L` => steering `0.2`
- `R` => steering `0.8`
- centre steering `0.5`

## Build / Flash

Depuis la racine du projet ESP-IDF (dossier contenant `CMakeLists.txt` principal):

```bash
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

## Test avec smartphone ou PC

1. Se connecter au Wi-Fi `ROBOCAR_CTRL`.
2. Ouvrir une connexion TCP vers `192.168.4.1:3333`.
3. Envoyer des caracteres ASCII (`F`, `R`, `r`, `f`, `S`, etc.).

Exemple avec netcat:

```bash
printf "F" | nc 192.168.4.1 3333
```

Verifier dans les logs serie:
- `Wi-Fi receiver initialized`
- `Wi-Fi AP started: ssid=ROBOCAR_CTRL channel=1`
- `Wi-Fi control server listening on TCP port 3333`
