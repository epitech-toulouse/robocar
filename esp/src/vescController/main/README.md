# VESC Controller + BLE (ESP32-S3)

Ce composant supporte:
- Pilotage automatique via LiDAR
- Override manuel via BLE GATT (ESP32-S3 compatible)

## BLE GATT

- Device name: `ESP32S3_BLE_CTRL`
- Service UUID (16-bit): `0xFFE0`
- Characteristic UUID (16-bit): `0xFFE1`
- Propriete: Write + Write Without Response
- Payload: 1 caractere ASCII par evenement

Protocole (respecte le document `BluetoothProtocol.md`):
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
- `F` => duty `+0.08`
- `B` => duty `-0.08`
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

## Test avec smartphone

1. Ouvrir nRF Connect (ou LightBlue)
2. Scanner et connecter `ESP32S3_BLE_CTRL`
3. Ouvrir le service `0xFFE0`
4. Ecrire dans la caracteristique `0xFFE1`
5. Envoyer `F` puis `f`

Verifier dans les logs serie:
- `BLE receiver initialized`
- `BLE advertising started`
- `Manual BLE cmd duty=... steer=...`
