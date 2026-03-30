# Robocar Control Protocol

Ce document décrit le protocole de commande utilise pour communiquer entre l'application mobile et la voiture autonome.

## Transport

- **Type de communication actuel** : Wi-Fi (TCP), envoi de caracteres ASCII.
- **Compatibilite protocole** : Les commandes sont identiques a l'ancien transport BLE.

## Format des Commandes (Payload)

Le protocole envoie de simples caractères ASCII (1 octet) pour chaque action. Cela garantit un décodage très simple et extrêmement rapide côté microcontrôleur (Arduino, STM32, ESP32, etc.) de la voiture.

Chaque état (Appui / Relâchement) possède sa propre commande unique :

| Action  | État | Caractère ASCII envoyé | Description |
| ------------- | ------------- | :-------------: | ------------- |
| **Avancer** | Enfoncé (Press) | `F` | Forward commence |
| **Avancer** | Relâché (Release) | `f` | Forward s'arrête |
| **Reculer** | Enfoncé (Press) | `B` | Backward commence |
| **Reculer** | Relâché (Release) | `b` | Backward s'arrête |
| **Gauche** | Enfoncé (Press) | `L` | Left commence |
| **Gauche** | Relâché (Release) | `l` | Left s'arrête |
| **Droite** | Enfoncé (Press) | `R` | Right commence |
| **Droite** | Relâché (Release) | `r` | Right s'arrête |
| **STOP (Urgence)** | Appui (Trigger) | `S` | Arrêt général absolu (Emergency Stop) |

### Exemple de scénario :
1. L'utilisateur pose le doigt sur "Avancer" ➔ L'application envoie `F`.
2. L'utilisateur pose un 2ème doigt sur "Droite" (tout en maintenant "Avancer") ➔ L'application envoie `R`. La voiture sait donc qu'elle doit Avancer ET tourner à Droite en même temps.
3. L'utilisateur relâche "Droite" ➔ L'application envoie `r`. La voiture arrête de tourner mais continue d'avancer.
4. L'utilisateur relâche "Avancer" ➔ L'application envoie `f`. La voiture s'arrête complètement.

## Integration cote Microcontroleur (Pseudo-code)

```cpp
void loop() {
  if (socket.available()) {
    char cmd = socket.read();
    
    switch(cmd) {
      case 'F': moteur_avant_marche(); break;
      case 'f': moteur_avant_arret(); break;
      case 'B': moteur_arriere_marche(); break;
      case 'b': moteur_arriere_arret(); break;
      case 'L': direction_gauche(); break;
      case 'l': direction_centre(); break;
      case 'R': direction_droite(); break;
      case 'r': direction_centre(); break;
      case 'S': arret_urgence_coupure_totale(); break;
    }
  }
}
```