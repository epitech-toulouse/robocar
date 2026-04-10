# Robocar WiFi Control Protocol

Ce document decrit le protocole de commande WiFi utilise entre l application mobile (ou page web) et la voiture via ESP32.

## Vue d ensemble

Le systeme expose 2 canaux reseau differents :

- Canal HTTP (interface web + API REST simple)
- Canal TCP brut (commandes ASCII, 1 octet par action)

## Transport WiFi

- Mode reseau : SoftAP (l ESP32 cree son propre reseau)
- SSID : ROBOCAR_CTRL
- Mot de passe : YohannBoniface
- IP ESP32 (par defaut SoftAP) : 192.168.4.1
- Port HTTP : 3333
- Port TCP controle : 3334

## API HTTP (port 3333)

Endpoints principaux :

- GET / : page web de pilotage
- GET /status : etat service, reponse JSON
- GET /cmd?c=X : envoi d une commande (X = caractere protocole)
- GET /logs?since=N : recupere les logs ESP a partir de la sequence N (optionnel)

Notes :

- CORS active
- OPTIONS gere sur /cmd, /status, /logs
- Cache-Control no-store

## Protocole commandes (payload)

Le protocole envoie des caracteres ASCII (1 octet) pour chaque action. C est volontairement minimal pour rester robuste et rapide cote microcontroleur.

Chaque etat (appui / relachement) a sa commande :

| Action | Etat | Caractere envoye | Description |
| --- | --- | :---: | --- |
| Avancer | Press | F | Active forward |
| Avancer | Release | f | Desactive forward |
| Reculer | Press | B | Active backward |
| Reculer | Release | b | Desactive backward |
| Gauche | Press | L | Active left |
| Gauche | Release | l | Desactive left |
| Droite | Press | R | Active right |
| Droite | Release | r | Desactive right |
| Stop urgence | Trigger | S | Arret urgence immediat |

## Fonctionnement logique

Le controle est base sur un etat interne compose de 4 booleens :

- forward
- backward
- left
- right

Regles de calcul :

- Duty = +0.05 si forward actif et backward inactif
- Duty = -0.05 si backward actif et forward inactif
- Duty = 0 sinon
- Steering = 0.2 si left actif et right inactif
- Steering = 0.8 si right actif et left inactif
- Steering = 0.5 sinon

Commande urgence :

- S force duty a 0
- S recentre steering a 0.5
- S remet les 4 booleens de direction/vitesse a false

Timeout manuel :

- Si aucune commande manuelle recente, le mode manuel expire apres 2000 ms

## Exemple de scenario

1. Appui sur Avancer -> envoi F
2. Tout en gardant Avancer, appui sur Droite -> envoi R
3. Relachement Droite -> envoi r
4. Relachement Avancer -> envoi f

Resultat : avance puis avance + droite, puis avance seul, puis arret.

## Redirection optionnelle des logs vers la page web

La page web peut afficher les logs ESP en temps reel de maniere optionnelle :

- Checkbox Stream ESP logs
- Polling GET /logs?since=N toutes les 500 ms quand active
- Le serveur garde un buffer circulaire de logs (sequence + message)

Important :

- La sortie logs serie/UART reste active
- La redirection web est une copie, pas un remplacement

## Pseudo code cote microcontroleur

```cpp
void on_command(char cmd) {
  switch (cmd) {
    case 'F': forward = true;  break;
    case 'f': forward = false; break;
    case 'B': backward = true;  break;
    case 'b': backward = false; break;
    case 'L': left = true;  break;
    case 'l': left = false; break;
    case 'R': right = true;  break;
    case 'r': right = false; break;
    case 'S': emergency_stop(); return;
    default: return;
  }
  recompute_output_from_state();
}
```