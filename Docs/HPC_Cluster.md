# High Performance Cluster

## Verbindung aufbauen

### 1. Zugang mit VPN
Aktivierung des VPN Clients, falls man nicht an der *Hochschule Osnabrück* ist

### 2. Anmeldung über Schnittstelle
Link: https://web.hpc.hs-osnabrueck.de (VPN Client muss aktiv sein)

### eine Python-Umgebung aufbauen
Um den Reibungslosen einsatzt von Python sowie Jupyter zu gewährleisten muss ein Venv erstellt werden 

- Docs zu den Venvs von Python https://docs.python.org/3/library/venv.html

Um ein **venv** zu erstellen gehen wir auf auf der oberen blauen Leiste und Klicken 

- *Clusters* 

- *HiPer4All Shell Access*
somit haben wir den Terminal oder auch *Shh* um unsere **venv** zuerstellen.

Ausführliche Docs der HS: https://docs.hpc.hs-osnabrueck.de/de/pages/usage/web/apps/python/create-env.html

Navigation zu unseren Arbeitsverzeichnis

1. ``cd /cluster/user/$USER``

Erstellen Sie hier ein Verzeichnis für alle Ihre venvs:

`*mkdir venvs*`

Gehen Sie in dieses neue Verzeichnis:

`cd venvs`

`python3 -m venv myenv`

`source /cluster/user/$USER/venvs/myenv/bin/activate`

(myenv) [username@m10-09 venvs]$





### 3. Jupyter Notebook verbinden
Durch anklicken des folgenden Links in Schritt 2. https://web.hpc.hs-osnabrueck.de wird man auf den HPCluster Webschnittstelle geführt.

- Klicke nun auf *Interactve Apps* mittig oben oder auch mittig unter der Suchleiste

-  Es öffnet sich ein Dropdown menü Klicke nun Jupyter Lab

**ACHTUNG:** die Ressourcen einteilung sollte sinnvoll eingesetzt werden da auch Verwarnung bis hinzu Konto sperrungen enstehen können, sollte man uneffizient und alle Ressourcen ausnutzen.