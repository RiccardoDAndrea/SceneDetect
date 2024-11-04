# High Performance Cluster

## Verbindung aufbauen

### 1. Zugang mit VPN
Aktivierung des VPN Clients, falls man nicht an der *Hochschule Osnabrück* ist

### 2. Anmeldung über Schnittstelle
Link: https://web.hpc.hs-osnabrueck.de (VPN Client muss aktiv sein)

### eine Python-Umgebung aufbauen
Um den Reibungslosen einsatzt von Python sowie Jupyter zu gewährleisten muss ein Venv erstellt werden 

- Docs zu den Venvs von Python https://docs.python.org/3/library/venv.html

Um eine virtuelle Umgebung (venv) zu erstellen, folgen Sie diesen Schritten:

1. Navigieren Sie in der oberen blauen Leiste zu:
   - **Clusters**
   - **HiPer4All Shell Access**

   Dadurch erhalten Sie Zugriff auf das Terminal (SSH), um Ihre venv zu erstellen.

2. Weitere Informationen finden Sie in den ausführlichen Dokumentationen der Hochschule:
[Dokumentation zur Python-Umgebung](https://docs.hpc.hs-osnabrueck.de/de/pages/usage/web/apps/python/create-env.html).

3. Navigieren Sie zu Ihrem Arbeitsverzeichnis:

   `cd /cluster/user/$USER`
   
4. Erstellen Sie ein Verzeichnis für alle Ihre virtuellen Umgebungen:

    `mkdir venvs`

5. Wechseln Sie in das neu erstellte Verzeichnis:

    `cd venvs`

6. Erstellen Sie die virtuelle Umgebung:

    `python3 -m venv myenv`

    Aktivierung des venvs
    `source /cluster/user/$USER/venvs/myenv/bin/activate`

7. Im Terminal sollte nun im Verzeichnis:

    `(myenv) [username@m10-09 venvs]$`


### 3. Jupyter Notebook verbinden
Durch anklicken des folgenden Links in Schritt 2. https://web.hpc.hs-osnabrueck.de wird man auf den HPCluster Webschnittstelle geführt.

- Klicke nun auf *Interactve Apps* mittig oben oder auch mittig unter der Suchleiste

-  Es öffnet sich ein Dropdown menü Klicke nun Jupyter Lab

**ACHTUNG:** die Ressourcen einteilung sollte sinnvoll eingesetzt werden da auch Verwarnung bis hinzu Konto sperrungen enstehen können, sollte man uneffizient und alle Ressourcen ausnutzen.