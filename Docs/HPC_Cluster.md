# High Performance Cluster


## 1. Zugang mit VPN
Aktivierung des VPN Clients, falls man nicht an der *Hochschule Osnabrück* ist

## 2. Anmeldung über Schnittstelle
Link: https://web.hpc.hs-osnabrueck.de (VPN Client muss aktiv sein)

### Eine Python-Umgebung aufbauen
1. **Um den Reibungslosen einsatzt von Python sowie Jupyter zu gewährleisten muss ein Venv erstellt werden**

    - Docs zu den Venvs von Python https://docs.python.org/3/library/venv.html

Um eine virtuelle Umgebung (venv) zu erstellen, folgen Sie diesen Schritten:


2. **Navigieren Sie in der oberen blauen Leiste zu:**
   - **Clusters**
   - **HiPer4All Shell Access**

   Dadurch erhalten Sie Zugriff auf das Terminal (SSH), um Ihre venv zu erstellen.

3. **Weitere Informationen finden Sie in den ausführlichen Dokumentationen der Hochschule:**
[Dokumentation zur Python-Umgebung](https://docs.hpc.hs-osnabrueck.de/de/pages/usage/web/apps/python/create-env.html).

4. **Navigieren Sie zu Ihrem Arbeitsverzeichnis:**

   `cd /cluster/user/$USER`
   
5. **Erstellen Sie ein Verzeichnis für alle Ihre virtuellen Umgebungen:**

    `mkdir venvs`

6. **Wechseln Sie in das neu erstellte Verzeichnis:**

    `cd venvs`

7. **Erstellen Sie die virtuelle Umgebung:**

    `python3 -m venv myenv`     <- bennung deines *venvs*

8. **Aktivierung des venvs**
    
    `source /cluster/user/$USER/venvs/myenv/bin/activate`

9. **Im Terminal sollte nun im Verzeichnis:**

    `(myenv) [username@m10-09 venvs]$`

10. **Wenn alles erfolgreich ist kann der pip manager geupdatet werden mit dem Befehl:**

    `pip install --upgrade pip`

11. **Nachdem pip aktualisiert wurde können nun auch Libaries wie pandas oder Matplotlib insterliert werden in der Umgebung**

    `pip install matplotlib pandas`

12. **Anzeige der Libaries im venv** 

    `pip list` 

13. Nächste sitzung
In der nächste Sitzung sollte durch folgenden Befehl der venv direkt aktiviert werden

    `source /cluster/user/$USER/venvs/myenv/bin/activate`

14. Jupyter Nootebook nutzen

    `pip install jupyter`

## 3. Jupyter Notebook verbinden
Durch anklicken des folgenden Links in Schritt 2. https://web.hpc.hs-osnabrueck.de wird man auf den HPCluster Webschnittstelle geführt.

1. Klicke nun auf *Interactve Apps* mittig oben oder auch mittig unter der Suchleiste

2. Es öffnet sich ein Dropdown menü Klicke nun *Jupyter Lab*

Alternativ kannst du auch direkt auf
Pinned Apps:
- Juptyer Lab anklicken beide führen dich auf die gleiche Seite


**Einrichtung von Jupyter Lab:**

3. Account:
Such dein Account für  das jewelige Projekt. Durch das Anklicken des Feldes siehst du deine Accounts die dir zur Verfügung stehen.

4. Ersetzte nun "Python Virtual Environment":

    4.1.  `/cluster/user/$USER/venvs/myenv` 
    - Solltest du die Anleitung genau befolgt haben ist alles bereits fertig und es müssen keine änderung durchgeführt werden.

    - Solltest du jedoch dein Venv nicht wie in der Anleitung bennant haben setzt du das Verzeichnis wo dein Venv vorzufinden ist sowie den namen deines venvs

        bsp: `pfad/zu/deinem/venv/projekt1` <- name des venvs `projekt1`
---


**ACHTUNG:** die Ressourcen einteilung sollte sinnvoll eingesetzt werden da auch Verwarnung bis hinzu Konto sperrungen enstehen können, sollte man uneffizient und alle Ressourcen ausnutzen.

*Anzahl der Stunden*:
- Die Anzahl der Stunden, die diese interaktive Anwendung laufen wird. Wenn die Zeit abgelaufen ist, wird der Auftrag beendet und die Anwendung wird gestoppt.

*Kerne:*
- Die Anzahl der CPU-Kerne, die Ihrer Anwendung zugewiesen werden sollen. Bitte gehen Sie sparsam mit den Ressourcen um, um eine gute Benutzererfahrung für alle zu gewährleisten.

*GPU-Typ:*
- Typ und Größe der GPU (falls vorhanden), der Ihre Anwendung zugewiesen wird. Bitte gehen Sie sparsam mit den Ressourcen um, um eine gute Benutzererfahrung für alle zu gewährleisten.

---

Sobald alles Spezifiziert ist kann durch runter scrollen auf Launch der Jupyter Notebook gestartet werden. 
- Dies kann einige Sekunden andauern bis das Jupyter Nootebook gelunched wird

Nach eininger Zeit Startet das Juptyer Nootebook und der Button
`Connect to Jupyter` kann betätigt werden.
