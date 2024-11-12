# Ausgewählte Modelle von Staty
> Seite 122 Kapitel 5.1.1

Die Unterschiede zwischen "Dense" und "Convolution" sind folgende

- Dense layer lernen globale Muster in ihrem Input-Merkmalsraum (z.B.,
für eine MNIST-Ziffer, Muster, die alle Pixel umfassen) 

- wobei Convolution layer lokale Muster im Falle unserer MNIST Ziffern sind ecken, texturen und mehr.

Ein Bild kann in mehere Lokale Muster eingeteilt werden. Wie in Fall einer 4 hat sie anderen Erknennungsmerkmale wie Ecken, rundungen als andere Zahlen.


> S.123 

Zwei Schüsselmerkmale verleiht Convnets zwei besonder Eigentschaften:

1. Übersetzungsinvariant:
Wenn das Convnet ein bestimmtes Muster erkannt hat wie oben Rechts im Bild kann er diese Muster wiedererkennen in der unteren Linken ecke. Er kann es überall wiedererkennen im gesamten Bild.

Eine Denselayer müsste diese Muster immer wieder neu erlernen was die berechnungszeit siginifikant erhöhen würde.
- Sie brauchen weniger Trainingsmuster um Repräsentationen zulernen die sich verallgeminern lassen.

2. Hierarchies of patterns
Die erste Faltungssicht lernt kleine lokale Muster wie Kanten, eine zweite Faltungssicht lernt größere Muster, die aus den Merkmalen der ersten Sichten besteht, und so weiter
Im Falle einer Katze werden Muster im ersten Layer erkannt wie 
- Augen, Nase, Ohren
Im zweiten Layer werden muster erkannt wie:
- die Form und rundgen der Ohren
- Die Augen und ihre rundgen alle Formen und rundgen ecken die das Auge, Ohren und andere Erknunngsmerkmale der Katze darstellen.

3D-Matrizen auch genannt "Feature Maps"
*"height, length and depth axis"* also höhe und breite und die Tiefe des Bildes. 
Für ein *RBG* (Red, Green and Blue) Bild sind es 3 Farbkanäle somit hat es eine Tiefe (depth) von 3. 

Für ein Schwarz-weiß Bild wie die MNIST Ziffern die Tiefe ist 1 (Grautöne)

Bei der Convolution werden alle Transformationen auf jeder Sichtangewendet  und übergeben und es ensteht ein *Output feature map*.
Die *Output feature map* ist weiterhin eine 3D Matrix mit der breite und tiefe jedoch ist die tiefe nichtmehr die Farbkanäle (RGB) sondern sind die Filter

In unseren Anwendungsfall MINST Ziffern der erste onvolution layer nimmt eine Feature map mit der größe 28,28,1 also 28 breit, 28 lang, 1 Farbkanal (Grautöne). Als Output ensteht eine Feature map von (26,26,32). Die 32 stehen nun für die Filter die die berechnungen ergeben haben.










**SENet** (Squeeze-and-Excitation Network):
Die SE-Blöcke verleihen dem Netzwerk die Fähigkeit, relevante Merkmale in den Feature-Maps zu verstärken und weniger wichtige zu unterdrücken.


