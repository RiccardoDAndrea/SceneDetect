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

2. Hierarchiesof patterns
Die erste Faltungssicht lernt kleine lokale Muster wie Kanten, eine zweite Faltungssicht lernt größere Muster, die aus den Merkmalen der ersten Sichten besteht, und so weiter.











**SENet** (Squeeze-and-Excitation Network):
Die SE-Blöcke verleihen dem Netzwerk die Fähigkeit, relevante Merkmale in den Feature-Maps zu verstärken und weniger wichtige zu unterdrücken.


