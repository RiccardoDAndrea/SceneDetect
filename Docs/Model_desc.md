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

In unseren Anwendungsfall MINST Ziffern der erste Convolution layer nimmt eine Feature map mit der größe 28,28,1 also 28 breit, 28 lang, 1 Farbkanal (Grautöne). Als Output ensteht eine Feature map von (26,26,32). Die 32 stehen nun für die Filter die die berechnungen ergeben haben. Jeder dieser 32 Output Kanäle enthalten eine 26 x 26 Werte die auch *respone map* genannt wird.
Die die Antwort dieses Filtermusters an verschiedenen Stellen im Eingang angibt verschiedenen Stellen des Eingangs anzeigt.
Das ist auch die Bedeutung der *Feature map* jede Dimension in der Tiefenachse ist ein *feature* (or filter) und der 2D-Tensor
output[:, :, n] ist die räumliche 2D-Abbildung der Antwort dieses Filters auf den Eingang.



Convolution sind definiert an zwei Key Parameter:
- Size of the patches extracted form the Inputs:
    Überlicherweise 3x3 or 5x5. Im bsp. waren es 3x3 was eine übliche Entscheidung ist.

- Depth of the output feature map:
    Die Anzahl der filter die vom Convolution berechnet wurden. Das bsp begann mit der Tiefe und endete mit 64.

In Keras werden bei einer Conv2D-Schicht (also einer 2D-Faltungsschicht) die ersten Parameter, die der Schicht übergeben werden, wie folgt definiert:

Conv2D(output_depth, (window_height, window_width))


output_depth: Die Anzahl der Filter, die in dieser Schicht verwendet werden. Jeder Filter erzeugt eine Feature Map, und output_depth gibt an, wie viele verschiedene Feature Maps erzeugt werden.
(window_height, window_width): Die Größe des Fensters (oder des Filters), das über das Bild „geschoben“ wird. Zum Beispiel ist ein häufig verwendetes Fenster die Größe 3×3 oder 5×5.
Wie funktioniert eine Faltung?
Das Fenster (Filter):

Die Faltung arbeitet, indem ein Fenster (Filter) mit der Größe 3×3 oder 5×5 über das Bild „geschoben“ wird.
Jedes dieser Fenster erfasst einen bestimmten Bereich des Bildes – die umgebenden Merkmale.
Wenn das Bild 3D ist (z. B. ein Farbbild mit drei Farbkanälen: Rot, Grün und Blau), dann hat das Fenster beim Falten eine Größe von (window_height, window_width, input_depth). Das bedeutet, es betrachtet ein kleines Quadrat des Bildes, das auch die Tiefe des Bildes (die Anzahl der Farbkanäle) umfasst.
Transformation der 3D-Patches:

Jedes dieser kleinen 3D-Patches, das das Fenster erfasst, wird durch das Filter (das eine Gewichtsmatrix ist) verarbeitet. Dies geschieht durch eine Tensor-Produkt (eine mathematische Operation), und das Ergebnis dieser Berechnung ist ein 1D-Vektor.
Der Vektor hat die Form (output_depth,), wobei output_depth die Anzahl der Filter ist, die du in dieser Schicht verwendest.
Erstellung der Ausgabematrix:

Alle diese 1D-Vektoren, die aus den 3D-Patches stammen, werden zu einer neuen 3D-Ausgabematrix (oder Feature Map) zusammengesetzt. Diese neue Matrix hat die Dimensionen (height, width, output_depth).
Jeder Punkt in dieser neuen Feature Map entspricht einem Punkt im Eingangbild, aber jetzt mit den extrahierten Merkmalen von der Faltung.
Zum Beispiel: Der Punkt output[i, j, :] in der Ausgabematrix wird aus dem 3D-Patch des Eingangsbildes berechnet, der von den Pixeln im Bereich input[i-1
+1, j-1
+1, :] stammt. Das bedeutet, dass der Filter das kleine 3x3-Bereich im Eingangsbild anschaut und daraus einen Wert berechnet.


Achtung:
Die Output breite sowie die länge können anders ausfallen wegen folgeden 2 möglichkeiten:

- Randeffekte, denen durch Auffüllen der Eingabe-Merkmalskarte entgegengewirkt werden kann
- Die Verwendung von Strides, die ich in einer Sekunde definieren werde
































**SENet** (Squeeze-and-Excitation Network):
Die SE-Blöcke verleihen dem Netzwerk die Fähigkeit, relevante Merkmale in den Feature-Maps zu verstärken und weniger wichtige zu unterdrücken.


