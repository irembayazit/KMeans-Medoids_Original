from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import matplotlib.pyplot as plt


# Amaç !! yapılan her bir tahmini sınıflandırma için doğruluk değerini dondurur
def Func(realValue, predictValue):
    confusion = confusion_matrix(realValue, predictValue)
    print("confusion matrix : ")
    print(confusion)

    basariOranlari = []
    for line in range(confusion.shape[1]):
        for column in range(confusion.shape[0]):
            if (line == column):
                if (Counter(realValue)[line] != 0):
                    basariOranlari.append(confusion[line][column] * (100 / Counter(realValue)[line]))
                else:
                    print("elbow value is not true")

    print("Basari Oranlari : ", basariOranlari)

    toplam = 0
    for i in range(len(basariOranlari)):
        toplam += basariOranlari[i]
    toplam /= len(Counter(realValue).keys())
    print("Ortalama Sınıf Doğruluğu : ", toplam)

    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion)

    cm_display.plot()
    plt.show()
