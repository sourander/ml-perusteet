# Mallin suorituskyky

## Ali- ja ylisovittaminen

Alisovittaminen (engl. underfitting) ja ylisovittaminen (engl. overfitting) ovat ongelmatekijää konemalleissa, ja jokainen malli etsii tasapainoa näiden kahden välillä.

![Joke about overfitting and connect the dots](../images/optimization_overfitting.png)

**Kuvio 1:** *Yhdistä pisteet -piirroskirjassa ihmisen tehtävä on yhdistää pisteet numeroidusti ja oikein. Koneoppimismallin tehtävä olisi pikemminkin etsiä ympäripyöreä kissan muoto annetuilla havainnoilla.*

## Regressiomallit

### MSE

MSE (Mean Squared Error) on virhefunktio, joka laskee keskimääräisen virheen neliösumman. Se on yksi yleisimmistä virhefunktioista regressiomalleissa, ja se lasketaan seuraavalla kaavalla:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - f(x_i))^2
$$

Kyseistä virhefunktiota käytetään [Normaaliyhtälö](../algoritmit/linear/normal_equation.md) -materiaalissa. Käy kurkkaamassa kyseisestä materiaalista, kuinka virhefunktio ja jäännökset liittyvät toisiinsa.

### R^2

R^2-luku kuvaa selitysastetta (engl. coefficient of determination) prosentteina välillä 0.00 - 1.00. Jos R^2 on 0.75, se tarkoittaa, että 75% muuttujan vaihtelusta voidaan selittää mallilla. Loput ovat virhettä, jonka selittää jokin muu tekijä.

R^2 voidaan laskea seuraavalla kaavalla:

$$
R^2 = 1 - \frac{\text{RSS}}{\text{TSS}}
$$

Jossa: 

* RSS (Residual Sum of Squares) on virheiden neliösumma
* TSS (Total Sum of Squares) on kokonaisneliösumma. 

* Huomaa, että RSS on siis sama kuin MSE, mutta jakolasku jätetään tekemättä (eli "mean"). Matemaattisina yhtälöinä RSS ja TSS ovat siis:

$$
RSS = \sum_{i=1}^{n} (y_i - f(x_i))^2
$$

$$
TSS = \sum_{i=1}^{n} (y_i - mean(y))^2
$$

Pythonina sen voi kirjoittaa alla olevalla tavalla, ja todistaa oikeasi vertaamalla sitä Scikit Learnin `r2_score` -funktion palauttamaan arvoon.

```python
import sklearn
import numpy as np

def rss(y_true, y_pred):
    rss = sum((y_true - y_pred) ** 2)
    return rss

def tss(y_true):
    mean_y = sum(y_true) / len(y_true)
    tss = sum((y_true - mean_y) ** 2)
    return tss

def r_squared(y_true, y_pred):
    rss_value = rss(y_true, y_pred)
    tss_value = tss(y_true)
    r_squared = 1 - (rss_value / tss_value)
    return r_squared

# Fake data and predictions
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

r_squared(y_true, y_pred) == sklearn.metrics.r2_score(y_true, y_pred)
```

## Luokittelumallit

### Hämmennysmatriisi

Käsittele tässä:

* True Positive (TP)
* True Negative (TN)
* False Positive (FP)
* False Negative (FN)

### Accuracy

TODO

### Precision

TODO

### Recall

TODO

### F1 score

TODO

### Ehkä: ROC ja AUC

TODO



