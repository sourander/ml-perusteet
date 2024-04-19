# Hill Climbing

Edellisessä luvussa normaaliyhtälöä, joka sovitti suoran dataan yhden kaavan ratkaisuna. Tässä luvussa käsitellään optimointia toisella tavalla: Hill Climbing -algoritmilla, joka pyrkii etsimään ratkaisua iteratiivisesti eli toistamalla itseään silmukassa. Hill Climbing on tuskin tuotantokäyttöön soveltuva algoritmi, mutta se on hyvä esimerkki siitä, miten optimointi voidaan toteuttaa, ja toimii pohjustuksena Gradient Descent -algoritmille, joka on yleisempi ja tehokkaampi optimointimenetelmä. Sitä käsitellään seuraavassa luvussa.


## Hill Climbing

Optimointiin voi käyttää yllä olevan kaavan sijasta eri koneoppimisen keinoja, joista yksi on Hill Climbing. Hill Climbing on yksinkertainen algoritmi, joka pyrkii löytämään paikallisen maksimin tai minimin. Se toimii seuraavasti:

1. Alusta kertoimet satunnaisesti
2. Laske virhe
3. Lisää kertoimiin satunnaisluku (välillä ±1.0)
4. Laske virhe
5. Jos virhe pienenee, hyväksy muutos
6. Toista 3-5 kunnes pysäytyskriteeri täyttyy

Luonnollinen pysäytyskriteeri on brute force -menetelmän vuoksi `max_iter` eli iteraatioiden maksimimäärä.

Tämän yksinkertaisen metodin sijasta voisimme käyttää Gradient Descent -algoritmia, joka on tehokkaampi ja yleisempi menetelmä optimointiin. Gradient Descent on algoritmi, joka pyrkii löytämään virhefunktion minimin iteratiivisesti derivoimalla funktion ja liikkumalla vastakkaiseen suuntaan gradientin suhteen pieni askel kerrallaan. Tähän tutustutaan hyvin pintapuolisesti seuraavassa luvussa, jossa käsitellään n-uloitteista lineaarista regressiota (engl. multivariate linear regression).

!!! note

    Huomaa, että yllä olevalla algoritmilla on mitättömät mahdollisuudet onnistua, jos piirteitä ei ole normalisoitu. Tämä johtuu siitä, että suuret arvot voivat dominoivat virhefunktiota ja estää algoritmin löytämästä optimaalista ratkaisua. Lisäksi valittu satunnaisluku on todennäköisesti liian pieni, mikäli piirteet edustavat suuria arvoja (kuvittele MSE, jos kenttä sisältää lukuja, kuten `233_535.124`)

## Ongelman esittely

Kerrataan vielä tärkeät termit:

* **Virhefunktio**. Virhefunktio tai tappiofunktio mittaa mallin ennusteen virheellisyyttä. Meille on jo aiemmin tullut tutuksi MSE eli keskimääräinen neliövirhe.
* **Optimointi**. Optimointi on prosessi, jossa pyritään minimoimaan yllä mainittua virhefunktiota. Tässä luvussa käytämme Hill Climbing -algoritmia.
* **Parametrit**. Parametrit ovat mallin kertoimia, jotka määrittävät mallin käyttäytymisen. Esimerkiksi lineaarisessa regressiossa kertoimet `a` ja `b` määrittävät suoran kulmakertoimen ja vakiotermin. Nämä painot määrittävät mallin ennusteen. Yllä oleva `.fit(X, y)` laskee nämä kertoimet.

### Parametrit (W)

Keskitytään hetkeksi parametreihin eli painokertoimiin. Aiemmassa luvussa käsittelimme yksinkertaista lineaarista regressiota, jossa malli oli muotoa $y = b + wx$ (engl. univariate regression). Tässä luvussa datamme on monimuuttujaista (engl. multivariate). Tarkemmin ottaen meillä on kaksi selittävää muuttujaa: ==käärmeen pituus senttimetreinä== ja ==sään lämpötila== puremahetkellä.

| Käärmeen mitta | Sää (°C) | Sairasloma |
| -------------- | -------- | ---------- |
| 78.38          | 32.55    | 116        |
| 300.00         | 35.00    | 196        |
| 208.11         | 0.00     | 5          |
| ...            | ...      | ...        |

!!! info

    Tavoitteena on siis oppia generalisoimaan, että jos :snake: puree sinua, sään ollessa 25.2 °C ja käärmeen ollessa 1.5 metriä pitkä, kuinkako monta sairaslomapäivää on odotettavissa.

Kahden muuttujan kohdalla suora vaihtuu tasoksi ja sen kaava on:

$$
\hat{y} = b + w_1x_1 + w_2x_2
$$

Koska muuttujia on kaksi, myös kulmakertoimia (tai muuttujien painokertoimia) on kaksi. Kirjain `b` edustaa vakiotermiä, joka on sama kuin aiemmin. Koska haluamme, että myös `b` on optimoitava parametri, lisätään se matriisiin `X` staattisena ==numerona yksi==. Kun tätä ykköstä kertoo millä tahansa painolla, tulos on aina sama kuin paino (koska `w = w * 1`). Jatkossa matriisi `X`, kun siihen lisätään vakiotermille oma sarakkeensa, on muotoa:

| x[0] | x[1]   | x[2]  |
| ---- | ------ | ----- |
| 1    | 78.38  | 32.55 |
| 1    | 300.00 | 35.00 |
| 1    | 208.11 | 0.00  |
| ...  | ...    | ...   |

Jatkossa kutakin kaikkia näitä kolmea, `x[0], x[1], x[2]`, kohden on olemassa oma kulmakerroin `w[0], w[1], w[2]`. Matriisi `X` no siis kokoa `(m, n)`, jossa `m` on piirteiden määrä ja `n` on havaintojen määrä. Vektori `w` on kokoa `(m, 1)`.

!!! tip

    Jatkossa siis `b` on `x[0]`

### Silmukassa

Tämän voi siis suorittaa silmukassa, jossa käydään kukin sample läpi, ja kerrotaan sen samplen kukin piirre sitä vastaavalla painolla (eli `x[0] * w[0] ... x[n] * w[n]`). Tämän jälkeen kaikki tulokset summataan yhteen ja saadaan ennuste. Koodina se näyttää tältä:

```python
from random import random

X = [
    (1, 78.38, 32.55),
    (1, 300.00, 35.00),
    (1, 208.11, 0.00),
    (1, 100.00, 7.22)
]
y = ... # Doesn't matter when predicting

m_features = len(X[0])           # 3
n_samples = len(X)               # 4
w = [random() for i in range(m)] # Randomize all three

y_hat = []
for row in X:
    y_feat = sum([row[i] * w[i] for i in range(m)])
    y_hat.append(y_feat
```

### Matriisitulona

Koska yllä esitelty operaatio on sama kuin matriisin `X` ja vektorin `w` välinen tulo, voimme korvata silmukoiden käytön vektorisoidulla operaatiolla. Tässä voit käyttää joko omaa `Vector`- ja `Matrix`-luokan toteutusta tai käyttää NumPy-kirjastoa. Jälkimmäinen on nopea ja tehokas tapa, aiempi on hyvä tapa nähdä konepellin alle. :nerd:

$$
y = Xw
$$

Käytännössä siis:

$$
Xw =\begin{bmatrix}
   1 & x_{1_1} & x_{1_2} & \cdots & x_{1_m}\\
   1 & x_{2_1} & x_{2_2} & \cdots & x_{2_m}\\
   1 & x_{3_1} & x_{3_2} & \cdots & x_{3_m}\\
   \vdots & \vdots & \vdots & \vdots & \vdots \\
   1 & x_{n_1} & x_{n_2} & \cdots & x_{n_m}\\
   \end{bmatrix}
   \dot{}
    \begin{bmatrix}
    w_0\\
    w_1\\
    w_2\\
    \vdots\\
    w_m
    \end{bmatrix}
    =
    \begin{bmatrix}
    y_1\\
    y_2\\
    y_3\\
    \vdots\\
    y_n
    \end{bmatrix}
$$

Jos yllä olevan kaavan oikealta puolelta avaa `y[1]`-ennusteen tulon kaavaksi, se on:

$$
y_1 = 1 \cdot w_0 + x_{1_1}w_1 + x_{1_2}w_2 + \cdots + x_{1_m} \cdot w_m
$$

!!! note

    Toisin kuin Pythonissa, käytämme havaintojen yhteydessä `1`-indeksiä, eli ensimmäinen havainto, `x[n]`, on `x[1]` eikä `x[0]`.


```python
import numpy as np

# Convert to numpy arrays
X = np.array(X)
w = np.array(w)

# Prediction is the dot product
y_hat = X @ w

# Note that
len(X) == len(y_hat) == n_samples
```

## Käärmedatan esittely

Jatkamme saman käärmeenpuremia käsittelevän kuvittelevan datan kanssa, joka yllä on esiteltynä. Havaintoja on yhteensä 200 kappaletta. Ennen kuin arvomme satunnaiset painot, tarkastellaan hieman dataa. Ensimmäiset viisi havaintoa näyttävät tältä:

```
    x[0],     x[1],        y
  179.16,    20.39,   202.00
  191.44,    20.39,   204.00
  244.53,    12.53,   212.00
  148.65,    24.22,   184.00
  164.50,     8.64,   119.00
    ...       ...       ...
```

Muistutuksena `x[0]` on :snake: mitta (cm) ja `x[1]` on sää (°C). Y on sairaslomapäivien määrä. Tavoitteena on siis ennustaa sairaslomapäivien määrä käärmeen pureman jälkeen, riippuen käärmeen pituudesta ja säästä. Tarkastellaan hieman oletuksia, mitä datasta voidaan päätellä. Alla on korrelaatio-matriisi, joka kertoo, kuinka paljon muuttujat korreloivat keskenään.

![](../../images/hillclimb_snake_corr_heatmap.png)

**Kuvio 1:** *Seabornin heatmap-funktiolla plotattu `df.corr()` -funktion palauttama korrelaatiomatriisi paljastaa numeraalisena arvona, kuinka samat parit korreloivat keskenään.*

Korrelaatiomatsiisista on pääteltävissä, että:

* `Temperature` <=> `y` korrelaatio on 0.58
    * Sään lämpötila ja sairaslomapäivien määrä korreloivat keskenään
    * Jos sinua purraan helteellä, saat todennäköisesti enemmän sairaslomapäiviä
* `Snake Length` <=> `y` korrelaatio on 0.62
    * Käärmeen pituus ja sairaslomapäivien määrä korreloivat keskenään
    * Jos pitkä käärme puree, saat todenäköisesti enemmän sairaslomapäiviä
* `Temperature` <=> `Snake Length` korrelaatio on -0.25
    * Lämpötila ja käärmeen pituus korreloivat miedon negatiivisesti keskenään
    * Jos lämpötila kasvaa, purevan käärmeen pituus laskee.

![](../../images/hillclimb_snake_scatter_3d.gif)

**Kuvio 2:** *Scatter 3D -kuvaaja, joka on luotu Plotly Express -kirjastolla. Kuvaajasta on ihmissilmin pääteltävissä, mihin kohtaan taso kuuluisi piirtää.*

## Ensimäinen iteraatio

Käytetään Hill Climbing -algoritmia ensimmäisen iteraation suorittamiseen. Kuten yllä mainittiin, algoritmi aloittaa arpomalla painot satunnaisesti.

```python
w = np.array([random() for i in range(m)])
print(w)
```

| intercept (w0) | length (w1) | temp (w2) |
| -------------- | ----------- | --------- |
| 0.78           | 0.01        | 0.60      |

### Datan skaalaus

Käärmeen mitta on suuruusluokkaa 50-300, sää on 0-30. Huomaa, että virhefunktio perustuu etäisyyden neliöön, joten suuret arvot dominoivat virhefunktiota. Tämän vuoksi on tärkeää normalisoida data ennen kuin käytämme sitä: muutoin painotamme käärmeen mittaa enemmän kuin lämpötilaa.

```python
class StandardScaler:
    
    def standardize(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        return (X - self.mean) / self.std
    
    def revert(self, X):
        return X * self.std + self.mean

## Normalize the dataset
scaler = StandardScaler()
X_std = scaler.standardize(X)
```

Z-score -skaalaus on esitelty jo aiemmin, joten tässä materiaalissa emme perehdy sen toimintaa. Luomme luokan (ks. koodi yllä), joka sekä normalisoi että palauttaa normalisoidun datan alkuperäiseen muotoon. Skaalattu data näyttää tältä:

```
    x[0],     x[1],        y
    1.13,     0.93,   180.00
    0.74,    -0.16,   125.00
   -1.82,    -1.82,    55.00
    0.52,    -0.49,   129.00
    0.24,    -0.11,   154.00
    ...       ...       ...
```

### Ennusteen laskeminen

Ensimmäinen ennuste voidaan laskea siis seuraavasti:

$$
\begin{align*}
\hat{y}_1 &= (0.78 \cdot 1) + (0.01 \cdot 1.13) + (0.60 \cdot 0.93) \\ 
&= 0.79 + 0.02 + 0.56 \\
&= 1.36
\end{align*}
$$

Oikea arvo on 180.00, joten virhe on 178.64.

### Hill Climb Silmukka

### TODO: Virhefunktio

### TODO: Painojen päivitys

### TODO: Plottaa virheen kehitys