# Mallin suorituskyky

Koneoppimismallit eivät suinkaan automaattisesti ole täydellisiä, vaan niiden suorituskykyä tulee arvioida ja vertailla muihin malleihin. Tämä vaatii mallintajalta ammattitaitoa sekä työkaluja, joilla mallin suorituskykyä voidaan arvioida.

Tämä dokumentti tulee vastaan niin varhaisessa vaiheessa kurssia, että sisältö saattaa tuntua vielä hieman abstraktilta. Palaa tähän materiaaliin myöhemmin kunkin koneoppimismallin kohdalla, kun olet valmis arvioimaan mallin suorituskykyä. On tärkeää miettiä, mistä tiedät, että malli ennustaa hyvin sellaista dataa, jota se ei ole nähnyt ennen.

Tässä dokumentissa käsitellään seuraavia aiheita:

* Alisovittaminen (engl. underfitting)
* Ylisovittaminen (engl. overfitting)
* Vinouma (engl. bias)
* Hajonta (engl. variance)

Koneoppimismallin parametrien optimointi on tasapainoilua ääripäiden välillä. Tavoitteena on löytää optimaalinen malli, joka ennustaa hyvin sekä opetus- että testidataa.

!!! warning

    Huomaa, että tässä luvussa keskitytään ohjatun oppimisen (engl. supervised learning) malleihin. Ohjaamattoman oppimisen (engl. unsupervised learning) malleissa ei ole koulutus- ja testidataa, joten niiden suorituskykyä arviointi on huomatavasti vaikeampaa.

## Koulutus- ja testidata

Ennen kuin pohditaan aihetta yhtään enempää, on hyvä varmistaa, että tiedät, mitä tarkoitetaan koulutus- ja testidatalla. Koulutusdata on dataa, jolla malli opetetaan. Testidata on dataa, jolla mallin suorituskykyä arvioidaan. Testidataa ei saa käyttää mallin opettamiseen, sillä silloin malli ei ole enää riippumaton. Mikäli käytät Pandas-kirjastoa, testidataa voi olla helppo jakaa koulutusdatasta `train_test_split` funktiolla:

```python title="IPython"
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("data.csv")
X_train, X_test = train_test_split(data, test_size=0.2)
```

Usein testi- ja koulutusdata jaetaan suhteessa 80/20 tai 70/30. Tämä tarkoittaa, että 80% tai 70% datasta käytetään koulutukseen ja loput testaukseen. Tämä ei ole kiveen hakattu sääntö, vaan riippuu datasta ja mallista, kuinka paljon dataa tarvitaan koulutukseen. Data tyypillisesti sekoitetaan ennen sen jakamista. Puhdas Python-ratkaisu datan jakamiseksi olisi:

```python title="IPython"
import random

X = [
    (1, 52.7),
    (0, 51.3),
    ...
    (1, 49.2),
    (0, 50.1)
]

# Shuffle
random.shuffle(X)

# Split point
i = int(len(X) * 0.8)

# Split
X_train = X[:i]
X_test = X[i:]

# All samples are included but in two different sets
assert len(X_train) + len(X_test) == len(X)
```


## Ali- ja ylisovittaminen

Alisovittaminen (engl. underfitting) ja ylisovittaminen (engl. overfitting) ovat ongelmatekijää konemalleissa, ja jokainen malli etsii tasapainoa näiden kahden välillä.

![Joke about overfitting and connect the dots](../images/optimization_overfitting.png)

**Kuvio 1:** *Yhdistä pisteet -piirroskirjassa ihmisen tehtävä on yhdistää pisteet numeroidusti ja oikein. Koneoppimismallin tehtävä olisi pikemminkin etsiä ympäripyöreä kissan muoto annetuilla havainnoilla.*

![Overfitting and underfitting](../images/optimization_scikit_doc_overfitting.png)

**Kuvio 2:** *Scikit-learnin dokumentaatiosta poimittu kuva, joka havainnollistaa alisovittamista ja ylisovittamista. Katso kuvan luonut koodi selityksineen: [Underfitting vs. Overfitting](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html#sphx-glr-auto-examples-model-selection-plot-underfitting-overfitting-py)*

### Vinouma

Vinouma (engl. bias) on mallin virhe, joka johtuu vääristä oletuksista. Vinouman tapauksessa malli **alisovittaa** dataa, eli malli ei kykene selittämään ilmiön monimutkaisuutta. Malli on siis liian yksinkertainen datan monimutkaisuuteen nähden.

Vinouman tunnistaa siitä, että koulutusdatan virhe on suurempi kuin testidatan virhe.

![Training model with 1 sample](../images/optimization_training_size_1_parabol.png)

**Kuvio 3:** *Kuvassa on esitetty jokin ilmiö, joka sattuu noudattamaan hyvin lähelle toisen asteen yhtälöä eli parabolia. Valittu malli on koulutettu tasan yhdellä havainnolla: kaikkia muita käytetään testidatana. Huomaa, että `X_train`-virhe eli testivirhe on tasan 0. Viiva läpäisee pisteen.*

![Training model with 10 samples](../images/optimization_training_size_10_parabol.png.png)

**Kuvio 4:** *Kuvassa on esitetty sama ilmiö kuin kuviossa 3, mutta tällä kertaa malli on koulutettu kymmenellä havainnolla. Malli on yhä liian yksinkertainen, ja se alisovittaa dataa. Treenidatan suhteen virhe ei ole enää 0.0, mutta vastaavasti testidatan virhe on vähemmän altis satunnaisuudelle.*

Tutustu yllä oleviin kuvioihin (Kuviot 3 ja 4) tarkasti. Kuvioissa on `MSE test` ja `MSE train` arvot eli koulutus- ja testidatan virhe suhteessa malliin. Valittu malli (eli suora viiva) on liian yksinkertainen selittämään ilmiötä, joka sattuu noudattamaan parabolista käyrää. Koulutusdatan lisääminen auttaa vähentämään virhettä, mutta todellinen ratkaisu olisi lisätä malliin monimutkaisuutta (eli mallin polynomisen asteen nostaminen). Huomaa kuitenkin, että jos koulutusdataa on liian vähän, kuten vain 1 tai 2 havaintoa, niin pelkkä monimutkaisuuden lisääminen voi johtaa jopa entistä suurempaan virheeseen.

!!! question "Tehtävä"

    Ihminen on eräänlainen koneoppimismalli, joka osaa luokitella näkemäänsä. Tutustu aiheeseen **kognitiivinen vinouma** ja mieti, miten se suhtautuu koneoppimismallin vinoumaan. Asiantuntijana on tärkeää osata epäillä myös omia oletuksiaan.

### Hajonta

Hajonta (engl. variance) on vinouman vastakohta.  Hajonnan tapauksessa malli **ylisovittaa** dataa eli se pitää pienintäkin kohinaa merkittävänä, selittävänä tekijänä. Hajonta on mallin virhe, joka johtuu siitä, että malli on liian monimutkainen datan monimutkaisuuteen tai määrään nähden.

Hajonnan tunnistaa siitä, että koulutusdatan virhe on pienempi kuin testidatan virhe. Katso uusin silmin Kuviota 2.

### Regularisointi

Regularisointi on menetelmä, jolla voidaan vähentää ylisovittamista. Regularisointi lisää mallin virhefunktion (engl. loss function) rangaistusta, mikäli malli on liian monimutkainen. Regularisointi on erityisen tärkeä menetelmä, kun käytetään monimutkaisia malleja, kuten neuroverkkoja. Tällä kurssilla regularisointia käsitellään korkeintaan pintapuolisesti, mutta se on tärkeä tunnistaa jo nyt terminä.

### Trade-off

|                          | Ylisovitus | Alisovitus   |
| ------------------------ | ---------- | ------------ |
| **Suurempi virhe**       | Testidata  | Koulutusdata |
| **Vinouma vai hajonta?** | Hajonta    | Vinouma      |

Huomaa, että ylisovittamisen ja alisovittamisen välillä on tasapaino, jota kutsutaan trade-offiksi. Ellei ennustettu malli noudata **täydellisesti** ilman kohinaa jotakin matemaattista kaavaa, malli on aina väkisinkin ali- tai ylisovittava. Tavoitteena on löytää optimaalinen malli, joka ennustaa hyvin sekä opetus- että testidataa. Jos malli ennustaa hyvin opetusdataa, mutta huonosti testidataa, se on ylisovittava. Jos malli ennustaa huonosti sekä opetus- että testidataa, se on alisovittava. Alla on taulukko, joka kuvaa, miten jompaa kumpaa ääripäätä voidaan korjata.

|                           | Ylisovitus             | Alisovitus               |
| ------------------------- | ---------------------- | ------------------------ |
| **Mallin monimutkaisuus** | Laske                  | Nosta                    |
| **...tai regularisaatio** | Nosta                  | Laske                    |
| **# Muuttujaa**           | Poista muuttujia       | Tehtaile lisää muuttujia |
| **# Havaintoa**           | Kerää lisää havaintoja | ---                      |

 Yllä oleva taulukko tarjoaa ratkaisuehdotuksen yli- tai alisovittamisen tapauksessa.

 ![Model complexity vs errors](../images/optimization_model_complexity_intuition.png)

 **Kuvio 5:** *Kuvassa on esitetty mallin monimutkaisuuden vaikutus virheeseen. Kun mallin monimutkaisuutta lisätään, esimerkiksi regularisaatiota vähentämällä tai muuttujia lisäämällä, koulutusvirhe laskee. Alisovitus vaihtuu vaihtua ylisovitukseksi, kun yleistettävyys heikkenee. Kuvaajassa tätä ilmentää koulutusvirheen ja testivirheen etäisyys toisistaan. Vastaava graafi, jossa x-akselilla on epookkien määrä, tulee sinulle myöhemmissä opinnoissa tutuksi neuroverkkojen kanssa.*

## Mittariston valinta

Yllä olevissa esimerkeissä käytetty virhe oli `MSE`. Datalla ei ollut luokkaa vaan juokseva arvo, eli kaikissa yllä olevissa kuvissa ongelma oli tyyppiä regressio. Luokitteluongelmissa käytetään erilaisia mittareita. Alla on listattu yleisimmät mittarit, joita käytetään koneoppimismallien suorituskyvyn arvioimiseen kussakin koneoppimismallityypissä.

### Regressiomallit

#### MSE

MSE (Mean Squared Error) on virhefunktio, joka laskee keskimääräisen virheen neliösumman. Se on yksi yleisimmistä virhefunktioista regressiomalleissa, ja se lasketaan seuraavalla kaavalla:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - f(x_i))^2
$$

Kyseistä virhefunktiota käytetään [Normaaliyhtälö](../algoritmit/linear/normal_equation.md) -materiaalissa.

Huomaa, että virhe on nostettu neliöön. Jos ennustat esimerkiksi asunnon hintaa ja MSE on 10,000 euroa, se tarkoittaa, että nimenomaan **neliösumma** virheestä on 40,000 euroa. Neliö voidaan palauttaa alkuperäiseen skaalaan ottamalla neliöjuuri virheestä: `sqrt(40_000)` palauttaa arvon `200`, koska `200 * 200 = 40_000`. Neliösummaa käytetään ==mallin koulutuksessa==, mutta alkuperäiseen skaalaan palautettu `RMSE` (Root Mean Squared Error) on helpompi ymmärtää evaluaatiovaiheessa.

Sekä MSE:n että RMSE:n voi laskea myös Scikit-Learn kirjaston funktioilla:

```python title="IPython"
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error

X_train, X_test, y_train, y_test = ... # Load data

# Train any regression model here
#   model = SomeRegressionModel()
#   model.fit(X_train, y_train)
#   y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# These should be equal
assert rmse == root_mean_squared_error(y_test, y_pred)
```

#### R^2

R^2-luku kuvaa selitysastetta (engl. coefficient of determination) prosentteina välillä 0.00 - 1.00. Jos R^2 on 0.75, se tarkoittaa, että 75% muuttujan vaihtelusta voidaan selittää mallilla. Loput ovat virhettä, jonka selittää jokin muu tekijä.

R^2 voidaan laskea seuraavalla kaavalla:

$$
R^2 = 1 - \frac{\text{RSS}}{\text{TSS}}
$$

Jossa: 

* RSS (Residual Sum of Squares) on virheiden neliösumma
* TSS (Total Sum of Squares) on kokonaisneliösumma. 

Huomaa, että RSS on siis sama kuin MSE, mutta jakolasku jätetään tekemättä (eli "mean"). Matemaattisina yhtälöinä RSS ja TSS ovat:

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

### Luokittelumallit

#### Hämmennysmatriisi

Käsittele tässä:

* True Positive (TP)
* True Negative (TN)
* False Positive (FP)
* False Negative (FN)

#### Accuracy

TODO

#### Precision

TODO

#### Recall

TODO

#### F1 score

TODO

#### Ehkä: ROC ja AUC

TODO



