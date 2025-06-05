# 📝 Tehtävät

## Tehtävä: Automaattivaihteet (Pt. 1)

Kouluta Decision Tree ja/tai Random Forest -luokittelumalli. Tehtävänäsi on luoda koneoppimismalli, joka pyrkii ennustamaan muiden kenttien avulla, onko kyseessä automaattiauto. Datan voit ladata alkuperäisestä lähteestään, [Kaggle: Car Features and MSRP](https://www.kaggle.com/datasets/CooperUnion/cardataset) tai repositoriosta, jossa sitä jakavat edelleen sitä käyttäneet. Näistä yksi on: [Github: Car-Prices-Prediction](https://github.com/suhasmaddali/Car-Prices-Prediction/tree/main)

Huomaa, että datasettiä on lähtökohtaisesti käytetty MSRP:n (Manufacturer's Suggested Retail Price) ennustamiseen, mutta tässä tehtävässä keskitymme vain automaattiauton tunnistamiseen. Käytämme siis eri ennustettavaa muuttujaa kuin useimmissa Kaggle-koodeissa, joita tulet löytämään datasettiin liittyen. Tämä ei poista sitä, että saatat saada hyviä ideoita datan käsittelyyn muista repositorioista.

Käännä arvot esimerkiksi seuraavalla tavalla kahdeksi luokaksi:

| Alkuperäinen arvo (str) | Uusi arvo (bool) | N samplea |
| ----------------------- | ---------------- | --------- |
| AUTOMATIC               | 1                | 8266      |
| MANUAL                  | 0                | 2935      |
| AUTOMATED_MANUAL        | 0/1              | 626       |
| DIRECT_DRIVE            | (drop)           | 68        |
| UNKNOWN                 | (drop)           | 56        |

Pudota siis rivit, joissa Transmission Type on `DIRECT_DRIVE` tai `UNKNOWN`. Sen sijaan arvon `AUTOMATED_MANUAL` voit kääntää joko 0:ksi tai 1:ksi. Kokeile, kumman kanssa saat paremman tuloksen. Arvo `1` viittaa siis `True` arvoon, eli automaattiautoon.

Jos käytät jotakin muuta jakoa, kuten luokittelet `UNKNOWN` tai `DIRECT_DRIVE` vaihteistot manuaaliseksi, dokumentoi se oppimispäiväkirjassasi ja perustele valintasi.

!!! tip 

    Dataa on käytetty useissa eri projekteissa, koska se löytyy Kagglestä. Mikäli et tiedä, mistä aloittaa eksploratiivinen data-analyysi ja featureiden muokkaus tai valinta, voit etsiä apua Kagglesta. Valtaosa esimerkeistä ennustaa hintaa, mutta tästä huolimatta datan käsittely on muiden muuttujien osalta samanlaista. Jos et halua ladata dataa Kagglesta, voit ladata sen myös esimerkiksi [Github: Car-Prices-Prediction](https://github.com/suhasmaddali/Car-Prices-Prediction/tree/main) -repositoriosta, mistä löytyy myös esimerkkiä datan käsittelystä.

### Vinkkejä

#### Kirjastot

Aivan kuten aiemmassa tehtävässä, myös tässä tulet tarvisemaan `scikit-learn`-kirjastoa. Tämä pätee kaikkiin kurssin tehtäviin. Kenties seuraavista löytyy apuja?

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
```

#### One Hot Encoding

Kentät `Make` ja `Model` ovat kategorisia muuttujia, joilla on todella suuri määrä uniikkeja arvoja. Tämä pätee myös muihin kenttiin, mutta nämä kentät ovat erityisen ongelmallisia, joten keskitytään niihin tässä vinkissä.

!!! tip

    Kuinka paljon? Selvitä! Tee eksploratiivinen data-analyysi datasetille.

Kentät eivät ole numeerisia vaan kategorisia. Ne on muunnettava numeeriseen muotoon. Toisin kuin vaikkapa "Transmission Type", jossa on vain muutama uniikki arvo, näiden kenttien arvot omaavat suuren granulariteetin eli uniikkien arvojen määrän. Tämä tuskin on mallillle hyväksi: mallin tulee oppia *generalisoimaan* ongelma, joten on ongemallista, jos datasetissä on vaikkapa vain kolme Bugatti Veyronia.

Kenties sinun siis kannattaa käsitellä data siten, että käytät **one hot encoding** -menetelmää, joka muuttaa kategoriset muuttujat numeeriseen muotoon siten, että jokainen uniikki arvo saa oman sarakkeensa, ==mutta pidät vain n yleisintä==. Tämän voi tehdä monella tavalla, mutta yksi tapa on OneHotEncoder(..., min_frequency=10), jolloin kaikki ne arvot, jotka esiintyvät ==alle 10 kertaa==, muutetaan luokan `infrequent` edustajaksi.

Tähän soveltuu esimerkiksi seuraava luokka:

```python
from sklearn.preprocessing import OneHotEncoder
```

!!! tip

    Lisähaasteena sarakkeilla on keskinäinen, toiminnallinen riippuvuus (Make → Model). Tämä one-to-many suhde, jossa Make on Modelin yläluokka, voi vaikuttaa siihen, kuinka One Hot Encoding kannattaa toteuttaa.

    Keksitkö tavan yhdistää näiden tiedot? Jos keksit, muistathan vertailla mallin suorituskykyä ennen ja jälkeen tämän yhdistämisen. Onko mallin tarkkuus parempi vai huonompi? Vai pitäisikö jompi kumpi kenttä pudottaa kokonaan?


#### Tulkittavuus

Puumallien yksi etu, varsinkin neuroverkkoihin nähden, on niiden tulkittavuus (engl. interpretability, explainability). Tutustu seuraaviin tapoihin tulkita eri kenttien vaikutusta mallin ennusteeseen:

* `feature_importances_`-attribuutti, joka kertoo, kuinka paljon kukin piirre vaikuttaa mallin ennusteeseen.
* `plot_tree`-funktio, joka piirtää puun rakenteen ja näyttää, miten eri kentät vaikuttavat ennusteeseen.

Voit kokeilla myös haastaa itseäsi edistyneemmillä tavoilla, kuten [SHAP](https://pypi.org/project/shap/) ja sen [violin summary plot](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/violin.html), mutta älä keskity niihin liikaa. Muista, että oppimispäiväkirjan merkinnän aihe ovat puumallit.