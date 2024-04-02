## Skaalan suhteen herkät algoritmit

Osa koneoppimisalgoritmeista ovat herkkiä skaalaukselle. Tämä tarkoittaa, että algoritmin suorituskyky voi rampautua, mikäli eri piirteiden välillä on suuria eroja. Esimerkiksi huoneiden lukumäärä on yleensä pieni luku, kun taas asunnon hinta dollareita on valtava lukema.

Jos mallin virhe perustuu etäisyyteen, skaalaus on tärkeää. Esimerkiksi K-means-algoritmi käyttää etäisyyttä klustereiden muodostamiseen. Jos yksi piirre on suurempi kuin toinen, se vaikuttaa enemmän etäisyyteen. Huomaa, että skaalauksen tarkka muoto vaihtelee. Kenties piirre puristetaan välille `[0,1]`, tai kenties se skaalataan siten, että keskihajonta mahtuu alueelle `[-1,1]`. Näihin tutustutaan alla tarkemmin, mutta tarkistathan aina että valitsemasi datan esikäsittelijä on se, mitä mallisi tarvitsee. Esimerkiksi neuroverkot käyttävät usein Min-Max skaalausta, joka puristaa datan välille `[0,1]`.

!!! tip

    Tämän kurssin puitteissa riittää seuraava karkea listaus, jossa :white_check_mark: tarkoittaa, että skaalaus on suositeltavaa ja :no_entry: tarkoittaa, että skaalaus ei ole tarpeellista.
    
    * Puut :no_entry:
    * Naive Bayes :no_entry:
    * Muut :white_check_mark:



## Skaalauksen apuvälineet

Dataa voi kuvata kuvailevan tilastotieteen avulla. Näitä ovat keskiarvo, mediaani, moodi, varianssi, keskihajonta ja kvartiilit. Tässä dokumentissa keskitytään keskiarvoon, varianssiin ja keskihajontaan. Kukin näistä esitellään ensin matemaattisessa muodossa ja sen jälkeen Python-koodina. Näitä tarvitaan myöhemmin skaalausta tehdessä.

Tämä materiaali pohjautuu osin [BMC Genomics](https://bmcgenomics.biomedcentral.com/articles/10.1186/1471-2164-7-142/tables/1)-sivuston kaavoihin.

### Keskiarvo (mean)

Keskiarvo on kaikkien otannan (engl. sample) lukujen summa jaettuna lukumäärällä. Jatkossa kun näet `x̄`-symbolin tässä dokumentissa, se tarkoittaa keskiarvoa.

$$
{\overline{x}} = \frac{\sum x_{i}}{N}
$$

```python
# From Scratch - TODO: Use a custom Vector type
def mean(x):
    return sum(x) / len(x)
```

### Varianssi

Varianssi kertoo, kuinka paljon data poikkea keskiarvosta. Luku on nostettu neliöön, jotta negatiiviset poikkeamat eivät kumoaisi positiivisia, ja jotta suuret poikkeamat painottuisivat enemmän.

$$
s^{2} = \frac{\sum \left( {x_{i} - {\overline{x}} } \right) ^{2}}{N - 1}
$$

```python
def variance(x, ddof=1):
    """
    Calculate the variance of a list of numbers.

    Parameters:
    x (list): A list of numbers.
    ddof (int): Delta Degrees of Freedom (Default: 1).

    Returns:
    float: Variance
    """
    return sum((x - mean(x))**2) / (len(x) - 1)
```

!!! question "Miksi - 1?"

    Jakajassa oleva `N - 1` vähentää populaatiosta yhden asteen vapautta. Tämä yhden vapausaste (engl. degrees of freedom) on käytössä otannan (engl. sample) varianssia laskettaessa. Mikäli N edustaa koko populaatiota, sitä ei käytetä. Huomaa, että koska jakaja on pienempi, varianssi on suurempi kuin jos jakajana olisi `N`. Koko populaation varianssin oletetaan siis olevan suurempi kuin otannan varianssin.

### Keskihajonta

Keskihajonta on varianssin neliöjuuri. Se palauttaa neliöön nostetun varianssin takaisin alkuperäiseen mittayksikköön. Esimerkissä oletetaan, että käsittelemme otantaa, joten hyväksymme aiemman toteutuksen `ddof=1` default-arvon.

$$
s = \sqrt{s^{2}}
$$

```python
def std(x):
    return variance(x) ** 0.5
```


## Piirteiden skaalaus

Piirteiden skaalaus on menetelmä, jolla yhtenäistetään eri muuttujien tai piirteiden alue. Tietojenkäsittelyssä sitä kutsutaan myös datan normalisoinniksi ja se suoritetaan yleensä datan esikäsittelyvaiheessa. Se ottaa datataulukon ja palauttaa uuden taulukon samalla muodolla, mutta skaalattuna standardialueelle.

### Keskitys (centering)

Keskitys ei varsinaisesti skaalaa mitään, mutta se on tärkeä osa alla esiteltyjä skaalauksia. Keskityksessä lukujen keskiarvo vähennetään jokaisesta arvosta. Toisin sanoen keskiarvo siirretään nollaan. Mikäli muuttuja noudattaa normaalijakaumaa, puolet arvoista on positiivisia ja puolet negatiivisia.

$$
{\widetilde{x}} = x - {\overline{x}}
$$

```python
def center(x):
    return x - mean(x)
```

### Z-score

BMC's taulukossa tätä kutsutaan autoskaalaukseksi (engl. autoscaling). Tämä on yleisin skaalausmenetelmä. Keskitettu data jaetaan keskihajonnalla, mistä lopputuloksena listan lukujen keskiarvo on 0 ja keskihajonta 1.

$$
{z} = \frac{\widetilde{x}}{s}
$$

```python
def z_score(x):
    return center(x) / std(x)
```

!!! tip

    Tulet törmäämään tähän usein eri koneoppimisesimerkeissä. Mikäli näet jossakin esimerkissä käytössä [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)-esikäsittelijän, se on juurikin tämä.

### Keskiarvon normalisointi

Englanniksi tämä on range normalization tai mean normalization. Kun keskitetty data jaetaan suurimman ja pienimmän arvon erotuksella, saadaan arvot pakotettua välille -1 ja 1.

$$
x' = \frac{ \widetilde{x} }{ max(x) - min(x) }
$$

In Python:

```python
def range_scale(x):
    return center(x) / (max(x) - min(x))
```

!!! tip

    Huomaa tämän variantti, min-max skaalaus, jonka avulla skaalataan arvot välille 0 ja 1. Tällöin kaava on seuraava:

    $$
    x' = \frac{ \widetilde{x} - min(x) }{ max(x) - min(x) }
    $$

