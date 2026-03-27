---
priority: 220
---

# Naive Bayes

Aiemmat kaksi lukua ovat luoneet hyvän pohjan tälle aiheelle: tiedät jo, kuinka Count Vectorizer toimii, ja ymmärrät myös tapahtumien riippumattomuuden sekä ehdollisen todennäköisyyden perusidean. Naive Bayes on todennäköisyyksiin perustuva (engl. *probabilistic*) luokittelija, joka tekee naiivin oletuksen siitä, että piirteet, kuten sanat, ovat ehdollisesti riippumattomia silloin, kun luokka tiedetään [^kämäräinen]. Käytännössä tämä tarkoittaa, että kun viestiä arvioidaan esimerkiksi roskapostina tai ei-roskapostina, mallin oletetaan tarkastelevan kunkin sanan vaikutusta erillään muista sanoista. Esimerkiksi sana *“ilmainen”* voi kasvattaa todennäköisyyttä, että viesti kuuluu roskapostiluokkaan. Roskapostin tunnistus onkin yksi Naive Bayesin tunnetuimmista käytännön sovelluksista. [^fromscratch]

Kämäräinen nostaa bayesilaista koneoppimista nostavat suomalaiset tutkijat esille. Heistä ensin mainittu on Aki Vehtari Aalto-yliopistosta. Kaksi muuta mainittua ovat Simo Särkkä ja Arno Solin. [^kämäräinen] Ethän siis oleta, että algoritmi olisi hyödytön nykypäivänä. Esimerkiksi Facebookin Prophet-aikasarjaennustin hyödyntää bayesilaista lähestymistapaa. [^fpppy].

!!! tip

    Käsittelemme tässä luvussa vain multinomiaalisen Naive Bayes luokittelijan, joka on erityisen suosittu tekstiluokittelussa. On olemassa muitakin bayesin teoreemaan perustuvia luokittelijoita (ja jopa regressoreita), mutta niitä ei käsitellä tässä luvussa. Näitä käsitellään esimerkiksi scikit-learn:n dokumentaatiossa.

## Teoreemasta luokittelijaksi

Päädyimme edellisessä luvussa Bayesin teoreemassa kaavaan, joka näkyy alla. Tällä kertaa olemme tosin korvanneet `A`:n sanalla belief ja `B`:n sanalla evidence. *Belief* tarkoittaa tarkasteltavaa hypoteesia (esim. roskaposti) ja *evidence* havaittua dataa (esim. tarkasteltavan viestin sanat).

$$
P(\text{belief} \mid \text{evidence}) = \frac{P(\text{evidence} \mid \text{belief}) \times P(\text{belief})}{P(\text{evidence})}
$$

Tämän voisi kirjoittaa myös seuraavalla tavalla:

$$
\text{posterior probability} = \frac{\text{likelihood} \times \text{prior probability}}{\text{evidence}}
$$

Kuten [scikit-learnin dokumentaatio esittää](https://scikit-learn.org/stable/modules/naive_bayes.html), jakaja on vakio annetulle syötteelle, joten sen voi jättää pois. Jakajan hylkääminen tekee lopputuloksesta normalisoimattoman, mutta se ei ole ongelma, koska meitä kiinnostaa vain suhteellinen todennäköisyys. Alla olevan kaavan kalan näköinen symboli luetaan "proportional to". Tämä tarkoittaa, että vasen puoli on suoraan verrannollinen oikean puolen tulon kanssa. [^scikit-bayes]

$$
P(\text{belief}|\text{evidence}) \propto P(\text{evidence}|\text{belief}) \times P(\text{belief})
$$

Voimme myös korvata kaavasta sanan `belief` sanalla `spam` ja sanan `evidence` useita datasetissä olevia sanoja korvaavalla `x_1, x_2, ..., x_n`.

$$
P(spam|x_1, x_2, ..., x_n) \propto P(x_1, x_2, ..., x_n|spam) \times P(spam)
$$

Kukin sana osallistuu todennäköisyyteen itsenäisesti (eli ehdollisesti riippumattomana annetun luokan suhteen). Multinomiaalisessa tekstiluokittelussa sanan vaikutus näkyy käytännössä sen esiintymismäärän kautta: useammin esiintyvä sana vaikuttaa enemmän kuin kerran esiintyvä sana. Toisin sanoen Count Vectorizer on hyvä pohjaoletus enkoodaustavaksi, joskin TF-IDF voi myös toimia [^scikit-bayes]. Tämän naiivin oletuksen kanssa luokittelusäännön kaava on:

$$
\hat{y} = \arg\max_{y} P(y) \prod_{i=1}^{n} P(x_i|y)
$$

Eri bayesilaiset luokittelijat toteuttavat tämän kaavan $P(x_i \mid y)$ osan eri tavoin. Alla avataan, miten multinomiaalinen Naive Bayes estimoituu opetusdatasta.

## Multinomiaalinen

Scikit-learn:n dokumentaatiossa Multinomial Naive Bayes esitellään kaavana, joka näkyy tämän tekstikappaleen alla. Kaavassa $y$ on luokka (0, 1) ja $i$ on sarake (eli meidän tapauksessa sana). Näin multinomiaalisen luokittelijan tarvitsema todennäköisyys $P(x_i|y)$ estimoidaan opetusdatasta. Termi $\theta_{yi}$ on todennäköisyys, että piirre $i$ (eli sana) esiintyy näytteessä, joka kuuluu luokkaan $y$. Eli siis $P(x_i|y) \approx \hat{\theta}_{yi}$.

$$
\hat{\theta}_{yi} = \frac{ N_{yi} + \alpha}{N_y + \alpha n}
$$

* $N_{yi}$ - kuinka monta kertaa kukin sana esiintyy luokan $y$ dokumenteissa
* $N_y$ - kaikkien sanojen määrä luokassa $y$
* $n$ - kaikkien sanojen yhteismäärä (eli *vocabulary*)
* $a$ - smoothing eli silotusparametri

Kun käyt läpi harjoituksessa annettua koodia, tulet yhdistämään yllä olevan kaavan seuraavaan koodin osaan. Alla koodin muuttujat on käännetty suomeksi ja siihen on lisätty kommentit.

```python
sanan_todennakoisyys_luokassa = {
    0: {}, # ei-roskaposti
    1: {}  # roskaposti
}
sana_maarat_luokassa = {
    0: {"lorem": 42, "ipsum": 32, "dolor": 2, "sit": 3, "amet": 5, "viagra": 3},
    1: {"lorem": 67, "ipsum": 33, "dolor": 0, "sit": 3, "amet": 5, "viagra": 1337}
}
luokan_sanat_totals = {0: 42+32+2+3+5+3, 1:67+33+0+3+5+1337}
n = 6   # sanaston koko
a = 0.5 # "siloitus" eli smoothing

for luokka in (0, 1):

    jakaja = luokan_sanat_totals[luokka] + (a * n)

    for sana in ("lorem", "ipsum", "dolor", "sit", "amet", "viagra"):
        jaettava = sana_maarat_luokassa[luokka][sana] + a
        sanan_todennakoisyys_luokassa[luokka][sana] = jaettava / jakaja
```

Kun tutustut koodiin, yksi äärimmäisen yksinkertainen ja tehokas tapa selvittää sen toimintaa on lisätä `print()`-komentoja eri kohtiin. Tämä on erityisen toimiva pienen datasetin kanssa. Yllä olevan koodin saa tulostamaan esimerkiksi:

```plaintext
Käsitellään luokkaa:  0
 jakaja= 90
  Käsitellään sanaa:  lorem
    jaettava= 42.5
    sanan_todennakoisyys_luokassa= 0.4722222222222222
  Käsitellään sanaa:  ipsum
    jaettava= 32.5
    sanan_todennakoisyys_luokassa= 0.3611111111111111
  Käsitellään sanaa:  dolor
    jaettava= 2.5
    sanan_todennakoisyys_luokassa= 0.027777777777777776
  Käsitellään sanaa:  sit
    jaettava= 3.5
    sanan_todennakoisyys_luokassa= 0.03888888888888889
  Käsitellään sanaa:  amet
    jaettava= 5.5
    sanan_todennakoisyys_luokassa= 0.06111111111111111
  Käsitellään sanaa:  viagra
    jaettava= 3.5
    sanan_todennakoisyys_luokassa= 0.03888888888888889
Käsitellään luokkaa:  1
 jakaja= 1448
  Käsitellään sanaa:  lorem
    jaettava= 67.5
    sanan_todennakoisyys_luokassa= 0.04661602209944751
  Käsitellään sanaa:  ipsum
    jaettava= 33.5
    sanan_todennakoisyys_luokassa= 0.0231353591160221
  Käsitellään sanaa:  dolor
    jaettava= 0.5
    sanan_todennakoisyys_luokassa= 0.0003453038674033149
  Käsitellään sanaa:  sit
    jaettava= 3.5
    sanan_todennakoisyys_luokassa= 0.0024171270718232043
  Käsitellään sanaa:  amet
    jaettava= 5.5
    sanan_todennakoisyys_luokassa= 0.003798342541436464
  Käsitellään sanaa:  viagra
    jaettava= 1337.5
    sanan_todennakoisyys_luokassa= 0.9236878453038674
```

!!! tip

    Termiä "smoothing" ei juuri tässä materiaalissa selitetty. Kenties tämän merkitys kannattaa tarkistaa lähteistä?


## Tehtävät

!!! question "Tehtävä: Naive Bayes from Scratch"

    Aja notebook `220_nb_from_scratch.py`. Tutustu koodiin ja vertaa sitä kurssin (sekä löytämiesi lähteiden) teoriaan.

    Voi olla suositeltavaa käyttää lähteenä esimerkiki StatQuest with Josh Starmeria. Hänen videonsa [Naive Bayes, Clearly Explained!!!](https://youtu.be/O2L2Uv9pdDA) on juuri niin selkeä kuin voi huutomerkkien määrästä uskoa.

    P.S. Huomaa, että tässä ratkaisussa on huijattu sinänsä, että tämä ei ole aivan *from scratch*. Meillä on apuna Polars. Jos haluat tutustua pelkkään Pythoniin nojaavaan toteutukseen, suosittelen etsimään käsiisi Joel Grusin kirjan *Data Science from Scratch* tuoreimman painoksen.

!!! question "Tehtävä: Vihapuheen luokittelu Naive Bayesillä"

    Avaa notebook `221_nb_hate_speech.py`. Lataa Notebookin tarvitsema CSV-tiedosto repositoriosta: [gh:t-davison/hate-speech-and-offensive-language](https://github.com/t-davidson/hate-speech-and-offensive-language) ja ==lue samassa hakemistossa oleva README==-tiedosto. Kouluta Naive Bayes classifier, joka tunnistaa, onko viesti vihapuhetta vai ei. Raportoi tulokset oppimispäiväkirjassasi. 

    !!! tip

        Tutustu repositoriossa linkattuun Cornellin yliopiston artikkeliin [Automated Hate Speech Detection and the Problem of Offensive Language](https://doi.org/10.1609/icwsm.v11i1.14955). Artikkelia ei ole pakollista lukea ajatuksella tämän kurssin puitteissa, mutta artikkelia silmäilemällä saat hyvää osviittaa siitä, miltä projektikurssin raportti voi näyttää, jos tehty työ on luokittelumalli. Kannattaa vähintään katsoa kuvaajat läpi, ja miettiä, voiko niistä oppia jotakin jo tämän harjoituksen toteutukseen.

    Alla on pari sääntöä ja ohjaavaa väitettä, jotta työ pysyy kurssin aihealueissa ja laajuudessa sekä edistää sinun oppimistasi:

    * Älä anna tekoälyn kirjoittaa mitään koodia, mitä et osaa itse selittää ja puolustaa.
    * Älä tee *from scratch*-toteutusta (ellei sinulle ole kymmeniä tunteja ylimääräistä vapaa-aikaa).
    * Älä käytä edistynyttä stemmausta, lemmatisointia, word2veciä ynnä muuta.

    Mikä sitten on sallittua?

    * ✅ Notebook on nyt sinun Notebook. Muokkaa, miten haluat.
    * ✅ Muuta hyperparametreja ja selvitä eri asetusten vaikutus mallin suorituskykyyn.
    * ✅ Käytä kielimallia **haastamaan** sinun ajatteluasi.
    * ✅ Suosi ymmärrettäviä menetelmiä.

    Korostan vielä, että lue tehtävänanto dataan liittyvissä ongelmissa huolellisesti. Esimerkiksi yllä on esitelty, että sinun tulee tunnistaa, onko viesti vihapuhetta vai ei. Vastaus on siis "Kyllä on" tai "Ei ole". **Älä siis tee** luokitusta, jossa on useampia luokkia, kuten "vihapuhe", "loukkaava puhe", "ei kumpikaan". 
    
    !!! tip
    
        Voi olla perusteltua kokeilla, kuinka koulutetun mallin tarkkuus reagoi riippuen siitä, mitä teet *offensive*-luokan twiiteille. Sinulle on ainakin kolme vaihtoehtoa:

        1. Jätät ne pois kokonaan.
        2. Sisällytät ne ei-vihapuheeksi
        3. Sisällytät ne vihapuheeksi.

        Huomaa, että kohdassa kolme kouluttamastasi mallista tulee *is offensive or worse*-luokittelija, jolloin ongelma on eri kuin tehtävänannon alkuperäinen.

## Lähteet

[^kämäräinen]: Kämäräinen, J. *Koneoppimisen perusteet*. Otatieto. 2023.
[^fromscratch]: Grus, J. *Data Science from Scratch 2nd Edition*. O'Reilly Media. 2019.
[^fpppy]: Hyndman, R. et. al. *Forecasting: Principles and Practice, the Pythonic Way*. https://otexts.com/fpppy/
[^scikit-bayes]: Scikit-learn. *Naive Bayes*. https://scikit-learn.org/stable/modules/naive_bayes.html
