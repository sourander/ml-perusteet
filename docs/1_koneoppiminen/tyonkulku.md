---
priority: 140
---

# Työnkulku

Ennen kuin tätä ohjetta seuraa pidemmälle, pitäisi olla selvillä, **miksi** kyseistä mallia ollaan ottamassa käyttöön. Toisin sanoen esivaatimus koko touhulle on vaihe, jossa ongelma ja siihen toivottu ratkaisu määritellään.

Olettaen, että ongelma on jo määritelty, niin vaiheet voi jakaa muun muassa alla esitellyllä tavalla. Listaus löytyi kurssikirjan edeltäjän liitteistä, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd Edition*, vuodelta 2022 [^geron3rd]. Tuoreimmasta versiosta löytyy yhä lista, mutta ei enää liitteistä, vaan se on ulkoistettu [gh:ageron/handson-mlp/ml-project-checklist-md](https://github.com/ageron/handson-mlp/blob/main/ml-project-checklist.md)-tiedostoon online. Lista on vapaasti suomennettuna:

1. Määrittele ratkaistava ongelma
2. Hanki data
3. Tutki dataa saadaksesi ymmärrystä
4. Esikäsittele data algoritmille sopivaksi
5. Kokeile useita malleja ja valitse parhaat jatkoon
6. Säädä mallien hyperparametreja
7. Esittele ratkaisusi
8. Mallin käyttöönotto

!!! warning

    Huomaa, että vaiheet ovat iteratiivisia ja niillä on keskenään vahvoja riippuvuuksia. Jos vaiheista piirtäisi kaavion, niin kukin vaihe sisältäisi takaisinkytkentöjä edellisiin vaiheisiin. Esimerkiksi mallin evaluoinnissa voi tulla ilmi, että malli ei saavuta riittävää tarkkuutta ilman lisädatan keräämistä. Käytännössä tällöin palataan takaisin lähtöruutuun (joskin hieman viisaampana).

Kurssikirjan luku 2 [^geronpytorch] käy nämä vaiheet läpi käyttäen California Housing -datasettiä esimerkkinä. Tässä materiaalissa ei toisinneta tätä tekstiä, eikä kyseinen kirja ole lähteenä seuraaville vaiheille, vaan ne on kirjoitettu omasta kokemuksesta ja muista lähteistä. Ne toimivat täydentävänä materiaalina. Tämän luvun, ja samalla ensimmäisen opiskeluviikon, lopettaa tehtävä, jossa sinun tulee lukea `140_first_model.py`-tiedostoon toteutettun Titanic-luokittelija. Näin pääset ensimmäisellä viikolla näkemään kaksi end-to-end -projektia: regressio-ongelma (California Housing) ja luokittelu-ongelma (Titanic). On ymmärrettävää, että ensimmäisellä viikolla osa aihepiiristä tulee menemään yli hilseen: keskity kokonaiskuvan ymmärtämiseen, älä yksityiskohtiin. Seuraavilla viikoilla sukelletaan algoritmeihin ja datan käsittelyyn tarkemmin.

## Vaiheet

### 1. Määrittele ratkaistava ongelma

Tämä on varsinkin myöhemmillä projektikursseilla sekä opinnäytetyössä äärimmäisen tärkeää. Jos et osaa selittää tiivisti, minkä ongelman koneoppimismallisi valmiissa muodossaan ratkaisee, olet todennäköisesti hakoteillä. Hyvä tapa tutustua tähän on etsiä sinua kiinnostavan aihepiirin tutkimus käsiisi ja lukea sen *Johdanto* eli *Introduction*.

1. Navigoi [Google Scholariin](https://scholar.google.com/)
2. Etsi esimerkiksi `"horror games" AND "deep learning"` tai `"Red Palm Weevil" AND "deep learning"`
3. Avaa PDF-linkki
 
Jos PDF:ään ei ole vapaata pääsyä, etsi seuraava. Voit etsiä myös *machine learning* aiheella, mutta näin 2020-luvulla tuoreimmat julkaisut ovat usein syväoppimiseen perustuvia. Ratkaistavaan ongelmaan tutustuessa ei ole sinänsä väliä, millä keinoin se on ratkaistu.

### 2. Hanki data

Koneoppimismallin rakentamiseen tarvitaan dataa. Datan pitää olla valitun ongelman kannalta merkitsevää. Ongelma on, että usein datan merkitsevyys ei ole itsestään selvää vaan paljastuu vasta myöhemmissä vaiheissa. Datan kerääminen on kuitenkin tiukasti säädeltyä (GDPR, AI Act).

Dataa voidaan kerätä julkisista lähteistä, oman palvelun käyttäjiltä, kyselyillä, sensoreilta, jne. Datan keräämisessä tehdyt ==virheet kostautuvat kaikissa seuraavissa vaiheissa==. Yrityksen data voi hyvin sijaita siiloutuneena eri järjestelmissä, jolloin datan yhdistäminen ja puhdistaminen voi olla yllättävänkin suuritöistä. Sivuoireena tästä saattaa syntyä data-alusta, joka tuottaa yritykselle arvoa myös muilla tavoin kuin koneoppimismallien kautta.

Tähän vaiheeseen saattaa kuulua myös datan annotointi tai merkitseminen. Annotointi on työläs vaihe, jossa ihmiset merkkaavat, että onko kuvassa esimerkiksi kissa vai koira, tai onko sähköposti roskapostia vai ei. Laadukkaan, labelöidyn datan puute on herkästi pullonkaula. Projekti voi siis tyssätä jo alkuvaiheisiin.

!!! tip

    Joskus annotointi saadaan joukkoistettua, jolloin useat ihmiset merkkaavat samaa dataa ja virheet voidaan havaita ja korjata. Jos olet joskus vastannut CAPTCHA-kyselyihin, olet todennäköisesti ollut osa joukkoistettua annotointia. Myös "Minkä ikäiseltä näytät?" -tyyppiset some-lelut ovat joukkoistettua annotointia. Kun merkkaat Gmailissa sähköpostia roskapostiksi, olet osa joukkoistettua annotointia.

### 3. Tutki dataa

TODO

### 4. Esikäsittele data algoritmille sopivaksi

Datan esikäsittelyssä käsitellään dataa ennen koneoppimismallin rakentamista. Tähän voi kuulua esimerkiksi puuttuvien arvojen täyttäminen, datan normalisointi, piirteiden yhdistäminen (esim. `etäisyys / aika = nopeus`), kategorisen datan enkoodaus useiksi binääripiirteiksi ja niin edelleen. Kenties datasetti sisältää päivämääriä, jotka ovat tulevaisuudessa. Se, kuinka näihin virheisiin reagoidaan, vaatii ymmärrystä siitä, kuinka data on kerätty, ja siitä, mitä data edustaa (eli substanssiosaamista.)

Datan käsittely ei ole mekaaninen vaihe vaan vaatii ymmärrystä datan luonteesta. Tähän vaiheeseen sisältyy myös eksploratiivinen analyysi, jossa pyritään ymmärtämään datan piirteitä ja mahdollisia ongelmia. Esimerkiksi jatkuvien lukujen jakautumista voidaan tarkastella histogrammien avulla ja eri piirteiden välisiä korrelaatioita voidaan tarkastella korrelaatiomatriisin avulla. Tätä piirteiden yhdistämis- ja luomisprosessi tunnetaan englanninkielisellä termillä *feature engineering*.

Datan käsittelyn aikana tulisi muun muassa selvittää ja dokumentoida, kuinka tasapainoista tai vinksahtanutta data on. Jos data on epätasapainoista, esimerkiksi 99 % negatiivisia ja 1 % positiivisia tapahtumia, mallin rakentaminen voi olla haastavampaa. Tällöin malli voi oppia ennustamaan kaikki tapahtumat negatiivisiksi ja silti saavuttaa korkean tarkkuuden.

Alla koneoppimismalli, joka ennustaa reilusti yli 99 % tarkkuudella, että voittaako kyseinen käyttäjä huomenna Lotto-arvonnassa päävoiton.

```python title="IPython"
def is_lotto_winner_tomorrow(user_id):
    return False
```

### 5. Kokeile useita malleja ja valitse parhaat jatkoon

Koneoppimismallin valinta riippuu siitä, millaista dataa on saatavilla ja mitä halutaan ennustaa. Katso aiempi luku [Tyypit](tyypit.md) kertauksena eri koneoppimismallityypeistä. On kuitenkin tärkeää korostaa, että data ja sen laatu vaikuttavat lopputulokseen enemmän kuin käytetty malli.

> It [his observation] implies that model behavior is not determined by architecture, hyperparameters, or optimizer choices.It’s determined by your dataset, nothing else. Everything else is a means to an end in efficiently delivery compute to approximating that dataset.
> 
> Then, when you refer to “Lambda”, “ChatGPT”, “Bard”, or “Claude” then, it’s not the model weights that you are referring to. It’s the dataset. 

**Lähde**: James Betker (Open AI), kirjoitus: [The “it” in AI models is the dataset](https://nonint.com/2023/06/10/the-it-in-ai-models-is-the-dataset/).

Scikit-learn tarjoaa vakaan API:n, joka mahdollistaa useiden eri mallien kouluttamisen ja käytön samoilla `.fit()` ja `.transform()` metodeilla. Mallien vertailun on siis kehittäjälle vaivatonta. Tästä syystä vaiheet 3-5 ovat usein iteratiivisia ja niiden välillä on paljon takaisinkytkentää. Mikäli käytetty data ei ole aivan big data -kokoluokassa, mallien kouluttaminen on usein kohtalaisen nopeaa, joten on täysin mahdollista vertailla useita eri malleja ja niiden hyperparametreja, vaikka näistä tulisikin melko suuri määrä testattavia kombinaatioita.

### 6. Mallin tuunaus

Koneoppimismallin tuunausta (engl. hyperparameter tuning) käytetään mallin suorituskyvyn parantamiseen.

Aivan kuten mallin koulutuksessa, myös hienosäädössä käytetään koulutusdataa, jotta voidaan varmistaa, ettei mallin hyperparametreja optimoida testidatan suhteen. Tavoite on, että hyperparametrien arvot ovat sellaiset, että malli yleistyy parhaiten uuteen dataan - ei se, että se toimii parhaiten koulutusdatalla.

Parametrien tuunaus voi tapahtua esimerkiksi ristivalidoinnin (engl. cross-validation) avulla. Ristivalidoinnissa data jaetaan useaan osaan, joista osa käytetään koulutukseen ja osa testaukseen. Tätä toistetaan useita kertoja, jolloin saadaan luotettavampi arvio mallin suorituskyvystä.

Pseudoesimerkki alla. Esimerkkiä lukiessa sinun ei tarvitse tietää, mitä Lasso-malli tekee. Riittää, että hyväksyt, että  `alpha`-arvon on hyperparametri, ja sen arvon valinta vaikuttaa merkittävästi siihen, kuinka hyvin malli ennustaa. Hyperparametrien tuunaus on näiden arvojen haarukoimista. Jos haarukoitavia arvoja on monta (kuvitteellisina esimerkkeinä alpha, beta, gamma, ...), niin kaikki näiden kombinaatiot pitää testata.

```python title="IPython"
alpha_grid = [0.1, 0.5, 1.0, 1.5, 2.0]

for alpha in alpha_grid:
    clf = linear_model.Lasso(alpha=alpha)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"Alpha: {alpha}, Scores: {scores}")
```

### 7. Esittele ratkaisusi

TODO!

### 8. Mallin käyttöönotto

Käyttöönotto (engl. deployment) on vaihe, jossa hyödylliseksi evaluoitu ja hienosäädetty malli otetaan käyttöön. Kevyt malli voidaan esimerkiksi julkaista yksinkertaisen REST-rajapinnan taakse (esim. Python FastAPI), jolloin se on helposti käytettävissä. Raskaampi malli voidaan julkaista hajautettuun järjestelmään, jossa kuormantasaajan takana olevat yksiköt skaalautuvat automaattisesti käyttäjämäärän mukaan (esim. Kubernetes).

Käytännön tasolla projekti ei kuitenkaan ole välttämättä vielä ohi. Mallista voi mennä niin sanotusti parasta ennen päivä umpeen (engl. model decay, model rot, model drift). Tältä voi välttyä keräämällä jatkuvasti dataa ja päivittämällä mallia säännöllisesti. Toisin sanoen yllä olevista vaiheista tulee loop. Ideaalitilanteessa loppukäyttäjät osallistuvat virheellisten ennusteiden merkintään, jolloin he osallistuvat mallin jatkokehitykseen (ks. vaihe 1).

## Tehtävät

TODO!

## Lähteet

[^geron3rd]: Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd Edition*. O'Reilly. 2022.
[^geronpytorch]: Géron, A. *Hands-On Machine Learning with Scikit-Learn and PyTorch*. O'Reilly. 2025.