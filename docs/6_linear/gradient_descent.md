---
priority: 630
---

# Gradient Descent

Edellisessä luvussa käsittelimme Hill Climbing algoritmia. Algoritmin koulutus koostui n-määrästä iteraatioita, joissa kussakin arvottiin sattumanvaraiset muutokset painoille ja laskettiin niiden perusteella mallin virhe. Eikö oliskin varsin kätevää, jos satunnaisuuden sijasta voisimme laskea, mihin suuntaan painoja tulisi muuttaa. Tämän meille mahdollistaa Gradient Descent algoritmi! Materiaalissa käytämme nimenomaan *eräajo*-versiota algoritmista – *vanilla*-toteutusta – eli Batch Gradient Descentia, joka laskee gradientin ==koko datasetin== perusteella. Tämä on hyvä vertailukohta Hill Climbing -algoritmille, joka myös arvioi muutokset koko datasetin perusteella. Materiaalin lopussa on lyhyesti esiteltynä myös Stochastic Gradient Descent, joka laskee gradientin vain yhden havainnon perusteella, ja Mini-batch Gradient Descent, joka laskee gradientin pienen otoksen perusteella [^dl4cv]. Näistä jälkimmäinen on tehokäytössä Syväoppiminen I -kurssilla. 

!!! warning

    Heti alkuun on hyvä mainita ero, joka liittyy algoritmin käyttöön eri konteksteissa: lineaarisessa regressiossa ja syväoppimisessa. Aiemmissa kurssin toteutuksissa on huomattu, että opiskelijat alkavat herkästi kirjoittaa päiväkirjoissaan aiheesta *getting stuck in local minima* – eli ajatus on, että liian pienen oppimisnoepuden vuoksi algoritmi ei löydä globaalia optimia eli ns. oikeaa vastausta. Tämä on totta syväoppimisen kontekstissa neuroverkkojen epälineaarisuuden vuoksi: lineaarisen regression yhteydessä ns. liian pieni oppimisnopeus tarkoittaa vain ja ainoastaan sitä, että prosessissa kestää kauan.

    > "(...), we end up with a convex optimization problem, which implies that any local minimum is also a global minimum. This means that as long as we pick the learning rate to be small enough, gradient descent will always converge to the optimum solution." [^ldl]
    >
    > — Magnus Ekman

    > "In the case of NNs, however, the MSE is a nonlinear function of parameters with many local optima. Therefore, gradient‐based optimization algorithms can only find local optima." [^surrogate]
    >
    > — Nam-Ho Kim

## Vertailu Hill Climbingiin

Gradient Descent -algoritmin koulutuksen vaiheet myötäilevät Hill Climbing -algoritmin vaiheita, mutta painojen muutokset lasketaan tavalla, joka esitellään tässä dokumentissa. Verrataan näitä vaiheita Hill Climbing -algoritmin vastaaviin taulukkomuodossa.

Muistutuksena aiemmat vaiheet olivat:

1. Alusta kertoimet satunnaisesti
2. Laske virhe
3. Lisää kertoimiin satunnaisluku (välillä ±1.0)
4. Laske virhe
5. Jos virhe pienenee, hyväksy muutos
6. Toista 3-5 kunnes pysäytyskriteeri täyttyy

| Vaihe | Hill Climbing                                 | Gradient Descent                           |
| ----- | --------------------------------------------- | ------------------------------------------ |
| 1     | Alusta kertoimet satunnaisesti                | Alusta kertoimet satunnaisesti             |
| 2     | Ennusta ja laske MSE/SSE                      | Ennusta ja laske MSE/SSE                   |
| 3     | Lisää kertoimiin satunnaisluku (välillä ±1.0) | Laske gradientti                           |
| 4     | Laske virhe                                   | Kerro gradientti oppimisnopeudella         |
| 5     | Hyväksy tai hylkää uudet kertoimet            | Päivitä kertoimet                          |
| 6     | Toista 3-5 kunnes pysäytyskriteeri täyttyy    | Toista 2-5 kunnes pysäytyskriteeri täyttyy |

Huomaa, että vaiheet ovat pääosin samat, mutta seuraavan termit saattavat olla sinulle vieraita: ==osittaisderivaatta, gradientti ja oppimisnopeus==. Näistä kaksi ensimmäistä ovat differentiaalilaskennan käsitteitä.

!!! tip  

    Tämä ei ole matematiikan kurssi, joten emme derivoi yhtäkään funktiota käsin emmekä täten tarvitse derivointisääntöjä. Keskitymme ilmiön ymmärtämiseen intuition tasolla. Syväoppiminen I -kurssilla tulet oppimaan, että automaattinen derivointi (autograd) on syväoppimiskirjastojen kuten PyTorchin ja TensorFlow:n keskeisimpiä ominaisuuksia.

## Kulmakertoimen selvittäminen

### Diabetes ja deltametodi

Tutustumme algoritmin toimintaan Scikit-learn kirjaston [Diabetes datasetin](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) avulla, joka sisältää 10 piirrettä (age, sex, bmi, ...). Aloitamme arpomalla kertoimet satunnaisesti (Vaihe 1). Kertoimiksi tai painoiksi valikoituvat alla näkyvässä taulukossa olevat arvot. Mallin ennuste tehtäisiin näiden painojen avulla (`y = w1*x1 + w2*x2 + ... + w10*x10`). Vektori `w` on tuttuun tapaan yhtä pitkä kuin havaintomatriisin kukin rivi eli featureiden määrä. Jos bias otetaan mukaan, malli on muotoa `y = w0 * w0 + ....`

| age  | sex   | bmi  | bp   | s1    | ... |
| ---- | ----- | ---- | ---- | ----- | --- |
| 0.50 | -0.14 | 0.65 | 1.52 | -0.23 | ... |

Tutustutaan kulmakertoimeen yhden piirteen avulla. Tämä piirre on `weights[2]` eli bmi eli painoindeksi (engl. body mass index). Pidämme toistaiseksi muut piirteet vakiona. Näin voimme laskea, kuinka mallin virhe (MSE) muuttuu, kun vaihdetaan arvoa valitulla välillä, joka on tässä esimerkissä etukäteen päätetty väli `-100` ja `180`. Tämä vaihteluväli on valittu siten, että graafista tulisi tässä tapauksessa mukavan symmetrinen.

```python title="IPython"
def mse_by_weight_range(feature_index, X, y, weights, values):
    mse_values = []
    weights_copy = weights.copy()

    for value in values:
        weights_copy[feature_index] = value
        y_hat = X.dot(weights_copy)
        mse = mean_squared_error(y, y_hat)
        mse_values.append(mse)

    return mse_values

feature_index = 2                      # bmi
values = np.linspace(-100, 180, 100)   # x-axis values
mse_values = mse_by_weight_range(
    feature_index, X, y, weights, values
)
```

Jos palautuneista `mse_values`-arvoista piirretään kuvaaja, saadaan paraabeli, joka kuvaa virheen muutosta `bmi`-arvon painokertoimen muuttuessa.

![Convex of MSE by bmi](../images/630_gradient_descent_bmi_convex.png)

**Kuvio 1:** *Mallin virheen muutos painoindeksin painokertoimen muuttuessa. Nykyinen arvottu painokerroin, `0.65`, on merkattu kuvaajaan punaisena katkoviivana.*

!!! warning

    Huomaa, että vaikka kuvaajan perusteella vaikuttaa, että `weights[2] == 45` minimoi virheen, niin olisi epäoptimaalista loikata suoraan kuopan pohjalle. Yhtä muuttujaa arvioidessa muut pidetään vakiona - mutta käytännössä ne vaikuttavat kokonaisuuteen. ==Suunta== on kuitenkin todennäköisesti oikea. Tätä varten oppiminen tapahtuu pienin askelin. Tämä oppimisen nopeus on `learning_rate` ja se esitellään myöhemmin.

![Convex of MSE by bmi zoomed in](../images/630_gradient_descent_bmi_convex_zoom.png)

**Kuvio 2:** *Kuvio 1:n lähikuvaaja, jossa näkyy paremmin virheen muutos. Huomaa, että x-akseli kattaa nyt vain arvot `0.6-0.7`*

![Convex of MSE with dy/dx](../images/630_gradient_descent_bmi_convex_slope.png)

**Kuvio 3:** *Kuvio 2:n lähikuvaaja, jossa esitellään pieni delta (`+ 0.01`), ja sen vaikutus virheeseen. Virhefunktion kulmakertoimen voisi laskea myös näin. Pieni delta on valittu sattumanvaraisesti.*

Huomaa, että vaikka Kuviossa 3 käyrä näyttää ihmissilmälle suoralta, se on yhä kaareva. Mitä pienemmän deltan avulla laskemme muutoksen, sitä tarkemman arvon saamme kulmakertoimesta. Alla olevassa taulukossa näkyy kulmakertoimen laskeminen eri deltoilla. Ensimmäinen sarake on siis askeleen koko, joka otetaan Kuvion 3 x-akselilla. Valitut askeleet ovat kymmenesosia toisistaan (`10 ** -2`, `10 ** -3`, `10 ** -4`, `10 ** -5`).

| Weight delta | MSE delta | Slope         |
| ------------ | --------- | ------------- |
| 0.01000      | -0.88411  | -88.410579813 |
| 0.00100      | -0.08842  | -88.419579813 |
| 0.00010      | -0.00884  | -88.420479769 |
| 0.00001      | -0.00088  | -88.420569591 |

!!! info

    Derivointi on seuraavan otsikon aihe, mutta otetaan aikahyppy tulevaisuuteen toistaiseksi vieraan `magic()`-funktion toiminnallisuuden avulla. Toistaiseksi riittää, että hyväksyt, että `slope = magic()`-rivi palauttaa koko datasetin perusteella lasketun gradientin: yhden kulmakertoimen jokaista painokerrointa kohti. Ajattele tätä niin, että kaikki havainnot ‘äänestävät’, mihin suuntaan kutakin painoa pitäisi siirtää.

    ```python title="IPython"
    def magic(X, y, weights):
        y_hat = X.dot(weights)
        slope = 2 * X.T.dot(y_hat - y) / len(X)
        return slope

    slope = magic(X, y, weights)[feature_index]
    print(f"Slope is: {slope:.9f}")
    ```

    ```plaintext title="stdout"
    Slope is: -88.420579813
    ```

### Feikkidata ja derivointi

Diabetes-datasetissä on merkittävä määrä muuttujia (10 kpl). Vaihdetaan yhden muuttujan ja kahdeksan havainnon `X.shape == (8, 1)` esimerkkiin matematiikan helpottamiseksi. Kun `X:ään` lisätään bias, sen muodoksi tulee `(8, 2)`. Alla koodi, jolla data on generoitu, ja X_bias-matriisi taulukkona.

```python title="IPython"
import numpy as np

# Generate reproducible noise
np.random.seed(42)
noise_delta = 0.05
noise = np.random.normal(0, noise_delta, 8).round(2)

# Data
X = np.array(list(range(8))).reshape(8, -1)

# Target (1)
y = -0.25*X + 1.5
y = y.flatten() + noise

# Add bias
X_bias = np.c_[np.ones(X.shape[0]), X]

# Manually set wrong weights (2)
w = np.array([-0.5, 0.5])

# Predict y_hat
y_hat = np.dot(X, w)
```

1. Huomaa, että `y` on generoitu suoraan kaavalla `y = -0.25x + 1.5`.
2. Painot ovat valittu sattumanvaraisesti. Oikeat painot olisivat `[1.5, -0.25]`, koska `y = -0.25x + 1.5`.

| Havainnon # | Bias (x_0) | Feat (x_1) | y     | y_hat |
| ----------- | ---------- | ---------- | ----- | ----- |
| Ensimmäinen | 1.0        | 0.0        | 1.52  | -0.50 |
| Toinen      | 1.0        | 1.0        | 1.24  | 0.00  |
| Kolmas      | 1.0        | 2.0        | 1.03  | 0.50  |
| Neljäs      | 1.0        | 3.0        | 0.83  | 1.00  |
| Viides      | 1.0        | 4.0        | 0.49  | 1.50  |
| Kuudes      | 1.0        | 5.0        | 0.24  | 2.00  |
| Seitsemäs   | 1.0        | 6.0        | 0.08  | 2.50  |
| Kahdeksas   | 1.0        | 7.0        | -0.21 | 3.00  |

!!! note

    Huomaa, että emme skaalaa piirteitä, koska tämä on yksinkertainen esimerkki. Ethän toimi näin oikeassa elämässä! Oikeassa mallinnuksessa piirteiden skaalaus on gradienttipohjaisissa menetelmissä yleensä erittäin hyödyllistä, koska se helpottaa oppimisnopeuden valintaa ja nopeuttaa konvergenssia.

Ennuste on luonnollisesti väärä, koska painot ovat hyvin kaukana siitä, mitä niiden pitäisi olla. Piirre itsessään tulisi kertoa `-0.25`:lla, mutta se kerrotaan `0.5`:lla. Bias tulisi kertoa `1.5`:lla, mutta se kerrotaan `-0.5`:lla. Tämä on hyvä esimerkki siitä, miten painot vaikuttavat ennusteeseen.

![Lines at epoch 0](../images/600_gradient_descent_lines_at_epoch_0.png)

**Kuvio 4:** *Kuvaaja, jossa on feikkidatan datapisteet sinisinä ympyröinä, matemaattinen ideaali punaisena viivana, ja ennuste sinisenä viivana. Ennuste on merkitty sinisellä viivalla ja oikeat arvot punaisilla pisteillä.*

#### Verifioidaan deltametodilla

Voimme laskea kulmakertoimen yllä opitulla tavalla, eli tehdään pieni muutos painokertoimeen ja lasketaan virheen muutos. Tämä laskenta on alla piilotetussa solussa.

??? note "Koodi: Kulmakertoimen laskenta"

    ```python title="IPython"
    def mse(y, y_hat):
        return np.mean((y - y_hat)**2)

    def compute_slope_using_delta(X, y, w, delta=0.0000001):
        slopes = {}
        for i in range(len(w)):
            w_delta = w.copy()
            w_delta[i] += delta
            y_hat = predict(X, w_delta)
            slope = (mse(y, y_hat) - mse(y, predict(X, w))) / delta
            slopes[i] = round(slope, 2)
        return slopes

    compute_slope_using_delta(X_bias, y, w)
    ```

    ```plaintext title="stdout"
    {0: 1.2, 1: 12.01}
    ```

Tulokseksi syntyy luvut: **1.20** ja **12.01**. Ensimmäinen luku on biasin kulmakerroin ja toinen on piirteen kulmakerroin. Tarvitsemme näitä jatkossa varmentaaksemme derivoinnin oikeellisuuden.

#### SSE/MSE:n osittaisderivaatat

Käytän tässä derivoinnissa SSE:tä, koska kaava näyttää hieman siistimmältä. MSE saadaan samasta asiasta jakamalla havaintojen määrällä, joten optimoinnin intuitio ei muutu.

Sen sijaan, että derivoisimme virhefunktion käsin, käytämme SymPy-kirjastoa. Tähän löytyy hyvät ohjeet Essential Math for Data Science [^essential-math-for-ds] -kirjasta. SymPy-kirjaston `.diff()`-metodi derivoi funktion annetun muuttujan suhteen. Tässä tapauksessa meillä on kaksi muuttujaa, painokertoimet `w0` ja `w1`.

```python title="IPython"
import sympy as sp

def solve_partial_derivative_formula():

    y, w0, w1, x0, x1 = sp.symbols('y, w0, w1, x0, x1')

    # Sum of squared errors
    cost = (y - (w0 * x0 + w1 * x1))**2 # (1)

    print(cost.diff(w0).simplify())
    print(cost.diff(w1).simplify())

solve_partial_derivative_formula()
```

1. Virhefunktio on neliösumma. Jos haluaisimme tämän keskiarvon eli MSE:n, lisäisimme jakolaskun `m`:llä, jossa `m` on havaintojen määrä. Me teemme tämän myöhemmässä vaiheessa.

```plaintext title="stdout"
2*x0*(w0*x0 + w1*x1 - y)
2*x1*(w0*x0 + w1*x1 - y)
```

!!! tip

    Joissakin materiaaleissa SSE/MSE puolitetaan, jotta derivaatasta putoaa `2 *` pois. Jos näet kirjallisuudessa kaavan kyseisessä muodossa, tässä on syy.


Nyt kun tiedämme derivaatan kaavan, voimme luoda funktion, joka suorittaa kyseisen laskennan.

```python title="IPython"
def partial_derivative_full_sse(w, x, y, feature_index, observation_index):
    # With respect to w0 or w1
    # as in: 2*x0*(w0*x0 + w1*x1 - y)
    #    or: 2*x1*(w0*x0 + w1*x1 - y)

    # Long form
    # derivative = 2 * x[feature_index] * (w[0] * x[0] + w[1] * x[1] - y[observation_index])

    # Short form
    return 2 * x[feature_index] * (np.dot(w, x) - y[observation_index])

for obs_i, x in enumerate(X):
    part_0 = partial_derivative_full_sse(w, x, y, 0, obs_i)
    part_1 = partial_derivative_full_sse(w, x, y, 1, obs_i)

    print(f"Observation {obs_i} {x[0]=} {x[1]=} => {part_0=:.2f}, {part_1=:.2f}")
```

Raa'an tulosteen sijasta esitän saman datan taulukkomuodossa, jotta sitä on helpompi tulkita. Taulukossa on riveinä yksittäisten havaintojen derivaatat sekä näistä koostettu (engl. aggregrated) tulos. Kun tämä tulos jaetaan havaintojen määrällä, saadaan keskiarvo (engl. mean) ja tämä on osittaisderivaatta.

| X              | Derivative w_0 | Derivative w_1 |
| -------------- | -------------- | -------------- |
| [1, 0]         | -4.04          | -0.00          |
| [1, 1]         | -2.48          | -2.48          |
| [1, 2]         | -1.06          | -2.12          |
| [1, 3]         | 0.34           | 1.02           |
| [1, 4]         | 2.02           | 8.08           |
| [1, 5]         | 3.52           | 17.60          |
| [1, 6]         | 4.84           | 29.04          |
| [1, 7]         | 6.42           | 44.94          |
| ::::           | ::::           | ::::           |
| TOTAL          | 9.55           | 96.08          |
| MEAN (total/8) | 1.19           | 12.01          |


!!! tip

    Derivaatta kertoo, miten nopeasti virhe muuttuu juuri nykyisen painon ympäristössä, jos yhtä painoa kasvatetaan hieman ja muut pidetään vakioina. Yhden yksikön muutos painokertoimeen `w_0` kasvattaisi SSE:tä noin 9.56 yksikköä, jos muut pidetään vakioina. Sama muutos `w_1`:een aiheuttaa noin `96.08` muutoksen neliövirheiden summaan. MSE:n kohdalla nämä jaettaisiin yksinkertaisesti havaintojen määrällä.

Näistä osittaisderivaatoista pääsemmekin kiinni termiin `gradientti`. Osittaisderivaatta kertoo yhden painon suunnan. Gradientti on vektori, joka kokoaa nämä kaikki suunnat yhteen. Eli siis jos otat yllä olevasta taulukosta luvut `9.55` sekä `96.08` ja muodostat niistä vektorin, se on gradientti SSE:n suhteen. MSE:n gradientti olisi vastaavasti vektori `(1.19, 12.01)`.

!!! note

    Huomaa, että nämä arvot ovat pyöristysvirheitä lukuunottamatta samat kuin yllä deltametodilla lasketut arvot.

### MSE:n vektorimuotoinen gradientti

Silmukoiden sijaan voimme hyödyntää matriisilaskentaa ja laskea gradientin joko muutamassa vaiheessa tai jopa yhdellä rivillä. Alla on kumpikin muoto esiteltynä.

```python title="IPython"
# Note that the derivate is computed as:
#
#    multiply each feature...
#     ┌───────┴──────┐
# 2 * x[feature_index] * (np.dot(w, x) - y[observation_index])
#                        └─────────────────┬─────────────────┘
#                               ...with the residuals
y_hat = X @ w
residuals = y_hat - y
sum_of_partials = 2 * X.T @ residuals

# We need to divide by the number of observations to get the mean
gradient = sum_of_partials / len(y)
print("The gradient: ", gradient)

#### OR #######
# ONE-LINER ! #
###############
gradient = 2 * X.T @ (X @ w - y) / len(y)
print("The gradient with one-liner: ", gradient)
```

```plaintext title="stdout"
The gradient:  [ 1.195 12.01 ]
The gradient with one-liner:  [ 1.195 12.01 ]
```

Nyt kun olemme laskeneet gradientin, voimme hyödyntää sitä. Meidän pitää kuitenkin vielä käsitellä aiheen viimeinen osa: oppimisnopeus.

## Oppimisnopeus

### Kakku metsän pohjalla

Kuvittele valtava, pimeä metsä. On yleisesti tiedossa, että metsän syvimmässä kohdassa on kakku :cake:. Ja sinä tietenkin haluat sen! Tehtävä olisi helppo, jos sinulla olisi kartta ja kompassi. Näitä sinulla ei ole, joten joudut luottamaan aisteihisi.

Käytät navigointiin taskulamppua :flashlight:. Taskulampun paristot ovat vähissä, joten haluat käyttää sitä mahdollisimman harvoin. Siispä valitset strategian: pysähdyt paikoillesi, laitat valon päälle, ja arvioit, missä suunnassa on alamäki. Sammutat valon, kävelet ==jonkin matkaa== alamäen suuntaan, ja toistat prosessin.

![Flashlight lit in a nightly forest](../images/630_gradient_descent_flashlight_forest_dalle.jpg)

**Kuvio 5:** *Kuva henkilöstä navigoimassa pimeässä metsässä taskulampun avulla. DALL-E 3:n näkemys.*

Kun laitat valot päälle, saat tietoa rinteen paikallisesta kallistuksesta. Gradientti osoittaa jyrkimmän ylämäen suuntaan, joten Gradient Descent kulkee vastakkaiseen suuntaan eli negatiivisen gradientin suuntaan. Kenties rinne viettää (lat, lon) -suunnassa `1.195` ja `12.01` -asteen kulmassa. Kun alat lähestyä alamäen syvintä kohtaa, mäki oletettavasti loiventuu. Mutta kuinka kaukana olet syvimmästä kohdasta? 100 metriä? 600 kilometriä? Tätä et voi tietää, joten sinun pitää yksinkertaisesti arvata, kuinka pitkän matkan kävelet taskulampun käytön jälkeen. ==Tämä arvaus on oppimisnopeus== (engl. learning rate).

!!! question

    Jos oppimisnopeus on kävelymatka, niin mikä tässä kuvitteellisessa esimerkissä olisi hyvä vertauskuva virheelle? Kenties kaukana näkyvien puunlatvojen korkeus suhteessa sinuun? Tai ehkäpä metsän pohjalla olevan kakun hajun voimakkuus?

Lopulta koulutus on yksinkertaista: laske gradientti, kerro se oppimisnopeudella, ja päivitä painot. Toista kunnes pysäytyskriteeri täyttyy. [^fromscratch]

```python title="IPython"
# This is what we should choose.
learning_rate = 0.01

for _ in range(n_epochs):    
        gradient = compute_gradient(X, y, w, m)

        # Compute the step : a vector containing step sizes per each weight
        step = learning_rate * gradient

        # Change the weights based on the step size
        w -= step

        # Here you would implement a stopping criterion
```

Alla 2000 epookin yli koulutettu gradient descent animaationa. Data on yllä käytettyä feikkidataa. Jos valitset liian suuren oppimisnopeuden, voit ylittää kakun ja joutua takaisin metsän reunaan. Jos valitset liian pienen oppimisnopeuden, taskulampustasi loppuu paristot ennen kuin löydät kakun.

![Animation of gradient descent](../images/630_gradient_descent_lines_over_epochs.gif)

**Kuvio 6:** *Animaatio Gradient Descent -algoritmin toiminnasta. Valittu learning rate on `0.01`. Virhe on esitetty SSE:nä, jotta se ei lähestyisi yhtä nopeasti nollaa kuin MSE.*

![Training error over epochs](../images/630_gradient_descent_training_plot_2000_epochs.png)

**Kuvio 7:** *Kuvaaja, jossa on esitetty virheen muutos koulutuksen aikana 2000 epookin yli. Virhe on esitetty SSE:nä.*

![Contour Plot](../images/630_gradient_descent_contour_plot.png)

**Kuvio 8:** *Kuvaaja, jossa on esitetty virhefunktion contour plot, minkä viivat muistuttaa mukavasti kartan korkeuskäyriä. Kuva on havainnollistava eikä vastaa numeerisesti tämän esimerkin arvoja.*

## Lisää opiskeltavaa

Tämä materiaali on vain pintaraapaisu Gradient Descent -algoritmiin. Tutustu myös seuraaviin aiheisiin:

* **Dynaamisen oppimisnopeuden käyttö.** Oppimisnopeutta voidaan muuttaa koulutuksen aikana. Voit käyttää esimerkiksi aluksi suurempaa oppimisnopeutta ja pienentää sitä koulutuksen edetessä.
* **L1 tai L2 regularisaatio.** Regularisaatio on tekniikka, joka auttaa estämään ylifittauksen. L1 ja L2 ovat kaksi yleisintä regularisaatiotyyppiä. Esimerkiksi L2 regularisaatio on yksinkertaisimmillaan sitä, että virhefunktioon lisätään `cost = cost + reg_rate * np.sum(w**2)`, jossa `reg_rate` on regularisaatiokertoimen arvo. Lisäät siis virheeseen painojen neliöiden summan eli rankaiset mallia siitä, jos painojen itseisarvo kasvaa suureksi.
* **Stochastic Gradient Descent.** Tässä esimerkissä käytettiin koko datasettiä gradientin laskemiseen. Tämä ei ole realistista, jos dataa on aivan valtavia määriä. Stokastinen Gradient Descent käyttää yhtä havaintoa kerrallaan.
* **Mini-Batch Gradient Descent.** Tämä on kompromissi kahden edellisen välillä. Et käytä kaikki tai vain yhtä havaintoa, vaan esimerkiksi 64 satunnaisesti valittua havaintoa kerrallaan. [^dl4cv]

## Tehtävät

!!! question

    Tutustu `630_gradient_descent_from_scratch.py`-Notebookiin. Notebookissa on toteuttu muutoin tätä materiaalia täsmäävä algoritmi, mutta kyseessä on ==stochastic== gradient descent, joka käyttää yhtä havaintoa kerrallaan. Tutustu algoritmin toteutukseen ja erityisesti siihen, kuinka valittu oppimisnopeus (learning rate) vaikuttaa koulutukseen.

    Kun olet sinut algoritmin kanssa, toteuta itse tehtävä: muokkaa algoritmia siten, että se toteuttaa ==Mini-Batch== Gradient Descentin. Voit joko tehdä Notebookista kopion tai toteuttaa kummakin samaan Notebookiin. Jos haluat toteuttaa ne samaan, niin ohjeellinen TODO-lista voisi näyttää tältä:

    * Toteuta *batch size* -liukuvalitsin: `batch_slider = mo.ui.slider(...)`
    * Tee funktion `sgd()` rinnalle toinen funktio, `minibatch_gd()`, joka toteuttaa Mini-Batch Gradient Descentin.
    * Tee ennusteet kummallakin, eli:
        * `result_sgd = sgd(...)`
        * `result_mb = minibatch_gd(..., batch_size=batch_slider.value)`
    * Lisää kummatkin viivat oppimiskäyrien visualisointiin.

    Mielenkiintoinen tutkittava asia on esimerkiksi se, kuinka eri batch-koot vaikuttavat koulutuksen nopeuteen ja virheen laskuun. Vaatiiko SGD vai mini-BGD suuremman oppimisnopeuden?

## Lähteet

[^dl4cv]: Rosebrock, A. *Deep Learning for Computer Vision with Python. Starter Bundle. 3rd Edition*. PyImageSearch. 2019.
[^ldl]: Ekman, M. *Learning Deep Learning: Theory and Practice of Neural Networks, Computer Vision, NLP, and Transformers using TensorFlow*. Addison-Wesley. 2025.
[^surrogate]: Kim, N.H. *Surrogate Modeling and Optimization*. Wiley. 2026.
[^fromscratch]: Grus, J. *Data Science from Scratch 2nd Edition*. O'Reilly Media. 2019.
[^essential-math-for-ds]: Nield, T. *Essential Math for Data Science*. O'Reilly. 2022.
