!!! info

    Tämä osio, "Data", on tarkoitettu käytettäväksi yhdessä muiden osioiden kanssa. Lue aiheet kerran läpi nyt, mutta palaa aiheisiin muiden osioiden yhteydessä.

Koneoppimisen yhteydessä data on usein raakaa ja vaatii esikäsittelyä ennen kuin se voidaan syöttää koneoppimismalliin. Esikäsittely voi sisältää datan puhdistamista, skaalaamista, muuttujien valintaa ja muuta. Termillä "datasetti" tarkoitetaan jotakin kokoelmaa dataa. Data ei välttämättä ole esimerkiksi taulukkomuodossa - mutta usein se joko on, tai se muokataan taulukkomuotoon ennen koneoppimismallin soveltamista.

## Järjestyneisyys

### Strukturoitu data

Aloitetaan strukturoidusta datasta, koska sitä on helppo käsitellä ja ymmärtää. Strukturoitu data on dataa, joka on järjestetty taulukkomuotoon. Taulukko koostuu riveistä ja sarakkeista. Tyypillisesti strukturoitu data on tallennettu tietokantaan (esim. PostgreSQL, MySQL, SQLite) tai tietovarastoon (esim. Snowflake, Redshift). Se voi olla myös taulukkolaskentaohjelman taulukossa (esim. Excel, Google Sheets), mutta tällöin voitaisiin väitellä siitä, tippuuko se osittain strukturoidun datan kategoriaan (ks. alla). Tietokannat ovat tyypillisesti vahvasti tyyppisiä, kun taas taulukkolaskentaohjelmat ovat usein heikosti tyyppisiä. Esimerkiksi tietokannassa jokaisella sarakkeella on tyyppi (esim. `INTEGER`, `VARCHAR`, `DATE`), kun taas taulukkolaskentaohjelmassa sarakkeen tyyppi voi muuttua dynaamisesti. Mikäli kentän "Pituus" arvo voi olla merkkijono `182 cm`, tai kvalitatiivinen arvo `pitkä`, data ei sisällä varsinaista lukkoon lyötyä struktuuria.

| Selittävä a | Selittävä b | Selittävä c | Selitettävä muuttuja [y] |
| ----------- | ----------- | ----------- | ------------------------ |
| 1.01        | 512         | R           | 1                        |
| 1.10        | 124124      | Python      | 0                        |
| 0.99        | 42          | R           | 1                        |
| 0.97        | N/A         | R           | 1                        |
| 1.00        | 96141       | Python      | 0                        |
    

#### Havainnot

Havainnot ovat **rivejä** taulukossa. Huomaa, että kukin rivi vastaa yhtä havaintoa. Havainnot ovat datasetin yksittäisiä tapauksia. Yksi rivi voi edustaa esimerkiksi yhtä henkilöä, yhtä autoa, yhtä ajomatkaa, myyntitapahtumaa tai mitä tahansa muuta. Saman datasetin rivit edustavat kuitenkin samaa tyyppiä. Yllä olevan taulukon ensimmäinen rivi olisi siis:

```python
row_x0: tuple = (
    1.01, 512, 'R', 1
)
```

#### Muuttujat

Muuttujat ovat datasetin **sarakkeita**, toiselta nimeltään **pystyrivejä**, jotka ==kuvaavat havaintoja==. Ne ovat siis havaintojen piirteitä tai dimensioita. Strukturoidun datan tapauksessa muuttujalla on jokin tyyppi, kuten `INTEGER`, `VARCHAR` tai `DATE`. Tyypin lisäksi arvo voi useimmiten olla myös `NULL` tai `N/A`, mikä tarkoittaa, että arvoa ei ole saatavilla.

* numeerisia
    * Pituus
    * Paino
    * Huoneiden määrä
* kategorisia
    * Matkustajaluokka
    * Kotimaa

Data jaetaan usein **selittävien** ja **selitettävän** mukaan kahteen matriisiksi `X` ja vektoriksi `y`. `X` sisältää selittävät muuttujat ja `y` selitettävän muuttujan. Tämä on tärkeää, koska koneoppimismallit vaativat tietynlaista syötettä. Esimerkiksi lineaarinen regressio vaatii `X`-matriisin ja `y`-vektorin.

```python title="IPython"
X: list[tuple] = [
    (1.01, 512, 'R'),
    (1.10, 124124, 'Python'),
    (0.99, 42, 'R'),
    (0.97, None, 'R'),
    (1.00, 96141, 'Python')
]

y: list[int] = [1, 0, 1, 1, 0]
```

On tärkeää huomata, että on ==ihmisen päätös==, että mikä muuttuja on selittävä ja mitkä ovat selittäviä. Kukaan ei estä meitä ottamasta yllä näkyvää datasettiä ja kouluttamasta tekoälyä, joka pyrkii kolmen muun sarakkeen arvon perusteella ennustamaan arvon `Python` tai `R`.

On yllättävänkin tyypillistä, että jokin data puuttuu. Paras tapa korjata `NULL`-ongelma on tehostaa datan keräysprosessia. Emme elä täydellisessä maailmassa, joten valitettavasti meidän on kuitenkin opittava käsittelemään puuttuvaa dataa. Puuttuvaa dataa voidaan käsitellä monin eri tavoin, kuten poistamalla vialliset havainnot, täyttämällä puuttuvat arvot keskiarvolla tai mediaanilla. Puuttuvan datan paikkauksessa tai poistamisessa pitää olla kuitenkin tarkkana: arvon puuttuminen voi olla merkityksellistä. Eli siis se, että arvo puuttuu, voi olla itsessään tietoa, joka auttaa ennustamaan selitettävää muuttujaa.

!!! question "Tehtävä"

    Kuvittele, että työskentelet yritykselle, joka valmistaa e-lukulaitteita. Vikatilanteessa ohjelma lähettää virheraportin valmistajan palvelimille (mikäli käyttäjä on antanut tähän suostumuksen). Tutkit näiden vikatilanteiden *trace dataa*. Kaikki muut kentät sisältävät aina jotakin tietoa, mutta kenttä `MOTHERBOARD_FIRMWARE` on ajoittain `NULL`. Lasket, että näitä rivejä on datasetissä 10 %. Kuinka alkaisit selvittää, ihan maalaisjärkeä käyttäen, voiko `MOTHERBOARD_FIRMWARE`-kentän puuttuminen olla merkityksellistä?


### Osittain strukturoitu data

Osittain strukturoitu data (engl. semi-structured data) on dataa, joka ei ole täysin taulukkomuotoista, mutta sillä on kuitenkin selkeä koneluettava skeema. Esimerkiksi HTML-tiedostot ovat osittain strukturoitunutta dataa. Alla on esimerkki hyvin, hyvin yksinkertaisesta HTML-sivusta:

```html title="index.html"
<!DOCTYPE html>
<html>
    <head>
        <title>Pizza e-Bible</title>
    </head>
    <h1>Pizza Ingredients</h1>
    <ul>
        <li>Egg</li>
        <li>Ham</li>
        <li>Spam</li>
    </ul>
</html>
```

Yllä näkyvän kaltaista dataa voi louhia koneellisesti esimerkiksi Wikipediasta. Osittain strukturoitu data vaatii huomattavasti enemmän käsittelyä kuin strukturoitu data. Tämä käsittely on luonnollisesti virheherkkä prosessi. Datan skeema voi ajan kanssa elää. Esimerkiksi jos Wikipedian sivun HTML-rakenne muuttuu, koneellinen louhinta voi epäonnistua, tai voit päätyä kirjoittamaan väärää tietoa tietokantaan.

!!! info

    Tiedostopääte tai tiedoston formaatti ei itsessään tee datasta kokonaan tai osittain strukturoitua. Esimerkiksi JSON-tiedosto voi olla täysin strukturoitua dataa, jos se on koneellisesti kirjoitettu skeemaa noudattaen.


### Epästrukturoitu data

Data, joka on täysin vailla struktuuria (engl. unstructured data), on kaikkein haastavinta käsitellä. Sanat *täysin vailla struktuuria* eivät suinkaan tarkoita, että kyseessä olisi täysin kaoottinen data. Yksittäinen havainto voi olla esimerkiksi äänitiedosto, kuva, video tai täysin vapaamuotoinen teksti. Tiedostolla itsellään, kuten `.JPG`-tiedostolla, on oma rakenne, mutta *informaatio* tiedostossa on epästrukturoitua. Esimerkiksi kuvan pikseli xy-koordinaatissa `(100, 100)` ei itsessään kerro mitään siitä, onko kyseessä kissa vai koira. Kuvan sisältöä tulee siis *kuvailla* jollakin muulla tavalla numeraalisesti.

Mikäli koulutat tekoälyä, joka pyrkii tunnistamaan onko kuvassa kissa vai koira, sinun dataset voi vaikkapa hakemistorakenne:

```
dataset/
    cat/
        DSC_0123.jpg
        web_fluffy.jpg
        ...
    dog/
        IMG_0473.jpg
        P129719088.jpg
        ...
```

Se, kuuluuko kuva hakemistoon `cat/` vai `dog/`, on datasetin selitettävä muuttuja. Kuvan pikseleistä ==muodostetaan myöhemmin== selittäviä muuttujia. Tähän soveltuvat eri feature descriptorit eli piirrekuvaajat, kuten HOG, SIFT, SURF, ORB, LBP, jne.

Huomaa, että saman datan voi tallentaa tietokantaan, mutta se ei tee datasta sen enempää strukturoitua. Alla esimerkki, jossa kullakin kuvalla on `id`, `label` ja `data`, joista jälkimmäisin on binäärimuodossa tallennettu kuva.

| id  | label | data                |
| --- | ----- | ------------------- |
| 1   | cat   | 0x111001010...00111 |
| 2   | dog   | 0x001001010...11010 |
| ... | ...   | ...                 |


## Vektorit ja matriisit

Yllä datasetti on kuvailtu ihmiselle helpossa taulukkomuodossa. Koneoppimismallien tapauksessa data käsitellään yleensä matriiseina ja vektoreina. *Vektorisoidut operaatiot* ovat tehokkaampia kuin silmukat. Tyypillisesti koneoppimisessa käytetään `numpy`-kirjastoa edustamaan matriisia `X` ja vektoria `y`. Matriisi `X` on kaksiulotteinen taulukko, jossa rivit ovat havaintoja ja sarakkeet ovat muuttujia. Vektori `y` on yksiulotteinen taulukko, jossa on selitettävän muuttujan arvot. Alla on ylempää dokumentista tuttu datasetti `X` ja `y` `numpy`-muodossa.

```python title="IPython"
import numpy as np

X: np.ndarray = np.array([
    [1.01, 512, 0],
    [1.10, 124124, 1],
    [0.99, 42, 0],
    [0.97, 0, 0],
    [1.00, 96141, 1]
])

y: np.ndarray = np.array([1, 0, 1, 1, 0])
```

Tällä kurssilla suositaan Pythonin listoja, tupleja ja sanakirjoja, koska kurssilla pyritään ymmärtämään, kuinka algoritmit toimivat. On hyvä olla kuitenkin tietoinen, että usein koneoppimiseen liittyvissä matemaattisissa kaavoissa `x` ja `y` ovat vektoreita ja matriiseja ja laskuoperaatiot suoritetaan vektorisoituina operaatioina. 

### Matemaattinen kaava

Alla näkyvässä kaavassa `X` on matriisi, `w` on vektori ja `y_hat` on vektori. Se, mitä tämä kaava varsinaisesti tekee esitellään myöhemmin. Keskitytään tässä yhteydessä syntaksin ymmärtämiseen.

$$
\hat{y} = Xw
$$

Saman voisi kirjoittaa auki alla näkyvällä tavalla, olettaen että X on muodoltaan $3 \times 2$ ja w on pituudeltaan $2$. Vektorin $w$ voi myös kuvitella $2 \times 1$ matriisiksi. Koot voivat olla mitä tahansa, kunhan matriisin `(m, n)` ja vektorin `(n)` koot täsmäävät `n`:n eli sarakkeiden lukumäärän osalta.

$$
\begin{bmatrix}
\hat{y}_{1} \\
\hat{y}_{2} \\
\hat{y}_{3} \\
\end{bmatrix}
=
\begin{bmatrix}
x_{1,1} & x_{1,2} \\
x_{2,1} & x_{2,2} \\
x_{3,1} & x_{3,2} \\
\end{bmatrix}
\begin{bmatrix}
w_{1} \\
w_{2} \\
\end{bmatrix}
$$

Matriisin ja vektorin tulo lasketaan siten, että otetaan kunkin rivi ja kerrotaan se vektorilla, eli otetaan näiden pistetulo. Esimerkiksi:

$$
\begin{bmatrix}
\hat{y}_{1} \\
\hat{y}_{2} \\
\hat{y}_{3} \\
\end{bmatrix}
=
\begin{bmatrix}
11 & 22 \\
21 & 22 \\
31 & 22 \\
\end{bmatrix}
\begin{bmatrix}
1 \\
2 \\
\end{bmatrix}
=
\begin{bmatrix}
11 \cdot 1 + 22 \cdot 2 \\
21 \cdot 1 + 22 \cdot 2 \\
31 \cdot 1 + 22 \cdot 2 \\
\end{bmatrix}
$$

!!! question "Tehtävä"

    Kahden matriisin välinen välinen pistetulo lasketaan hyvin samalla tavalla. Selvitä, kuinka tämä toimii.


Tehdään tämä vielä Pythonissa. Mikäli haluamme luoda matriisin `X` ja vektorin `y` käyttämättä mitään kirjastoja tai luokkia, voimme kuvastaan niitä seuraavalla tavalla:

```python title="IPython"
X = [
    (11, 22),
    (21, 22),
    (31, 22),
]

w = (1, 2)
```

Tulon voisi laskea silmukkaa käyttäen seuraavalla tavalla:

```python title="IPython"
def dot(u, v):
    return sum(u_i * v_i for u_i, v_i in zip(u, v))

y_hat = [dot(x, w) for x in X]
```

Numpyä käyttäen mikään ei muutu, paitsi että `X` ja `y` ovat `numpy`-matriisi ja -vektori, jolloin voimme hyödyntää numpy-kirjaston metodeja.

```python title="IPython"
import numpy as np

X = np.array(
[
    (11, 22),
    (21, 22),
    (31, 22),
]
)

w = np.array([1, 2])

# Option 1: Using the dot method
y_hat = X.dot(w)

# Option 2: Using the matmul operator
y_hat = X @ w
```

!!! warning

    Huomaa, että `X * w` on eri asia kuin `X @ w`. Ensimmäinen näistä on `element-wise multiplication`, jossa kunkin alkion kohdalla kerrotaan vastaava alkio. Jälkimmäinen on matriisitulo, jossa otetaan kunkin rivin pistetulo vektorin kanssa. LateX-kaavoissa `u * v` olisi $u \odot v$ ja `X @ w` olisi $Xw$.

## Muuttujien enkoodaus

Yllä esiteltiin, että koneoppimisessa operaatiot suoritetaan usein vektorisoidussa muodossa. Tämä tarkoittaa, että kaikki muuttujat tulee olla numeerisia. Kategoriset muuttujat, kuten `Python` ja `R`, tulee muuntaa numeeriseen muotoon. Tämä prosessi tunnetaan enkoodauksena. Tämän voi tehdä monella eri tavalla, mutta kaksi yleistä tapaa ovat:

* **Label encoding:** Kategoriset muuttujat muutetaan numeeriseen muotoon. Esimerkiksi `[Cat, Dog, Dog, Hamster]` => `[0, 1, 1, 2]`.
* **One-hot encoding:**  Kategiruset muuttujat muutetaan useiksi binäärimuuttujiksi. Esimerkiksi `[Cat, Dog, Dog, Hamster]` => `[[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]]`.

Huomaa, että label encoding tekee jostakin luokasta numerona suuremman kuin toisen. Tämä voi johtaa virheellisiin johtopäätöksiin, koska koneoppimismallit voivat tulkita suuruusjärjestyksen olevan merkityksellinen. Kategoriset muuttujat, kuten t-paidan koko (S, M, L, XL jne.) ovat järjestyksellisiä eli ordinaalisia. Tällöin niiden vaihtaminen numeroiksi voi olla perusteltua. Parempi ratkaisu olisi kuitenkin käyttää esimerkiksi rinnan ympärysmittaa tai muuta oikeaa mittaustulosta. Kaikki kategoriset muuttujat eivät kuitenkaan ole ordinaalisia. Jos datasetissä on esimerkiksi piirre lempiväri, joka on kokonaislukuina enkoodattuna: `0: oranssi`ja `6: Sininen`, koneoppimismalli voi tulkita, että sininen on kuusi kertaa niin suuri kuin oranssi.

!!! question

    Mitä ongelmia koituisi, jos kääntäisit kategorisen lempivärin numeraaliseksi käyttäen värin hue-arvoa? Hue on numeerinen arvo välillä 0-360, joka kuvaa värin sävyä.

    Tämän pohtiminen voi olla helpompaa visuaalisesti. Käytä esimerkiksi [W3Schools: HTML HSL and HSLA Colors](https://www.w3schools.com/html/html_colors_hsl.asp)-interaktiivista työkalua ja mieti, mitä värejä edustavat todella pienet ja suuret hue-arvot.

Jos käytämme meidän alkuperäistä datasettiä, jossa kolmas sarake on kategorinen muuttuja, voimme enkoodata sen binääriseksi One-hot encoding -menetelmällä seuraavasti:

| Selittävä a | Selittävä b | is_python | Selitettävä muuttuja [y] |
| ----------- | ----------- | --------- | ------------------------ |
| 1.01        | 512         | 0         | 1                        |
| 1.10        | 124124      | 1         | 0                        |
| 0.99        | 42          | 0         | 1                        |
| 0.97        | N/A         | 0         | 1                        |
| 1.00        | 96141       | 1         | 0                        |

!!! warning

    Huomaa, että emme luo erikseen kahta eri saraketta: `is_r` ja `is_python`. Syy tälle on multikollineaarisuuden välttely. Emme halua sarakkeita, jotka ovat täysin korreloituneita keskenään, koska se voi aiheuttaa ongelmia mallin sovittamisessa.

## Vektorit ilman Numpyä

Koneoppimisessa käytetään usein `numpy`-kirjastoa. On hyvä ymmärtää, että vektori on käytännössä vain tuple tai lista, joka reagoi matemaattisiin operaatioihin määrätyllä tavalla: esimerkiksi kahden vektorin summa on `element-by-element sum`. Vastaavasti matriisia voidaan käsitellä listana vektoreita varsinkin siinä tapauksessa, että matriisi edustaa datasettiä.

Helppo tapa tutustua vektoreiden toimintaa on toteuttaa vektori- ja matriisiluokat itse. Suorituskyvyssä natiivi Python-toteutus häviää merkittävästi C-kieleen nojaavalle `numpy`-kirjastolle, mutta ymmärrys vektorien ja matriisien toiminnasta paranee. Alla on esimerkki vektoriluokasta, joka tukee pistetuloa ja elementtikohtaista kertolaskua.

Voit myös yrittää luoda oman matriisiluokan, joka tukee matriisituloa ja elementtikohtaista kertolaskua. Jos matriisiluokan tarkoitus on olla säilö datasetille, voit myös lisätä siihen metodeja, jotka tekevät datasetin käsittelystä helpompaa: eli lisätä luokkaan sitä, mitä Pandas tekee. Luokka voi esimerkiksi ylläpitää sisältämiensä vektorien nimet, tietotyypit ja tilastoja. Luokka voi myös tarjota metodeja datasetin jakamiseen koulutus- ja testidataan, datan sekoittamiseen ja datan esikäsittelyyn.

```python title="IPython"
class Vector:
    def __init__(self, *args: int|float):
        self.elements = list(args)

    def __iter__(self):
        return iter(self.elements)

    def __mul__(self, other):
        return Vector(*[a * b for a, b in zip(self, other)])
    
    def __matmul__(self, other):
        return sum(self * other)
    
    def __eq__(self, other):
        return [a == b for a, b in zip(self, other)]
    
u = Vector(1, 2, 3)
v = Vector(4, 5, 6)

u * v == Vector(4, 10, 18)
u @ v == 32
```

!!! question "Tehtävä"

    Tutustu `ml/vector.py`-tiedostossa olevaan `Vector`-luokkaan. Kokeile importata se esimerkiksi Jupyter Notebookiin ja laskea vektoreiden sekä skaalarien välisiä operaatioita.
