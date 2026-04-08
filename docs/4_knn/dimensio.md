---
priority: 410
---

# Dimensiovähennys

Dimensiovähennys on prosessi, jossa esimerkiki 200 eri sarakkeesta koostuvasta datasta luodaan uusi datasetti, jossa on vain 2, 5 tai 10 saraketta. Ajoittain dimensiot voi vähentää hyvin luontevalla tavalla, kuten alla olevassa datasetissä:

| $x_1$ | $x_2$ | $x_3$ | $z$  |
| ----- | ----- | ----- | ---- |
| 10    | 5     | 2     | 100  |
| 20    | 10    | 4     | 800  |
| 30    | 15    | 6     | 2700 |

Muuttuja $z$ on laskettu kaavalla $z = x_1 \cdot x_2 \cdot x_3$. Näin olemme pudottaneet dimensionalisuuden 3 → 1. Tämä operaatio tekee hetkessä järkeä, kun paljastan, että $x_1$, $x_2$ ja $x_3$ ovat huoneen pituus, leveys ja korkeus metreinä. Täten $z$ on huoneen tilavuus kuutiometreinä. Tulet huomaamaan, että syntyvät uudet dimentiot, eivät ole aina yhtä intuitiivisia kuin tämä yksinkertainen esimerkki.

!!! tip

    PCA on ohjaamaton menetelmä. Se tarkoittaa, että se ei käytä hyväkseen datan labeleita tai luokkia, vaan ainoastaan datan ominaisuuksia (features).

!!! warning

    Yllä oleva esimerkki on **käsin tehtyä** piirremuotoilua. PCA voisi löytää pikemminkin sellaisen kombinaation, joka täyttää kaavan:

    $$
    z = a_1 x_1 + a_2 + x_2 + a_3 + x_3
    $$

## Miksi dimensioita vähennetään?

Kaksi päämotiivia dimensioiden vähemtämiselle ovat **visualisointi** ja **yksinkertaisempien mallien rakentaminen**. Jälkimmäiseen liittyy termi *curse of dimensionality*. Käsitellään nämä alla.

### Visualisointi

Ihmisen on hyvin vaikea tulkita yli kolmiulotteisia avaruuksia, joten monilotteiden datan visualisointi on haastavaa. Tähän palataan Syväoppiminen I -kurssilla kielimallien embeddings-vektorien yhteydessä. Tämän kurssin puittteissa riittää, että tutustut asiaan pintapuoleisesti. Tästä voi olla sinulle hyötyä esimerkiksi projektityössä, jossa haluat visualisoida datasi.

Esimerkiksi voit tutustua Plotly-kirjaston [PCA Visualization in Python](https://plotly.com/python/pca-visualization/)-dokumenttiin. Kyseisessä esimerkissä käytetään kurjenmiekkojen (engl. iris) datasettiä, jossa on neljä muuttujaa: verholehden (engl. sepal) pituus ja leveys, sekä terälehden (engl. petal) pituus ja leveys. Näillä on hyvinkin intuitiivinen yhteys, aivan kuten huoneen pituudella, leveydellä ja korkeudella.

### Yksinkertaisemman mallin rakentaminen

Ensemblet ja neuroverkot ovat esimerkkejä malleista, jotka kykenevät käsittelemään suurta määrää muuttujia. Kuitenkin, mitä enemmän muuttujia datassa on, sitä vaikeampaa mallin on löytää niistä merkityksellisiä kuvioita. [^hundredpage]

> "In bioinformatics, for example, the potential dimension of a dataset can be enormous. Researchers may have thousands of gene expressions for each observation, many of which are highly correlated (and thus possibly redundant) with each other."
>
> — Alex J. Gutman & Jordan Goldmeier [^data-head]

Riippuu siis mallista, ovatko päällekäiset tai turhat piirteet ongelma, tai onko niiden suuri määrä ongelma. Termi *curse of dimensionality* viittaa siihen, että datan dimensioiden kasvaessa, datan tiheys avaruudessa pienenee eksponentiaalisesti. Tämä ajaa tilanteeseen, jossa lähes kaikki dataparit ovat `avg(distance)`-etäisyydellä toisistaan [^dl-java]. Tämä rampauttaa erityisesti k-NN:n ja muut etäisyyspohjaiset algoritmit, jotka luottavat siihen, että lähellä olevat datapisteet ovat merkityksellisiä. Ratkaisu tähän on piirteiden poistaminen tai yhdistäminen – mutta tämä on helpommin sanottu kuin tehty.

!!! tip "Bonus: pakkaus"

    Bonuksena mainittakoon kolmas mielenkiintoinen PCA:lle: kuvan pakkaus. PCA:n avulla voidaan luoda uusi datasetti, jossa on vähemmän sarakkeita kuin alkuperäisessä datasetissä, mutta joka säilyttää mahdollisimman paljon informaatiota alkuperäisestä datasta.

    Tähän voit tutustua Kaggle [Introduction to PCA: Image Compression example](https://www.kaggle.com/code/mirzarahim/introduction-to-pca-image-compression-example) -Notebookin avulla, jonka on luonut Mirza Rahim. Esimerkissä pakataan `(768, 1040)`-muotoinen harmaasävykuva `(768, k)`-kokoiseksi dataksi. Kukin kuvan rivi pakkautuu siis murto-osaan alkuperäisestä koostaan. Millainen mössö kuvasta tulee, jos k on 10? Entä onko 250 jo täysin alkuperäisen tasoinen ihmissilmin nähtynä?

## PCA

Burkovin mukaan [^hundredpage] dimensiovähennyksen eniten käytetyt tekniikat ovat:

* PCA (Principal Component Analysis)
* UMAP (Uniform Manifold Approximation and Projection)
* Autoencoderit

### Mikä?

PCA on yksi vanhimmista ja suosituimmista dimensionaalisuuden vähennysmenetelmistä, joten keskitymme tässä luvussa pääasiassa siihen. PCA on vuodelta 1901, jolloin Karl Pearson julkaisi artikkelin "On Lines and Planes of Closest Fit to Systems of Points in Space". [^pearson-k] Scikit-learn-kirjastosta ei löydy UMAP:ia, joten kurssin Notebookissa on edustettuna sen korvaajana samaan manifold learning -perheeseen kuuluvat MDS (Multidimensional Scaling) ja t-SNE (t-distributed Stochastic Neighbor Embedding). Autoencoderit kuuluvat Syväoppiminen I -kurssin aihepiiriin, joten niihin ei tässä yhteydessä syvennytä.

PCA:lla on juuret vahvasti lineaarialgebrassa. Olet todennäköisesti törmännyt termeihin ominaisvektori (engl. eigenvector) ja ominaisarvo (engl. eigenvalue) jossain vaiheessa matematiikan opintojasi, ja PCA on nimenomaan näihin käsitteisiin perustuva algoritmi.

> "The machine learning technique called principal component analysis (PCA) corresponds to applying SVD to a data matrix. Alternatively, you can think of the PCA as applying an eigendecomposition to the covariance matrix of the data."
>
> — Ivan Savov [^no-bs-linalg]

### Ominaisuudet

PCA ei valitse alkuperäisistä muuttujista osajoukkoa, vaan muodostaa uusia muuttujia eli pääkomponentteja. Jokainen pääkomponentti on alkuperäisten muuttujien painotettu summa eli siis uudet komponentit ovat alkuperäisten muuttujien lineaarikombinaatioita. Käytännössä PCA siis pyörittää (engl. rotate) dataa siten, että uusi koordinaatisto on suunnattu datan suurimman varianssin suuntaan. Tätä voi verrata valokuvaukseen: mistä suunnasta sinun tulee kuvata Gazan pyramidia, jotta tämä 3D-maailman esine säilyttää mahdollisimman paljon informaatiota 2D-esitysmuodossaan? [^data-head]

!!! warning

    PCA on herkkä muuttujien mittakaavalle. Muuttujat yleensä keskitetään (keskiarvo vähennetään) ja usein myös standardisoidaan (Z-score) ennen PCA:ta, jotta yksittäinen suurella skaalalla oleva muuttuja ei dominoi komponentteja.

Jos päädyt kokeilemaan PCA:ta käytännössä, kannattaa tutustua `pca.explained_variance_ratio_`-attribuuttiin, joka kertoo, kuinka paljon informaatiota (varianssia) säilyy, kun data pakataan tiettyyn määrään komponentteja. Esimerkiksi, jos palautuva ratio on `[0.5, 0.3]`, se tarkoittaa, että ensimmäinen komponentti säilyttää 50% datan varianssista ja toinen komponentti säilyttää 30% datan varianssista. Näin ollen, jos valitset vain nämä kaksi komponenttia, säilytät yhteensä 80% datan varianssista.

!!! note

    PCA ei itsessään vähennä dimensioita, vaan tarjoaa uuden koordinaatiston. Dimensiovähennys syntyy vasta, kun päätämme säilyttää vain osan pääkomponenteista.

### Haasteet

PCA toimii parhaiten silloin, kun datassa on lineaarista rakennetta, sillä se perustuu kovarianssiin ja korrelaatioon. Menetelmät kuten MDS ja t-SNE kykenevät käsittelemään myös ei-lineaarista dataa, mutta niiden käyttöön liittyy omia haasteitaan, kuten parametrien säätöä.

PCA:n pääkomponentit eivät ole aina helposti tulkittavia. Oppikirjaesimerkeissä niille voi löytyä intuitiivisia nimiä, mutta todellisessa datassa tällaiset tulkinnat ovat usein epävarmoja. PCA ei valitse alkuperäisistä muuttujista “parhaita” eikä poista niitä automaattisesti. Se muodostaa kaikista muuttujista uusia lineaarikombinaatioita, ja varsinainen dimensiovähennys tapahtuu vasta, kun analyytikko päättää, montako pääkomponenttia säilytetään. Tähän valintaan ei ole yhtä oikeaa sääntöä. [^data-head]

Lisäksi PCA olettaa, että suuri varianssi tarkoittaa jotakin tärkeää. Näin ei kuitenkaan aina ole: muuttuja voi vaihdella paljon mutta olla ilmiön kannalta epäolennainen. Tällöin PCA voi korostaa vaihtelua, joka ei auta ymmärtämään itse ongelmaa. Siksi PCA:ta ei pidä käyttää sokeasti, vaan sen hyödyllisyyttä kannattaa arvioida myös mallin suorituskykymittareiden ja oman domain-ymmärryksen avulla. [^data-head]

## Tehtävät

!!! question "Tehtävä: Dimensiovähennyksen leikkikenttä"

    Avaa `410_dimensionality_reduction.py` ja tutustu siihen. Notebookissa voit piirtää itse 2-ulotteisen datasetin Drawdata-widgetillä. Tarkoitus on rakentaa intuitiota siitä, miten PCA, MDS ja t-SNE muuttavat sen 2- tai 1-ulotteiseksi esitykseksi.

    - Kaksi viivaa, jotka leikkaavat toisensa (X-muoto)
    - Kaksi viivaa, jotka eivät leikkaa toisiaan (V tai II muoto)
    - S-kirjain
    - Windows-logon kaltainen neljän klusterin muoto.

    Huomaa, että tässä notebookissa mikään menetelmistä ei käytä pisteiden labeleita tai värejä. PCA, MDS ja t-SNE saavat syötteekseen vain pisteiden (x, y)-koordinaatit. Värit auttavat vain tulkitsemaan tulosta visuaalisesti: niiden avulla on helpompi nähdä, pysyvätkö datan eri osat erillään vai menevätkö ne sekaisin. Lopussa on lisäksi MNIST-datasetistä vastaava muunnos.


## Lähteet

[^no-bs-linalg]: Savov, I. *No bullshit guide to linear algebra (v2.1)*. Minireference. 2017.
[^data-head]: Gutman, A. & Goldmeier, J. *Becoming a Data Head". Wiley. 2021.
[^hundredpage]: Burkov, A. *The Hundred-Page Machine Learning Book*. Self-published, 2019.
[^pearson-k]: Pearson, K. *On Lines and Planes of Closest Fit to Systems of Points in Space.* Philosophical Magazine. 1901. https://doi.org/10.1080/14786440109462720
[^dl-java]: Sugomori, Y. et. al. *Deep Learning: Practical Neural Networks with Java*. Packt Publishing. 2017.
