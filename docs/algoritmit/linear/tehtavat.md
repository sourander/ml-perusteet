# 📝 Tehtävät

## Tehtävä: Auton hinta

Koulut Linear Regression. Käytä samaa autodataa aiemmissa tehtävissä, mutta tällä kertaa ennusta auton hintaa. Koska hinta on tyypillinen ennustettava muuttuja tässä datasetissä, löydät netistä referenssitoteutuksia. Ethän kuitenkaan kopioi koodia! Kirjoita itse oma koodi, minkä jokaisen rivin merkityksen ymmärrät.


### Vinkit

#### Pipeline

Mikäli olet aiemmin käyttänyt solu solulta etenevää vapaamuotoista koodia, kokeile tällä kertaa käyttää `sklearn`-kirjaston `Pipeline`- tai `ColumnTransformer`-luokkia. Tee kaikkesi, jotta koodi on helppolukuista ja helposti kehitettävissä. Ideaalitilanteessa esimerkiksi datan esikäsittely hoituu omalla pipelinellä, joka voi näyttää vaikkapa 

```python
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from feature_engine import selection as fs

class SlugifyColumns(BaseEstimator, TransformerMixin):
    pass

class Mapper(BaseEstimator, TransformerMixin):
    pass

class SplitGroupType(BaseEstimator, TransformerMixin):
    pass

class Whatchamacallit(BaseEstimator, TransformerMixin):
    pass

pipeline = Pipeline(steps=[
    ('slugify_column_names', SlugifyColumns()),
    ('likert_to_number', Mapper(variables=[TARGET], mappings=LIKERT_MAPPINGS)),
    ('split_group_type', SplitGroupType(variables=GROUP_TYPE_COL)),
    ('drop_columns', fs.DropFeatures(DROP_COLUMNS)),
    ('sex_normalize', Whatchamacallit(variables=["foo"])),
])

df_preprocessed = pipeline.fit_transform(df)
```

Huomaa, että koska kaikki transformaatiot ovat omia luokkiaan, voit helposti lisätä niitä lisää, ja voit helposti uusiokäyttää niitä eri pipelinen osissa. Tämä tekee koodista helposti laajennettavaa ja muokattavaa. Niitä voi myös testata. Alla yksinkertainen käsikutoinen testi:

```python
sgp_input = pd.DataFrame({
    'foo': [' AB123 4', ' 2 ', 'C3', 'D4AB'],
    'other_column': [1, 2, 3, 4]
})
sgp = SplitGroupType(variables=['foo'])
print(sgp.transform(sgp_input))
```

```cmd title="stdout"
   other_column   foo_letters   foo_numbers
0             1            AB          1234
1             2                           2
2             3             C             3
3             4           DAB             4
```

#### Muiden tehtävien parantelu

Tämä tehtävä on aiempia todennäköisesti helpompi, koska datasetti on sinulle jo entuudestaan tuttu, ja Linear Regression on äärimmäisen yksinkertainen algoritmi. Käytä tässä voitettua aikaa aiempien skriptien ja oppimispäiväkirjojen parantamiseen. Kenties huomaat oppineesi jotakin uutta Data-osioon liittyvistä videoista, jotka antavat sinulle uusia ideoita datan käsittelyyn tai mallin suorituskyvyn arviointiin? 

Kenties huomaat myös, että parin viikon tauon jälkeen luettuvuna sinun oma koodisi vaikuttaa vaikealta. Kenties voit kokeilla parantaa myös sitä Pipelinen avulla?

## Tehtävä: Kyberviha

Kouluta Logistic Regression -malli, joka ennustaa, onko viesti kybervihaa vai ei. Käytämme tätä datasettia: [Digital skills among youth: A dataset from a three-wave longitudinal survey in six European countries](https://www.sciencedirect.com/science/article/pii/S2352340924003652)

Sarakkeita on 862, joten olen opettajan roolissa pureskellut hieman dataa valmiiksi. Jupyter Notebook, jolla data on käsitelty, sekä käsitelty data, löytyvät kumpikin [gh:sourander/ml-perusteet-code](https://github.com/sourander/ml-perusteet-code)-repositoriosta. Polut ovat:

* Notebook: `src/playground/yskills_dataprep.ipynb`
* Data: `data/y_skills/ySKILLS_longitudinal_dataset_teacher_processed.csv`

Tavoitteena on ennustaa `RISK101`-kentän arvo muiden piirteiden avulla. `RISK101` on binäärinen muuttuja, joka kertoo, onko henkilöllä riski kohdata kybervihaa (engl. cyberhate). On äärimmäisen suositeltavaa ladata alkuperäinen datasetti ja sen kylkiäisenä toimitettava `ySKILLS_data_dictionary.xlsx`-tiedosto, joka kertoo, mitä kentät tarkoittavat. On myös äärimmäisen suositeltavaa lukea yllä mainittu Jupyter Notebook läpi. Tulemme käsittelemään aihetta myös live-luennolla.

### Vinkit

#### Kirjastot

Tutustu ainakin seuraaviin importattuihin luokkiin tai metodeihin:

```python
from sklearn.linear_model import SGDClassifier, LogisticRegression
```

Muista myös säätää mallin hyperparametreja. Esimerkiksi `SGDClassifier`-luokalla on useita parametreja, jotka vaikuttavat mallin oppimiseen ja suorituskykyyn. Voit käyttää `GridSearchCV`-luokkaa hyperparametrien arvojen haarukointiin. Kuten aiemminkin, muista dokumentoida oppimispäiväkirjassasi, mitä olet tehnyt ja miksi. Varmista, että ymmärrät oman koodisi.

#### Tavoite

Pyri vähintään 75 % tarkkuuteen. Alla classification report, joka on saavutettu `LogisticRegression` luokalla ilman minkään sortin älykästä hyperparametrien säätöä tai opettajan esikäsittelemän datasetin hiomista.

```
              precision    recall  f1-score   support

         0.0       0.68      0.28      0.39       687
         1.0       0.77      0.95      0.85      1796

    accuracy                           0.76      2483
   macro avg       0.73      0.61      0.62      2483
weighted avg       0.75      0.76      0.73      2483
```

#### Päiväkirjan parantelu

Tämä tehtävä saattaa osoittautua helpommaksi kuin arvasit. Nyt onkin hyvä aika ottaa kurssikirjat ja muut lähteet käsiin, ja varmistella, että oppimipäiväkirjasi faktat ovat tikissä. Jos törmäät väitteisiin, joista olet epävarma, kyseenalaista oma tekstisi. Etsi lähteistä, onko tosiaankin asia kuten olet kirjoittanut. ==Muista lähteviitteet!==
