# 📝 Tehtävät

## Tehtävä: Vihapuheen tunnistus

Kouluta Naive Bayes classifier, joka tunnistaa, ==onko viesti vihapuhetta vai ei==. Raportoi tulokset oppimispäiväkirjassasi. Valitse joko pienempi tai suurempi datasetti:

* Normaali tehtävä: [gh: hate-speech-and-offensive-language](https://github.com/t-davidson/hate-speech-and-offensive-language)
* Vaikea tehtävä: [Data in Brief: suurempi datasetti](https://www.sciencedirect.com/science/article/pii/S2352340922010356)

Kuten aina, dokumentoi se, mitä opit Naive Bayes algoritmista oppimispäiväkirjaasi.

Huomaa, että **sinun ei odoteta** käyttävän edistynyttä stemmausta. Tarkoitus on ymmärtää, miten Naive Bayes toimii ja miten sitä käytetään. Suosi yksinkertaisia ja ymmärrettäviä menetelmiä.

!!! warning

    Lue tehtävänanto dataan liittyvissä ongelmissa huolellisesti. Esimerkiksi yllä on esitelty, että sinun tulee tunnistaa, onko viesti vihapuhetta vai ei. Vastaus on siis "Kyllä on" tai "Ei ole". **Älä siis tee** luokitusta, jossa on useampia luokkia, kuten "vihapuhe", "loukkaava puhe", "ei kumpikaan".

    Sen sijaan voi olla perusteltua kokeilla, kuinka koulutetun mallin tarkkuus reagoi siihen, sisällytätkö loukkaavaksi puheeksi luokitellut viestit "ei kumpikaan"-luokkaan vai et.

### Vinkkejä

#### Kirjastot

Tulet tarvitsemaan `scikit-learn`-kirjastoa. Asenna se `uv add scikit-learn`-komennolla; se lisätään `pyproject.toml`-tiedostoon. Tutustu ainakin seuraaviin importattuihin luokkiin tai metodeihin:

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
```

Huomaa, että lista ei ole kattava. Kirjoitat oppimispäiväkirjaa, joten sinun on suositeltavaa tutustua vapaasti aihepiiriin. Älä tyydy *vastaamaan kysymykseen* vaan ota omistajuus omasta oppimisprosessistasi.


!!! tip

    Jos tutustut myös `ml-perusteet-code`-repositorion koodiin – eli siis niihin Allure-testattuihin koodeihin –, kannattaa pohtia, miten kyseisessä opettajan tekemässä *from scratch* -koodissa on hoidettu esimerkiksi tekstin vektorisointi. Onko käytössä vastaava tekniikka kuin TfidVectorizer tai CountVectorizer? Tekeekö malli sen itse vai onko se erillinen luokka/funktio kuten scikitin toteutuksessa?

#### Datan esikäsittely

Et voi syöttää dataa sinällään Naive Bayes -malliin. Sinun on ensin aivan vähimmillään muunnettava data numeeriseen muotoon, mutta kenties dataa kannattaa puhdistaa myös muutoin? Voit esimerkiksi haluta:

* Poista erikoismerkit
* Tee tekstistä lowercase
* Poista nimimerkit
* Poista yleisimmät sanat (usein termillä *stop words*)
* Korvaa urlit sanalla "urlhere" tai riisutulla domain-nimellä

Ethän tee tätä ensimmäisellä kielimallin ehdottamalla, todennäköisesti raskaalla tavalla, vaan suosi jotakin, minkä ymmärrät täysin. Voit tehdä sen käsin vaikkapa näin:

```python
import re
import pandas as pd

df = pd.read_csv('path_to_your_dataset.csv')

def preprocess_text(text):
    # Replace URLs with the word "urlhere"
    text = re.sub(r"http\S+|www\S+|https\S+", "urlhere", text, flags=re.MULTILINE)
    # Rest of the processing
    return text

df['cleaned_tweet'] = df['tweet'].apply(preprocess_text)
```