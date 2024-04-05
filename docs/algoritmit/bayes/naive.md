Päädyimme edellisessä luvussa Bayesin teemassa muotoon, joka näkyy alla. Tällä kertaa olemme tosin korvanneet `A`:n sanalla belief ja `B`:n sanalla evidence.

$$
P(\text{belief}|\text{evidence}) = \frac{P(\text{evidence}|\text{belief}) \times P(\text{belief})}{P(\text{evidence})}
$$

Tämän voisi kirjoittaa myös seuraavalla tavalla:

$$
\text{posterior probability} = \frac{\text{likelihood} \times \text{prior probability}}{\text{evidence}}
$$

Kuten [scikit learn:n dokumentaatio sanoo](https://scikit-learn.org/stable/modules/naive_bayes.html), jakaja on datasetin suhteen vakio, joten sen voi jättää pois. Jakajan hylkääminen tekee lopputuloksesta normalisoimattoman, mutta se ei ole ongelma, koska meitä kiinnostaa vain suhteellinen todennäköisyys. Alla olevan kaavan kalan näköinen symboli luetaan "proportional to". Tämä tarkoittaa, että vasen puoli on suoraan verrannollinen oikean puolen tulon kanssa.

$$
P(\text{belief}|\text{evidence}) \propto P(\text{evidence}|\text{belief}) \times P(\text{belief})
$$

Voimme myös korvata kaavasta sanan `belief` sanalla `spam` ja sanan `evidence` useita datasetissä olevia sanoja korvaavalla `x_1, x_2, ..., x_n`.

$$
P(spam|x_1, x_2, ..., x_n) \propto P(x_1, x_2, ..., x_n|spam) \times P(spam)
$$

## Naive Bayes luokittelija

Naive Bayes on Bayesin teoreeman implementaatio, jossa tehdään naiivi oletus, että yksittäiset datapisteet eivät ole keskenään riippuvaisia. Eli sanan `Viagra` esiintyminen viestissä lisää todennäköisyyttä, että viesti on roskapostia täysin riippumatta siitä, mitä muita sanoja sähköposti sisältää. Kukin sana osallistuu todennäköisyyteen itsenäisesti. Tämän naiivin oletuksen kanssa kaava on:

$$
\hat{y} = \arg\max_{y} P(y) \prod_{i=1}^{n} P(x_i|y)
$$

Saman voisi kirjoittaa Pythonina näin:

```python
from math import prod

DATASET = [
    ("Free Viagra now", "Spam"),
    ("A game of golf tomorrow?", "Not spam"),
]

def probability_of_word_being_in_class(word, y_val):
    """ P(edivence_i | y) """

    # N(x_i | y): (1)
    n_docs = sum(
        [
            1 for sentence, y 
            in DATASET 
            if y == y_val and word in sentence.lower()
        ]
    )

    # N(y): (2)
    n_class = len([1 for x, y in DATASET if y == y_val])
    
    # Laplace smoothing (3)
    n_docs += 0.5
    n_class += 1

    return n_docs / n_class

def predict_class_probability(evidence, y_val):
    # P(y) (4)
    prior = sum(
        [
            1 for _, y 
            in DATASET 
            if y == y_val
            ]
    ) / len(DATASET)

    # product[ P(evidence_i | y) ]
    likelihood = prod(
        [
            probability_of_word_being_in_class(word, y_val) 
            for word in evidence.lower().split()
        ]
    )

    return  likelihood * prior

############ TESTILAUSE #############
evidence = "Free Viagra of something"

probabilities = [
    predict_class_probability(evidence, "Spam"), 
    predict_class_probability(evidence, "Not spam")
]

y_hat_prob, other_prob = max(probabilities), min(probabilities)
y_hat = probabilities.index(y_hat_prob)

print(f"Predicted class: {y_hat} (with probability {y_hat_prob} > {other_prob})")
```

1. Kyseisen sanan sisältävien viestien määrä luokassa [spam/no spam]
2. Lauseiden määrä datasetissä, jotka eduustavat luokkaa [spam/no spam]
3. Smoothing estää tilanteen, jossa jokin sana ei esiinny yhdessäkään viestissä. Tällöin todennäköisyys olisi 0, ja koko lasku menisi nollalla, koska mikä tahansa kertaa 0 on 0.
4. Luokan todennäköisyys [spam/no spam]. Eli mikä osuus viesteistä on esimerkiksi roskapostia.


!!! tip

    Huomaa, että toteutus on aivan äärimmäisen yksinkertaistettu ja tarkoitettu pienten datasettien testailuun. Kestävämmän toteutuksen löydät kirjasta "Data Science from Scratch". Vielä paremman toteutuksen löydät scikit-learn kirjastosta.

  
