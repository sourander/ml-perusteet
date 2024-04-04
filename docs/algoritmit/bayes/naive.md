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

## Naive Bayes luokittelija

Naive Bayes on Bayesin teoreeman implementaatio, jossa tehdään naiivi oletus, että yksittäiset datapisteet eivät ole keskenään riippuvaisia. Eli sanan `Viagra` esiintyminen viestissä lisää todennäköisyyttä, että viesti on roskapostia, mutta se ei vaikuta siihen, onko viestissä myös sana `Free` tai `Guaranteed`. Tämän naiivin oletuksen kanssa kaava on:

$$
\hat{y} = \arg\max_{y} P(y) \prod_{i=1}^{n} P(x_i|y)
$$

Saman voisi kirjoittaa Pythonina näin:

```python
DATASET = [
    ("Free Viagra now", "Spam"),
    ("A game of golf tomorrow?", "Not spam"),
]

def probability_of_word_being_in_class(word, y_val):
    # P(belief_n | y)
    n_docs = len([1 for x, y in DATASET if y == y_val and word in x.lower()])
    n_class = len([1 for x, y in DATASET if y == y_val])
    return n_docs / n_class

def predict_class_probability(evidence, y_val):
    # P(belief)
    prior = len([y for x, y in DATASET if y == y_val]) / len(DATASET)

    # SUM[ P(belief_n | y) ]
    likelihood = sum([probability_of_word_being_in_class(word, y_val) for word in evidence.lower().split()])

    # P(belief | evidence) * P(y)
    return prior * likelihood

evidence = "Free Viagra of something"
probabilities = [predict_class_probability(evidence, "Spam"), predict_class_probability(evidence, "Not spam")]
y_hat_prob, other_prob = max(probabilities), min(probabilities)
y_hat = probabilities.index(y_hat_prob)

print(f"Predicted class: {y_hat} (with probability {y_hat_prob} > {other_prob})")
```

!!! tip

    Huomaa, että toteutus olettaa koulutusdataan kuulumattomien sanojen olevan todennäköisyydeltään 0. Tätä "zero-frequency problem"-ongelmaa voi korjata käyttämällä Laplace smoothingia.
