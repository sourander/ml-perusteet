---
priority: 500
---

# Syklinen enkoodaus

Käsittelemme tämän viikkoteeman tehtävissä k-means -algoritmilla dataa, jossa on syklinen piirre nimeltään hue eli värin sävy (HSV-mallista). Hue on kulma, joka kuvaa värisävyä. Hue-arvo on välillä `0-360` ja se on syklinen, koska `0` ja `360` kuvaavat samaa väriä, aivan kuten 24-tuntisessa kellossa ajat `00:00` ja `24:00` kuvaavat samaa hetkeä vuorokaudesta. 

Ilman syklistä enkoodausta esimerkiksi hue-arvot 1 ja 359 ovat kaukana toisistaan Euklidisessa avaruudessa, vaikka ne edustavat lähes samaa väriä. Tämä rikkoo monien algoritmien, kuten k-meansin, etäisyyslaskennan oletukset.

Tyypillisin syklinen piirre, mihin törmäät jatkuvasti, on aika. Kellon tunnit, minuutit, viikonpäivät, kuukaudet... Kaikki nämä ovat syklisiä piirteitä, jotka vaativat erityistä käsittelyä.

Kaava tämän laskemiseen on:

$$
x_{sin} = sin(2\pi \cdot \frac{x}{x_{max}})
$$

$$
x_{cos} = cos(2\pi \cdot \frac{x}{x_{max}})
$$

Tai sama koodina:

```python
import numpy as np

def cyclical_encoding(x_values, cycle_max=None):
    x_max = cycle_max or x_values.max() # +1 if x_values are 0-indexed
    x_sin = np.sin(2 * np.pi * x_values / x_max)
    x_cos = np.cos(2 * np.pi * x_values / x_max)
    return x_sin, x_cos
```

Tyypillistä on, että syklin pituus on tunnettu etukäteen, kuten kellon tunnit (12 tai 24), minuutit (60), viikonpäivät (7) tai kuukaudet (12). Tällöin `cycle_max` tulee asettaa tähän tunnettuun arvoon. Tämä lieneekin ainut selkeä kompastuskivi, jossa voi sattua laskuvirheitä. Esimerkkiarvoja `cycle_max`-parametrille ovat..

* Tunnit = 24 (jos 0-23)
* Kuukaudet = 12 (jos 1-12)
* Viikonpäivät = 7 (jos 0-6)
* Viikonpäivät = 7 (jos 1-7)
* Viikonpäivät = 7 (jos 0-7, jossa 0 ja 7 ovat sunnuntai)

Aihe on selitetty selkeästi `feature-engine`-kirjaston sivuilla, joten tutustu [CyclicalFeatures](https://feature-engine.trainindata.com/en/latest/user_guide/creation/CyclicalFeatures.html)-dokumenttiin.

## Tehtävät

!!! question "Tehtävä: Kellon enkoodaus"

    Avaa `500_cyclical_encoding.py` Notebook ja tutustu sen koodiin.
    
    Voit halutessasi muokata tai jatkaa koodia: kenties haluat enkoodata vaikka vuorokauden minuutit? Nyt alkuperäinen kello on aina tasatunneilla, mutta entä jos kello olisi `04:42`?

## Lähteet

[^fe-cookbook]: Galli, S. *Python Feature Engineering Cookbook - Third Edition*. O'Reilly. 2024.
