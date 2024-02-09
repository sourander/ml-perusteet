# Python Perusteet

Tämä on Python Perusteet -oppimateriaali. Projekti löytyy tavallisena HTTP-sivustona verkosta (ks. linkki tämän sivun About-osiosta), mutta sen voi tarpeen mukaan ajaa myös lokaalisti.

Projekti nojaa vahvasti `Material for MkDocs`-nimiseen Python-kirjastoon. Kyseinen kirjasto luo `docs/`-kansion sisällön perusteella verkkosivuston.

Tämä sivusto on luotu [Doc Skeleton](https://github.com/sourander/doc-skeleton)-templaatin avulla.

## Riippuvuudet
* Python >=3.10
* Python Poetry

## Kuinka ajaa lokaalisti

```bash
# Kloonaa
git clone 'this-repo-url'

# Päivitä Python Poetry ja asenna projektin riippuvuudet
poetry self update
poetry install

# Käännä HTML-tiedostoiksi ja käynnistä serveri
poetry run mkdocs serve

# Jos haluat varmistaa, ettei docs/images/-kansioon ole jäänyt turhia kuvia, voit ajaa
poetry run find-images
```