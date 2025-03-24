# ml-perusteet

Tämä on Johdatus Koneoppimiseen -oppimateriaali. Projekti löytyy tavallisena HTTP-sivustona verkosta (ks. linkki tämän sivun About-osiosta).

Projekti hyödyntää vahvasti [MkDocs](https://www.mkdocs.org/)-sivustogeneraattoria, joka luo `docs/`-hakemiston Markdown-tiedostoista staattisen nettisivuston. Projekti käyttää [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)-teemaa.

Tämä sivusto on luotu [Doc Skeleton](https://github.com/sourander/doc-skeleton)-templaatin avulla useiden muiden sivustojen ohessa: ne listataan [sourander.github.io](https://sourander.github.io/)-indeksisivustolla. Kaikkien näiden projektien konfiguraation- ja yhteisten elementtien ylläpito on keskitetty [doc-flesh](https://github.com/sourander/doc-flesh)-projektiin. Kyseessä on ikään kuin löyhä kopio Ansible-työkalusta: se puskee konfiguraatiotiedostoja ja yhteisiä elementtejä muihin projekteihin.

## Riippuvuudet

* [uv](https://docs.astral.sh/uv/)
    * ... ja sen riippuvuudet `pyproject.toml`-tiedostossa.

## Kuinka ajaa lokaalisti

```bash
# Kloonaa 
git clone 'this-repo-url'



# Aja development serveri
uv run mkdocs serve --open
```