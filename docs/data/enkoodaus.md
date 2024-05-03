Aiemmin esitettiin, että koneoppimisessa operaatiot suoritetaan tilastotieteeseen nojaten ja useimmiten vektorisoidussa muodossa. Tämä asettaa käytännössä sen vaatimuksen, että kaikkien muuttujien tulee olla numeerisia. Sanamuotoiset kategoriset muuttujat, kuten arvoavaruus `{ "Cat", "Dog", "Hamster" }`, eivät ole numeerisia. Tästä syystä kategoriset muuttujat tulee enkoodata numeeriseen muotoon. Kaksi usein käytettyä menetelmää ovat:

* **Label encoding:** Kategoriset muuttujat muutetaan numeeriseen muotoon. Esimerkiksi `[Cat, Dog, Dog, Hamster]` => `[0, 1, 1, 2]`.
* **One-hot encoding:**  Kategiruset muuttujat muutetaan useiksi binäärimuuttujiksi. Esimerkiksi `[Cat, Dog, Dog, Hamster]` => `[[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]]`.

Huomaa, että label encoding tekee jostakin luokasta numerona suuremman kuin toisen. Tämä voi johtaa virheellisiin johtopäätöksiin, koska monet koneoppimismallit tulkitsevat suuruusjärjestyksen olevan merkityksellinen. Kategoriset muuttujat, kuten t-paidan koko (S, M, L, XL jne.) ovat järjestyksellisiä eli ordinaalisia. Tällöin niiden vaihtaminen numeroiksi voi olla perusteltua. Parempi ratkaisu olisi kuitenkin käyttää esimerkiksi rinnan ympärysmittaa tai muuta oikeaa mittaustulosta.

Kaikki kategoriset muuttujat eivät kuitenkaan ole ordinaalisia - ne ovat nominaalisia. Jos datasetissä on esimerkiksi piirre lempiväri, joka on kokonaislukuina enkoodattuna: `0: oranssi`ja `6: Sininen`, koneoppimismalli voi tulkita, että sininen on kuusi kertaa niin suuri kuin oranssi.

!!! question

    Mitä ongelmia koituisi, jos kääntäisit kategorisen lempivärin numeraaliseksi käyttäen värin hue-arvoa? Hue on numeerinen arvo välillä 0-360, joka kuvaa värin sävyä.

    Tämän pohtiminen voi olla helpompaa visuaalisesti. Käytä esimerkiksi [W3Schools: HTML HSL and HSLA Colors](https://www.w3schools.com/html/html_colors_hsl.asp)-interaktiivista työkalua ja mieti, mitä värejä edustavat todella pienet ja suuret hue-arvot.

Jos käytämme [Datasetti](datasetti.md)-materiaalista tuttua datasettiä, jossa kolmas sarake sisältää vain ja ainoastaan arvoja "R" ja "Python". Kyseessä on tekstimuotoinen kategorinen muuttuja, ja voimme enkoodata sen binääriseksi One-hot encoding -menetelmällä seuraavasti:

| Selittävä a | Selittävä b | is_python | Selitettävä muuttuja [y] |
| ----------- | ----------- | --------- | ------------------------ |
| 1.01        | 512         | 0         | 1                        |
| 1.10        | 124124      | 1         | 0                        |
| 0.99        | 42          | 0         | 1                        |
| 0.97        | N/A         | 0         | 1                        |
| 1.00        | 96141       | 1         | 0                        |

!!! warning

    Huomaa, että emme luo erikseen kahta eri saraketta: `is_r` ja `is_python`. Syy tälle on multikollineaarisuuden välttely. Emme halua sarakkeita, jotka ovat täysin korreloituneita keskenään, koska se voi aiheuttaa ongelmia mallin sovittamisessa.
