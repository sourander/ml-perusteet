# Yleistä

Keskitymme tässä ensimmäisessä luvussa alan tärkeimpiin termeihin ja määritelmiin.

## Datatieteet

Tämä on kurssi koneoppimisesta ja tekoälystä, ja niihin vahvasti liittyvä käsite sekä tieteenala on *datatieteet (engl. data sciences)*.

> "Data science is the field of study that combines domain expertise, programming skills, and knowledge of mathematics and statistics to extract meaningful insights from data. Data science practitioners apply machine learning algorithms to numbers, text, images, video, audio, and more to produce artificial intelligence (AI) systems to perform tasks that ordinarily require human intelligence. In turn, these systems generate insights which analysts and business users can translate into tangible business value." - [DataRobot](https://www.datarobot.com/wiki/data-science/)

![Data_Science_VD](../images/Data_Science_VD.png)

**Kuvio 1:** *Venn-diagrammi datatieteistä. [Lähde: Dren Conway, The Data Science Venn Diagram (CC-BY)](http://drewconway.com/zia/2013/3/26/the-data-science-venn-diagram)*

Datatieteet ovat tieteenala, joka laittaa koneoppimisen käytäntöön. Konteksti, jossa koneoppiminen laitetaan käytäntöön liike-elämässä on jokin substanssiosaamista vaativa toimiala. Koneoppiminen tai tekoäly auttaa ratkaisemaan ongelmia. Henkilöt, jotka työskentelevät datatieteiden parissa ovat titteleiltään esimerkiksi data scientist, data analyst tai data engineer. Työnkuvissa voi olla päällekäisyyttä, mutta niissä on myös nyanssieroja.

!!! question

    Pohdi ja selvitä, mitä eroa on data scientistin, data analystin ja data engineerin työnkuvilla. Kenen työkalupakkiin kuuluisi todennäköisimmin ohjelmisto Tableau ja mitä se tekee? Kuka kirjoittaisi koodia, joka hyödyntää jotakin nimeltään Apache Spark? Entä pärjääkö yhden roolin osaaja ilman kahta muuta?

## Tekoäly

Tekoäly on vahvasti elokuvateollisuuden ja muun fiktion värittämä käsite. Osa fiktion tarjoamasta tiedosta on toki täyttä humpuukia, ja todellisuudessa tekoälyn hupun alta paljastuu pikemminkin tilastotiedettä ja matematiikkaa. Tämä ei kuitenkaan vähennä tekoälyn arvoa liiketoiminnan kannalta tärkeiden ongelmien ratkaisijana.

!!! question

    Onko jokin alla listatuista teoksista tuttu? Mitä tekoäly tarkoittaa kyseisessä tarinassa? Mitkä muut elokuvat, tv-sarjat tai kirjat kuuluisivat listalle?

    * 2001: A Space Odyssey (1968)
    * Hitchhiker's Guide to the Galaxy (1978/...)
    * Terminator (1984/...)
    * The Matrix (1999/...)
    * Moon (2009)
    * Her (2013)
    * Ex Machina (2014)

## Koneoppiminen

Tarkistetaan määritelmä kirjallisuudesta, jota löytyy [Finna-palvelusta](https://kamk.finna.fi/).

> Machine learning (ML) is a collection of algorithms and techniques used to design systems that learn from data. These systems are then able to perform predictions or deduce patterns from the supplied data.
> 
> Lähde: Lee, W. 2019. Python Machine Learning. Wiley.

Ja toinen määritelmä koneoppimiselle, jossa sitä verrataan yllä esiteltyyn AI-käsitteeseen.

> The machine learning portion of the picture enabled an AI to perform these tasks:
> 
> * Adapt to new circumstances that the original developer didn't envision
> * Detect patterns in all sorts of data sources
> * Create new behaviors based on the recognized patterns
> * Make decisions based on the success of failure of these behaviors.
> 
> Lähde: Mueller, P & Massaron, L. 2016. Machine Learning for Dummies. No Starch Press.

Tässä materiaalissa tutustutaan eri koneoppimisen algoritmeihin ja niiden käyttöön Python-ohjelmointikielellä. Materiaalissa tutustutaan myös datan esikäsittelyyn, jotta data olisi koneoppimisen algoritmeille sopivassa muodossa. Valmiiden kirjastojen tai palveluiden käytön sijasta pyrimme ymmärtämään, mitä koneoppimisen algoritmit tekevät ja miten ne toimivat, joten algoritmit koodataan pääasiallisesti itse. Tämä ei ole tuotannossa yleinen tapa, sillä valmiit kirjastot on yleisesti optimoitu käsin koodattua Python-skriptiä paremmin. Sen sijaan tämä on algoritmeihin tutustumisen kannalta hyödyllinen lähestymistapa.