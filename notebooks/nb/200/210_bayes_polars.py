import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Naive Bayes with Polars

    The data schema from the Stat.fi PDF. You can find the original dataset at [stat.fi: Opetusaineistot](https://stat.fi/fi/palvelut/palvelut-tutkijoille/tutkimusaineistot/opetusaineistot). It has the heading **Työssäkäynnin ja yritystietojen opetuskäyttöaineistot** and it is the **Työssäkäynnin opetuskäyttöaineisto**.

    > –	© Tilastokeskus, Työssäkäynnin opetuskäyttöaineisto, 2011 (assumed year)

    | field | desc                                             | uniq | null |
    | ----- | ------------------------------------------------ | ---- | ---- |
    | vuosi | (surrogate) year of the data. (filtered with 15) | 15   | 0    |

    After filtering, we will keep the following columns. Within this subset, the values are as follows:

    | field      | desc                                      | uniq | null |
    | ---------- | ----------------------------------------- | ---- | ---- |
    | sukup      | 1=male, 2=female                            | 2    | 0    |
    | syntyv     | birth year (1940-1995)                    | 76   | 0    |
    | kieli      | fi = Finnish, sv = Swedish, 9 = other     | 3    | 0    |
    | sose       | socio-economic status (check `sose_map`)  | 5    | 0    |
    | ptoim1     | last year work status (check `ptoim_map)` | 7    | 0    |
    | tyotu      | income (capped to 100k)                   | 98  | 2029 |
    | suuralue12 | region (check `suuralue_map`)             | 6   | 6    |

    Data on Zip-pakettina repositoriossa (Git LFS). Saat sen purettua oikeaan lokaatioon ajamalla komennot...

    ```bash
    cd notebooks
    unzip gitlfs-store/tyossakaynnin-aineisto.zip
    ```
    """)
    return


@app.cell
def _():
    # Downloaded originally from: https://stat.fi/fi/palvelut/palvelut-tutkijoille/tutkimusaineistot/opetusaineistot
    # (c) Tilastokeskus
    CSV_PATH = "data/stat_fi/fleed_puf_julk.csv"
    return (CSV_PATH,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Data, Setup Helper Functions
    """)
    return


@app.cell
def _(CSV_PATH):
    def P(expr):
        """Probability that expr is true."""
        return df.select(expr).to_series().mean()

    def P_given(expr_A, expr_B, smooth=False):
        """Conditional probability P(A | B)."""
        p_b = P(expr_B)
        if p_b == 0:
            return None
        return P(expr_A & expr_B) / p_b

    def odds(expr):
        """Odds in favor of expr: P(A) / (1 - P(A))."""
        p = P(expr)
        if p == 1:
            return float("inf")
        if p == 0:
            return 0.0
        return p / (1 - p)

    def likelihood(expr_A, expr_B):
        """
        Likelihood of B given A, i.e. P(B | A).

        Think Bayes style:
        A = hypothesis / condition
        B = observed event / data
        """
        return P_given(expr_B, expr_A)


    def bayes(expr_A, expr_B):
        """
        Posterior probability P(A | B) using Bayes' rule:
            P(A | B) = P(A) * P(B | A) / P(B)
        """
        prior = P(expr_A)
        evidence = P(expr_B)

        if evidence == 0:
            return None   # or float("nan")

        return prior * likelihood(expr_A, expr_B) / evidence


    schema_overrides = {
        "vuosi": pl.Int32,
        "sukup": pl.Int8,        # mies, nainen
        "syntyv": pl.Int16,      # syntymävuosi
        "kieli": pl.Utf8,        # fi, sv, "9"
        "sose": pl.Int8,         # sosioekonominen asema (ks. sose_map)
        "ptoim1": pl.Int8,       # pääasiallinen toiminta (ks. ptoim_map)
        "tyotu": pl.Int32,       # työtulot
        "suuralue12": pl.Int8,   # aluekoodi (1=Etelä-Suomi jne.)
    }
    df = pl.read_csv(
        CSV_PATH,
        columns=[str(x) for x in schema_overrides.keys()],
        schema_overrides=schema_overrides,
    )

    # Drop the data collection year column
    df = df.filter(pl.col("vuosi") == 15).select(pl.exclude("vuosi"))

    # Drop those who have NULL salaries but are workers
    no_salary_worker=(pl.col("tyotu").is_null()) & (pl.col("ptoim1") == 11)
    df = df.filter(~no_salary_worker)

    # Give everyone else a salary of 0
    df = df.with_columns(pl.col("tyotu").fill_null(0))

    # 6 NULL-living people are now in Etelä-Suomi
    df = df.with_columns(pl.col("suuralue12").fill_null(1))

    df
    return P, P_given, bayes, df


@app.cell
def _(df):
    df.select("tyotu").n_unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Maps

    Here are dictionary-mapping that will help you understand the data values.
    """)
    return


@app.cell
def _():
    sose_map = {
        1: "Maa- ja metsätalousyrittäjät",
        2: "Yrittäjät",
        3: "Ylemmät toimihenkilöt",
        4: "Alemmat toimihenkilöt",
        5: "Työntekijät",
        6: "Opiskelijat",
        7: "Eläkeläiset",
        8: "Muut",
        9: "Tuntematon"
    }
    ptoim_map = {
        11: "Työllinen",
        12: "Työtön",
        21: "0–14-vuotias",
        22: "Opiskelija",
        24: "Eläkeläinen",
        25: "Varus- tai siviilipalvelusmies",
        29: "Työttömyyseläkeläinen",
        99: "Muu työvoiman ulkopuolella oleva"
    }
    suuralue_map = {
        1: "Etelä-Suomi",
        2: "Länsi-Suomi",
        3: "Itä-Suomi",
        4: "Pohjois-Suomi",
        5: "Ahvenanmaa"
    }
    sukup_male = {1: "mies", 2: "nainen"}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Easy problem

    You might not see any need for Bayes theorem when looking at a single variable. For example, let's look at:

    * Direct probability: Probability of being a male, given that someone earns 70k+.
    * Reverse probability: Probability of earning 70k+, given that a person is a male.
    """)
    return


@app.cell
def _(P, P_given, bayes):
    is_male = pl.col("sukup") == 1
    high_income = pl.col("tyotu") > 70_000

    print("Separate probabilities:")
    print(f"  P(male): {P(is_male):.2f}")
    print(f"  P(70k+): {P(high_income):.2f}")
    print("Conditional probabilities:")
    print(f"  Direct: P(male | 70k+) = {P_given(is_male, high_income):.2f}")
    print(f"  Reverse: P(70k+ | male) = {P_given(high_income, is_male):.2f}")
    print("\nOr the same reverse with ...")
    print(f"  BAYES(male | high salary) = {bayes(high_income, is_male):.2f}")
    return (high_income,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A Harder Example

    The benefits of Bayes' theorem become apparent when there are more features contributing to the prediction.

    $$
    P({sose=3} \mid \text{tyoty>70k}, {sukup=2}, {region=4})
    $$

    So, in a sentence: probability of being an upper white-collar employee, given that a person is 70k+ earning male from Northern Finland. Let us start with a direct estimate

    ## Direct Estimate
    """)
    return


@app.cell
def _(P_given, df, high_income):
    # High salary is already defined; here are the rest
    is_upper = pl.col("sose") == 3
    in_north = pl.col("suuralue12") == 4
    is_female = pl.col("sukup") == 2

    conditioning_expr = high_income & in_north & is_female

    P_direct = P_given(is_upper, conditioning_expr)
    print(f"P(upper white-collar | 70k+, female, Northern Finland) = {P_direct:.2f}") # 1.00
    df.filter(conditioning_expr)
    return in_north, is_female, is_upper


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When we filter the data to match **all three conditions at once**  (high income AND Northern Finland AND female), we get only three rows.

    **Conclusion**

    If we estimate the probability directly from only these three rows, the result will be unstable; just like estimating a probability using a sample size of 3. This motivates using **Naive Bayes** instead.

    ## Naive Bayes: breaking the problem into simpler pieces

    The direct approach tried to estimate `P(upper white‑collar | high income, Northern Finland, female)` from only three rows. Naive Bayes takes a different approach: instead of estimating the full conditional probability at once, it estimates:

    - the **prior** P(A)
    - the **likelihoods** P(B₁ | A), P(B₂ | A), P(B₃ | A)

    Here: The prior $A$ is “upper white‑collar”. The likelihoods are: $B_1$ = income, $B_2$ = Northern Finland, $B_3$ = female. Naive Bayes makes one key assumption: the evidence variables B₁, B₂, B₃ are **independent of each other given A**.

    Under this assumption:

    $$
    P(A \mid B_1, B_2, B_3)
    =
    \frac{P(A)\,P(B_1 \mid A)\,P(B_2 \mid A)\,P(B_3 \mid A)}{Z}
    $$


    Here, $Z$ is a placeholder for *“whatever value makes the probabilities sum to 1”*, a normalizing constant.  In the following code blocks, $Z$ is computed by adding the Naive Bayes scores for all possible classes – meaning the two: **is** and **is not** upper white-collar. In ML practice, e.g. in scikit-learn implementation, this might get skipped completely. You will learn about this in the next lesson.

    This mirrors the Pizza example:

    $$
    P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}
    $$

    except now we have **three pieces of evidence**, so we multiply three conditional probabilities. Later, when we normalize the result, the denominator plays the same role as in the Pizza formula — it is just computed by adding the scores for both classes instead of writing $P(B)$ explicitly.

    ### Why this helps?

    Each term uses **many rows** of the dataset, so the estimates are much more stable than the 3‑row direct estimate.
    """)
    return


@app.cell
def _(P, P_given, high_income, in_north, is_female, is_upper):
    P_A = P(is_upper)

    L_salary = P_given(high_income, is_upper)
    L_region = P_given(in_north, is_upper)
    L_gender = P_given(is_female, is_upper)

    print(f"P(A) = P(upper white-collar) = {P_A:.2f}")
    print(f"P(70k+ | A) = {L_salary:.2f}")
    print(f"P(northern Finland | A) = {L_region:.2f}")
    print(f"P(female | A) = {L_gender:.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From Naive Bayes score to a real probability

    The Naive Bayes score is proportional to but is not yet a proper probability.

    To convert it into a real probability, we compare it to the score of the *alternative class*. In a binary setting, that means: NOT upper white-collar. Thus, we compute the Naive Bayes score for both:

    - score_upper_white_collar
    - score_other

    Then we treat these as the $Z$ from previous cells. Using them, we normalize the score just like in the denominator of Bayes formula:

    $$
    P(\text{upper} \mid \text{evidence})
    =
    \frac{\text{score}_{\text{upper}}}
         {\text{score}_{\text{upper}} + \text{score}_{\text{not upper}}}
    $$

    This last step is the exact analogue of dividing by P(B) in the Pizza example:

    $$
    P(A|B) = \frac{P(B|A)P(A)}{P(B)}
    $$

    The next code cell does this calculation.
    """)
    return


@app.cell
def _(P, P_given, high_income, in_north, is_female, is_upper):
    is_not_upper = ~is_upper

    score_upper_white_collar = (
        P(is_upper)
        * P_given(high_income, is_upper)
        * P_given(in_north, is_upper)
        * P_given(is_female, is_upper)
    )


    score_other = (
        P(is_not_upper)
        * P_given(high_income, is_not_upper)
        * P_given(in_north, is_not_upper)
        * P_given(is_female, is_not_upper)
    )

    posterior_upper_white_collar = score_upper_white_collar / (
        score_upper_white_collar + score_other
    )


    print(
        f"Unnormalized Naive Bayes score for upper white-collar: {score_upper_white_collar:.6f}"
    )
    print(
        f"Unnormalized Naive Bayes score for not upper white-collar: {score_other:.6f}"
    )
    print(f"Normalized posterior probability: {posterior_upper_white_collar:.2f}")
    return


if __name__ == "__main__":
    app.run()
