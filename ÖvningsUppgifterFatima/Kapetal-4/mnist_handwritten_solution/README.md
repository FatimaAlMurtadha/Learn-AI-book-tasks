# MNIST – egna handskrivna bilder

Projektet tränar en Extra Trees-modell på MNIST och använder samma preprocessing för notebooken och Streamlit-appen.

## Installera

```bash
pip install -r requirements.txt
```

## Kör notebooken

Öppna `MNIST_mobile_preprocessing.ipynb` från projektmappen och kör cellerna uppifrån och ned. Notebooken laddar MNIST från OpenML, tränar och utvärderar modellen, provar exempelbilderna i `digits_images/` och sparar modell och metadata under `artifacts/`. Kör notebooken igen efter en ändring av scikit-learn-versionen, så att den sparade modellen och Streamlit använder samma version (`scikit-learn==1.6.0`).

## Starta Streamlit

```bash
streamlit run app/streamlit_app.py
```

Ladda upp en bild med **en** handskriven siffra. För bäst resultat: skriv mörkt och tydligt på ett ljust, enkelt papper, fotografera rakt ovanifrån med jämnt ljus och låt siffran fylla en stor del av bilden. Appen visar den upptäckta bläckmasken och den slutliga 28×28-bilden som skickas till modellen. Om beskärningen missar siffran eller masken ser fel ut, ta en ny bild med bättre kontrast eller mindre störande bakgrund.