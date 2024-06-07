import os
import pickle

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import pycountry
import pandas as pd
import numpy as np

from utils import preprocess_text, declassify_revenue


class MovieInfo(BaseModel):
    title: str
    release_year: str
    runtime: str
    budget: str
    genres: str
    production_countries: str
    original_language: str
    popularity: str
    overview: str


app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "main.html")


@app.post("/predict")
async def predictRevenue(request: Request,
                         form: MovieInfo):

    description = form.overview + form.title
    if len(description) == 0:
        return JSONResponse(content={"error": "Title or overview needed."}, status_code=500)

    try:
        release_year = int(form.release_year)
    except ValueError:
        return JSONResponse(content={"error": "Incorrect release year."}, status_code=500)

    try:
        runtime = int(form.runtime)
    except ValueError:
        return JSONResponse(content={"error": "Incorrect runtime."}, status_code=500)

    try:
        budget = int(form.budget)
    except ValueError:
        return JSONResponse(content={"error": "Incorrect budget."}, status_code=500)

    try:
        popularity = float(form.popularity)
    except ValueError:
        return JSONResponse(content={"error": "Incorrect popularity."}, status_code=500)

    genres = []
    for g in form.genres.split(','):
        norm_name = []
        for word in g.strip().split():
            norm_name.append(word.capitalize())
        genres.append(' '.join(norm_name))

    print(genres)
    if len(genres) == 0:
        return JSONResponse(content={"error": "At least one genre needed."}, status_code=500)

    production_countries = []
    for country in form.production_countries.split(','):
        if country:
            try:
                country_code = pycountry.countries.search_fuzzy(country)[0]
            except LookupError:
                return JSONResponse(content={"error": "Incorrect production country."}, status_code=500)
            production_countries.append(country_code.alpha_2)

    if len(production_countries) == 0:
        return JSONResponse(content={"error": "Incorrect production country."}, status_code=500)

    try:
        lang = pycountry.languages.get(name=form.original_language)
        original_language = lang.alpha_2
    except Exception:
        return JSONResponse(content={"error": "Incorrect original language."}, status_code=500)

    with open(os.path.join('data', 'supported_values.pkl'), 'rb') as f:
        all_genres, all_languages, all_countries = pickle.load(f)

    genres = list(set(genres) & set(all_genres))
    if len(genres) == 0:
        return JSONResponse(content={"error": "No supported genres were found."}, status_code=500)

    if original_language not in all_languages:
        return JSONResponse(content={"error": f"Language '{form.original_language}' not supported."}, status_code=500)

    production_countries = list(set(production_countries) & set(all_countries))
    if len(production_countries) == 0:
        return JSONResponse(content={"error": "No supported production countries were found."}, status_code=500)

    print(release_year, runtime, budget, popularity, genres, production_countries, original_language)

    with open(os.path.join('data', 'dataframe_header.csv')) as f:
        df = pd.read_csv(f)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df.loc[0] = [0] * len(df.columns)

    df.loc[0, 'budget'] = budget
    df.loc[0, 'popularity'] = popularity
    df.loc[0, 'runtime'] = runtime
    df.loc[0, 'release_year'] = release_year

    for genre in genres:
        df.loc[0, genre] = 1
    for prod_coutry in production_countries:
        df.loc[0, f"prod_country_{prod_coutry}"] = 1
    df.loc[0, f"spoken_lang_{original_language}"] = 1
    df.loc[0, f"original_language_{original_language}"] = 1

    df.drop(columns=['revenue', 'description'], inplace=True)

    description = preprocess_text(description)
    with open(os.path.join('models', 'bow.pkl'), 'rb') as f:
        bow: CountVectorizer = pickle.load(f)
    with open(os.path.join('models', 'text_model.pkl'), 'rb') as f:
        text_model: SVC = pickle.load(f)

    df['text_prediction'] = text_model.predict(bow.transform([description]))

    with open(os.path.join('models', 'scaler_model_text.pkl'), 'rb') as f:
        scaler: MinMaxScaler = pickle.load(f)

    df = scaler.transform(df)
    print(np.array(df))

    with open(os.path.join('models', 'prediction_model_text.pkl'), 'rb') as f:
        main_model: RandomForestClassifier = pickle.load(f)

    prediction = main_model.predict(df)[0]
    print(prediction)

    return JSONResponse(content={"revenue": declassify_revenue(prediction)})
