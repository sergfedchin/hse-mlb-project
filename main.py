import os
import pickle

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import pycountry


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

    genres = [g.strip().capitalize() for g in form.genres.split(',')]
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

    with open(os.path.join('data', 'available_values.pkl'), 'r') as f:
        all_genres, all_languages, all_countries = pickle.load(f)  # TODO: fails for some reason, investigate

    print(all_genres)
    # TODO: check that genres, languages and countries have some matches with our available ones.
    print(release_year, runtime, budget, popularity, genres, production_countries, original_language)
    # TODO: configure the input vector and run model with the data.
    # Then write its predicted revenue range as str to 'prediction' variable.

    prediction = 0
    return JSONResponse(content={"revenue": prediction})
