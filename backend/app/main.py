from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.services.football_data import FootballDataClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = FootballDataClient()
    await client.start()
    app.state.football_data = client
    try:
        yield
    finally:
        await client.close()


app = FastAPI(title="Frpredictions API", version="0.1.0", lifespan=lifespan)
