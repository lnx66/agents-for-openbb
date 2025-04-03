import random
from typing import AsyncGenerator
from common.agent import reasoning_step
from pydantic import BaseModel
import httpx
import logging

logger = logging.getLogger(__name__)


class Rating(BaseModel):
    average: float
    reviews: int


class Beer(BaseModel):
    id: int
    price: str
    name: str
    rating: Rating
    image: str


async def get_random_stout_beers(n: int = 1) -> AsyncGenerator[str, None]:
    """Get a random stout beer from the Beer API.

    It is recommended to display the image url in the UI
    for the user so that they can see the beer.

    Parameters:
        n: int = 1
            The number of beers to return.
            Maximum is 10.


    """
    yield reasoning_step(
        event_type="INFO",
        message="Fetching random stout beers...",
        details={"number of beers": n},
    )

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.sampleapis.com/beers/stouts",
            headers={"User-Agent": "OpenBB Example Copilot"},
        )
        if response.status_code != 200:
            yield reasoning_step(
                event_type="ERROR",
                message="Failed to fetch beers.",
                details={"error": "Failed to fetch beers."},
            )
            yield "Failed to fetch beers."
            return

        yield reasoning_step(
            event_type="INFO",
            message="Beers fetched successfully.",
        )

        data = response.json()
        random_sample = random.sample(data, n)
        beers = [Beer(**beer) for beer in random_sample]

        response_str = "-- Beers --\n"
        for beer in beers:
            response_str += f"name: {beer.name}\n"
            response_str += f"price: {beer.price}\n"
            response_str += f"rating: {beer.rating.average}\n"
            response_str += f"reviews: {beer.rating.reviews}\n"
            response_str += (
                f"image (use this to display the beer image): {beer.image}\n"
            )
            response_str += "-----------------------------------\n"
        yield response_str
        return
