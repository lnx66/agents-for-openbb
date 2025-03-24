from pydantic import BaseModel
import httpx
import asyncio


class Colour(BaseModel):
    hex: str


class ColourPalette(BaseModel):
    id: str
    name: str
    colours: list[Colour]
    url: str
    imageUrl: str


async def get_random_palettes(n: int = 1) -> str:
    """Get a random palette from ColourLovers.

    It is recommended to display the imageUrl in the UI
    for the user so that they can see the palette.

    Parameters:
        n: int = 1
            The number of palettes to return.
            Maximum is 10.


    """
    async with httpx.AsyncClient() as client:
        tasks = []
        for _ in range(n):
            tasks.append(
                client.get(
                    "https://www.colourlovers.com/api/palettes/random",
                    params={"format": "json"},
                    headers={"User-Agent": "OpenBB Example Copilot"},
                )
            )
        responses = await asyncio.gather(*tasks)

        if all(response.status_code == 200 for response in responses):
            response_str = "-- Palettes --\n"
            for response in responses:
                payload = response.json()[0]
                colours = [Colour(hex=f"#{colour}") for colour in payload["colors"]]
                palette = ColourPalette(
                    id=str(payload["id"]),
                    name=payload["title"],
                    colours=colours,
                    url=payload["url"],
                    imageUrl=payload["imageUrl"],
                )
                # Let's format the response that the LLM will see
                # as the result of the function call.
                response_str += f"name: {palette.name}\n"
                response_str += f"url: {palette.url}\n"
                response_str += f"imageUrl (use this to display the palette image): {palette.imageUrl}\n"
                response_str += f"colours: {[c.hex for c in palette.colours]}\n"
                response_str += "-----------------------------------\n"
            return response_str

        else:
            return "Failed to fetch palettes."
