from datetime import date
import ssl
from typing import Any

import httpx
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.models import Competition, ResponseSet


class FootballDataError(Exception):
    """Base exception for football-data.org request failures."""


class FootballDataAuthenticationError(FootballDataError):
    pass


class FootballDataRateLimitError(FootballDataError):
    def __init__(self, message: str, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class FootballDataRequestError(FootballDataError):
    pass


class FootballDataResponseError(FootballDataError):
    pass


class FootballDataClient:
    """Authenticated client for the football-data.org v4 API."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._timeout = timeout

    async def __aenter__(self) -> "FootballDataClient":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.settings.football_data_base_url.rstrip("/"),
                timeout=self._timeout,
                verify=ssl.create_default_context(),
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _headers(self) -> dict[str, str]:
        token = self.settings.football_data_api_key
        if not token:
            raise FootballDataAuthenticationError(
                "FOOTBALL_DATA_API_KEY is not configured. Add it to the backend environment."
            )
        return {"X-Auth-Token": token}

    async def _get_payload(
        self, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if self._client is None:
            await self.start()
        assert self._client is not None

        try:
            response = await self._client.get(
                path, params=params, headers=self._headers()
            )
        except httpx.TimeoutException as exc:
            raise FootballDataRequestError("football-data.org request timed out") from exc
        except httpx.RequestError as exc:
            raise FootballDataRequestError("Could not reach football-data.org") from exc

        if response.status_code in (401, 403):
            raise FootballDataAuthenticationError(
                "football-data.org rejected authentication or the account lacks endpoint access"
            )
        if response.status_code == 429:
            raise FootballDataRateLimitError(
                "football-data.org rate limit reached",
                retry_after=response.headers.get("Retry-After"),
            )
        if not response.is_success:
            raise FootballDataRequestError(
                f"football-data.org returned HTTP {response.status_code}"
            )

        try:
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Expected an object response")
            return payload
        except (ValueError, ValidationError) as exc:
            raise FootballDataResponseError(
                "football-data.org returned an invalid response"
            ) from exc

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> ResponseSet:
        try:
            return ResponseSet.model_validate(await self._get_payload(path, params))
        except ValidationError as exc:
            raise FootballDataResponseError(
                "football-data.org returned an invalid response"
            ) from exc

    @staticmethod
    def _date_filters(
        date_from: date | None, date_to: date | None
    ) -> dict[str, str]:
        if (date_from is None) != (date_to is None):
            raise ValueError("date_from and date_to must be provided together")
        if date_from and date_to and date_from > date_to:
            raise ValueError("date_from must be before or equal to date_to")
        filters: dict[str, str] = {}
        if date_from:
            filters["dateFrom"] = date_from.isoformat()
        if date_to:
            filters["dateTo"] = date_to.isoformat()
        return filters

    async def get_competitions(self) -> ResponseSet:
        return await self._get("/competitions")

    async def get_competition(self, competition_id: str) -> Competition:
        try:
            return Competition.model_validate(
                await self._get_payload(f"/competitions/{competition_id}")
            )
        except ValidationError as exc:
            raise FootballDataResponseError(
                "football-data.org returned an invalid competition response"
            ) from exc

    async def get_matches(
        self,
        *,
        date_from: date | None = None,
        date_to: date | None = None,
        competition: str | None = None,
        status: str | None = None,
    ) -> ResponseSet:
        params: dict[str, Any] = self._date_filters(date_from, date_to)
        if competition:
            params["competitions"] = competition
        if status:
            params["status"] = status
        return await self._get("/matches", params or None)

    async def get_competition_matches(
        self,
        competition_id: str,
        *,
        date_from: date | None = None,
        date_to: date | None = None,
        status: str | None = None,
        season: int | None = None,
        matchday: int | None = None,
    ) -> ResponseSet:
        params: dict[str, Any] = self._date_filters(date_from, date_to)
        if status:
            params["status"] = status
        if season is not None:
            params["season"] = season
        if matchday is not None:
            params["matchday"] = matchday
        return await self._get(f"/competitions/{competition_id}/matches", params or None)

    async def get_standings(
        self,
        competition_id: str,
        *,
        season: int | None = None,
        matchday: int | None = None,
        as_of: date | None = None,
    ) -> ResponseSet:
        params: dict[str, Any] = {}
        if season is not None:
            params["season"] = season
        if matchday is not None:
            params["matchday"] = matchday
        if as_of is not None:
            params["date"] = as_of.isoformat()
        return await self._get(f"/competitions/{competition_id}/standings", params or None)

    async def get_teams(
        self, competition_id: str, *, season: int | None = None
    ) -> ResponseSet:
        params = {"season": season} if season is not None else None
        return await self._get(f"/competitions/{competition_id}/teams", params)
