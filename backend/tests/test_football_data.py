import asyncio
from datetime import date

import httpx
import pytest
import respx

from app.core.config import Settings
from app.models import CompetitionCode, INITIAL_COMPETITIONS, ResponseSet
from app.services.football_data import (
    FootballDataAuthenticationError,
    FootballDataClient,
    FootballDataRateLimitError,
)


def configured_client(transport: httpx.AsyncBaseTransport | None = None):
    client = httpx.AsyncClient(
        base_url="https://api.football-data.org/v4",
        headers={"X-Auth-Token": "test-token"},
        transport=transport,
    )
    return FootballDataClient(
        Settings(football_data_api_key="test-token"), client=client
    )


def test_initial_competitions_are_configured():
    assert [item.code for item in INITIAL_COMPETITIONS] == [
        CompetitionCode.PREMIER_LEAGUE,
        CompetitionCode.LA_LIGA,
        CompetitionCode.SERIE_A,
        CompetitionCode.BUNDESLIGA,
        CompetitionCode.LIGUE_1,
        CompetitionCode.EREDIVISIE,
    ]


def test_matches_authenticate_and_apply_supported_filters():
    async def run():
        with respx.mock(base_url="https://api.football-data.org/v4") as router:
            route = router.get("/matches").mock(
                return_value=httpx.Response(200, json={"matches": []})
            )
            client = configured_client()
            try:
                result = await client.get_matches(
                    date_from=date(2026, 8, 1),
                    date_to=date(2026, 8, 31),
                    competition="PL,PD",
                    status="SCHEDULED",
                )
            finally:
                await client.close()

            assert isinstance(result, ResponseSet)
            assert route.called
            assert route.calls[0].request.headers["X-Auth-Token"] == "test-token"
            assert dict(route.calls[0].request.url.params) == {
                "dateFrom": "2026-08-01",
                "dateTo": "2026-08-31",
                "competitions": "PL,PD",
                "status": "SCHEDULED",
            }

    asyncio.run(run())


def test_competition_resources_use_supported_filters():
    async def run():
        with respx.mock(base_url="https://api.football-data.org/v4") as router:
            matches = router.get("/competitions/PL/matches").mock(
                return_value=httpx.Response(200, json={"matches": []})
            )
            standings = router.get("/competitions/PL/standings").mock(
                return_value=httpx.Response(200, json={"standings": []})
            )
            teams = router.get("/competitions/PL/teams").mock(
                return_value=httpx.Response(200, json={"teams": []})
            )
            client = configured_client()
            try:
                await client.get_competition_matches(
                    "PL", date_from=date(2026, 8, 1), date_to=date(2026, 8, 31),
                    status="SCHEDULED", season=2026,
                )
                await client.get_standings("PL", season=2026, matchday=1)
                await client.get_teams("PL", season=2026)
            finally:
                await client.close()

            assert dict(matches.calls[0].request.url.params) == {
                "dateFrom": "2026-08-01", "dateTo": "2026-08-31",
                "status": "SCHEDULED", "season": "2026",
            }
            assert dict(standings.calls[0].request.url.params) == {
                "season": "2026", "matchday": "1"
            }
            assert dict(teams.calls[0].request.url.params) == {"season": "2026"}

    asyncio.run(run())


def test_competition_detail_validates_available_seasons():
    async def run():
        with respx.mock(base_url="https://api.football-data.org/v4") as router:
            route = router.get("/competitions/PL").mock(
                return_value=httpx.Response(200, json={
                    "id": 2021,
                    "name": "Premier League",
                    "code": "PL",
                    "seasons": [
                        {"id": 2285, "startDate": "2025-08-01", "endDate": "2026-05-31"}
                    ],
                })
            )
            client = configured_client()
            try:
                detail = await client.get_competition("PL")
            finally:
                await client.close()
            assert route.called
            assert detail.code == "PL"
            assert detail.seasons[0].id == 2285
            assert detail.seasons[0].startDate == date(2025, 8, 1)

    asyncio.run(run())


def test_date_range_requires_both_values():
    with pytest.raises(ValueError, match="provided together"):
        FootballDataClient._date_filters(date(2026, 1, 1), None)


def test_unauthorized_and_rate_limited_responses_are_explicit():
    async def run():
        with respx.mock(base_url="https://api.football-data.org/v4") as router:
            router.get("/competitions").mock(return_value=httpx.Response(401))
            client = configured_client()
            try:
                with pytest.raises(FootballDataAuthenticationError):
                    await client.get_competitions()
            finally:
                await client.close()

        with respx.mock(base_url="https://api.football-data.org/v4") as router:
            router.get("/competitions").mock(
                return_value=httpx.Response(429, headers={"Retry-After": "60"})
            )
            client = configured_client()
            try:
                with pytest.raises(FootballDataRateLimitError) as error:
                    await client.get_competitions()
            finally:
                await client.close()
            assert error.value.retry_after == "60"

    asyncio.run(run())


def test_missing_key_fails_before_making_request():
    client = FootballDataClient(Settings(football_data_api_key=None))
    with pytest.raises(FootballDataAuthenticationError, match="FOOTBALL_DATA_API_KEY"):
        client._headers()
