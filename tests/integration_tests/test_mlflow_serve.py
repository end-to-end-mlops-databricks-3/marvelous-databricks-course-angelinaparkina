"""Tests for mlflow serve."""

import pandas as pd
import json
import pytest
import requests
from loguru import logger
from conftest import CATALOG_DIR

BASE_URL = "http://127.0.0.1:5000"

test_data = {
    "type_of_meal_plan": "Meal Plan 1",
    "room_type_reserved": "Room_Type 1",
    "market_segment_type": "Online",
    "no_of_adults": 2,
    "no_of_children": 0,
    "no_of_weekend_nights": 2,
    "avg_price_per_room": 90.95,
    "no_of_special_requests": 0,
    "no_of_previous_cancellations": 0,
    "repeated_guest": 0,
    "lead_time": 188,
    "booking_status": 1,
    "Booking_ID": "INN23596",
    "update_timestamp_utc": "2025-06-19 13:00:23.842000",
}

pandas_df = pd.DataFrame([test_data])

payload_dataframe_split = json.dumps({"dataframe_split": pandas_df.to_dict(orient="split")})
payload_dataframe_records = json.dumps({"dataframe_records": pandas_df.to_dict(orient="records")})


@pytest.mark.ci_exclude
def test_inference_server_health() -> None:
    """Test that the inference server health endpoint is reachable.

    Sends a GET request to the health endpoint and asserts a 200 status is returned
    """
    response = requests.get(f"{BASE_URL}/health")
    logger.info(f"Received {response.status_code}")
    assert response.status_code == 200


@pytest.mark.ci_exclude
def test_inference_server_ping() -> None:
    """Test that the inference server ping endpoint is reachable.

    Sends a GET request to the ping endpoint and asserts a 200 status code is returned.
    """
    response = requests.get(f"{BASE_URL}/ping")
    logger.info(f"Received {response.status_code}.")
    assert response.status_code == 200


@pytest.mark.ci_exclude
def test_inference_server_version() -> None:
    """Test that the inference server version endpoint returns the expected version.

    Sends a GET request to the version endpoint, asserts a 200 status code, and checks the version string.
    """
    response = requests.get(f"{BASE_URL}/version")
    logger.info(f"Received {response.status_code} with response of '{response.text}'.")
    assert response.status_code == 200
    assert response.text == "2.17.0"


@pytest.mark.ci_exclude
def test_inference_server_invocations_with_dataframe_split() -> None:
    """Test that the inference server correctly handles DataFrame split payload.

    Sends a POST request with a DataFrame split payload and verifies the response contains a list of float predictions.
    """
    response = requests.post(
        f"{BASE_URL}/invocations", data=payload_dataframe_split, headers={"Content-Type": "application/json"}, timeout=2
    )
    logger.info(f"Received {response.status_code} with response of '{response.text}'.")
    assert response.status_code == 200
    logger.info(f"Received {response.json()}")
    values = response.json()["predictions"]
    assert isinstance(values, list)
    assert isinstance(values[0], int)


@pytest.mark.ci_exclude
def test_inference_server_invocations_with_dataframe_records_should_fail_when_contact_request_violation() -> None:
    """Test that the inference serve invocations with incomplete DataFrame records fail as expected.

    Drops each column from the dataframe in turn and verifies that the server returns a 400 error.
    """
    for col in pandas_df.columns.to_list():
        tmp_df = pandas_df.drop(columns=[col])

        tmp_payload_dataframe_records = json.dumps({"dataframe_records": tmp_df.to_dict(orient="records")})
        logger.info(f"Testing with {col} dropped.")
        response = requests.post(
            f"{BASE_URL}/invocations",
            data=tmp_payload_dataframe_records,
            headers={"Content-Type": "application/json"},
            timeout=2,
        )
        logger.info(f"Received {response.status_code} with response of '{response.text}'.")
        assert response.status_code == 400


@pytest.mark.ci_exclude
def test_inference_server_invocations_with_full_dataframe() -> None:
    """Test that the inference server processes a full dataframe and returns correct predictions.

    Loads test data, sends a POST request, and verifies the response contains a list of float predictions of correct length.
    """
    test_set = pd.read_csv(f"{CATALOG_DIR.as_posix()}/test_set.csv")
    input_data = test_set.drop(columns=["Booking_ID", "booking_status"])
    input_data = input_data.where(input_data.notna(), None)
    input_data = input_data.to_dict(orient="records")
    payload = json.dumps({"dataframe_records": input_data})

    response = requests.post(
        f"{BASE_URL}/invocations", data=payload, headers={"Content-Type": "application/json"}, timeout=2
    )
    logger.info(f"Received {response.status_code} with response of '{response.text}'.")
    assert response.status_code == 200
    values = response.json()["predictions"]
    assert isinstance(values, list)
    assert len(values) == len(input_data)
    assert all(isinstance(value, int) for value in values)
