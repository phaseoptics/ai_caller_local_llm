# app/outbound_call_handler.py
import os
import requests
from typing import Any, Dict, Optional


def create_outbound_call(
    to_number: str,
    twiml_url: str,
    status_callback_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Place an outbound call using Twilio's Calls API.

    Requirements (env):
      - TWILIO_ACCOUNT_SID
      - TWILIO_AUTH_TOKEN
      - TWILIO_FROM_NUMBER

    Parameters:
      - to_number: E.164 destination, for example +1XXXXXXXXXX
      - twiml_url: public URL Twilio will request for call instructions (your /voice)
      - status_callback_url: optional public URL for Twilio status callbacks
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    from_number = os.getenv("TWILIO_FROM_NUMBER", "").strip()

    if not account_sid or not auth_token:
        raise RuntimeError("Missing TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN")
    if not from_number:
        raise RuntimeError("Missing TWILIO_FROM_NUMBER")
    if not to_number:
        raise RuntimeError("Missing destination number (to_number)")
    if not twiml_url:
        raise RuntimeError("Missing twiml_url")

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls.json"

    data = {
        "To": to_number,
        "From": from_number,
        "Url": twiml_url,
        "Method": "POST",
    }

    if status_callback_url:
        data["StatusCallback"] = status_callback_url
        data["StatusCallbackMethod"] = "POST"
        data["StatusCallbackEvent"] = "initiated ringing answered completed"

    resp = requests.post(
        url,
        data=data,
        auth=(account_sid, auth_token),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()