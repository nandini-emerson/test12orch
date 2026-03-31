
# python

import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from loguru import logger

# Load environment variables from .env file if present
load_dotenv()

# --- API Key Management ---
def get_attendance_api_token() -> Optional[str]:
    token = os.getenv("ATTENDANCE_API_TOKEN")
    if not token:
        logger.error("Missing ATTENDANCE_API_TOKEN environment variable.")
        raise RuntimeError("Attendance API token is required for agent operation.")
    return token

def get_azure_openai_api_key() -> Optional[str]:
    key = os.getenv("AZURE_OPENAI_API_KEY")
    if not key:
        logger.error("Missing AZURE_OPENAI_API_KEY environment variable.")
        raise RuntimeError("Azure OpenAI API key is required for LLM operation.")
    return key

def get_azure_openai_endpoint() -> Optional[str]:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        logger.error("Missing AZURE_OPENAI_ENDPOINT environment variable.")
        raise RuntimeError("Azure OpenAI endpoint is required for LLM operation.")
    return endpoint

def get_azure_openai_deployment() -> Optional[str]:
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not deployment:
        logger.error("Missing AZURE_OPENAI_DEPLOYMENT environment variable.")
        raise RuntimeError("Azure OpenAI deployment name is required for LLM operation.")
    return deployment

# --- LLM Configuration ---
LLM_CONFIG = {
    "provider": "azure",
    "model": "gpt-4.1",
    "temperature": float(os.getenv("LLM_TEMPERATURE", 0.7)),
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", 2000)),
    "system_prompt": (
        "You are an IT Employee Attendance Agent. Your job is to classify and record daily attendance for IT employees "
        "using check-in logs, leave data, shift rules, and holiday calendars. Always follow the strict policy order: "
        "Holiday > Leave > Present > Late Present > Half Day > Absent. Validate all data before making a decision and "
        "provide clear, casual explanations for each status."
    ),
    "user_prompt_template": "Please provide the attendance details for {employee_id} on {attendance_date}.",
    "few_shot_examples": [
        "What is the attendance status for employee E123 on 2024-06-10? -> Employee E123 is marked as Present on 2024-06-10 because their check-in was within the allowed grace period.",
        "Why is employee E456 marked as Late Present today? -> Employee E456 checked in after the grace period but before the half-day cutoff, so their status is Late Present."
    ]
}

# --- Domain-specific Settings ---
DOMAIN = "general"
AGENT_NAME = "IT Employee Attendance Agent"
POLICY_ORDER = ["Holiday", "Leave", "Present", "Late Present", "Half Day", "Absent"]
ATTENDANCE_STATUSES = POLICY_ORDER

# --- API Endpoints ---
BASE_URL = "https://attendance.example.com"
API_ENDPOINTS = {
    "record_attendance": f"{BASE_URL}/api/attendance/record",
    "leave_data": f"{BASE_URL}/api/leave",
    "checkin_logs": f"{BASE_URL}/api/checkin/logs",
    "shift_rules": f"{BASE_URL}/api/shifts/rules",
    "holiday_calendar": f"{BASE_URL}/api/holidays/calendar"
}
API_HEADERS = {
    "Authorization": f"Bearer {get_attendance_api_token()}",
    "Content-Type": "application/json"
}

# --- Validation and Error Handling ---
def validate_config() -> None:
    errors = []
    try:
        get_attendance_api_token()
    except Exception as e:
        errors.append(str(e))
    try:
        get_azure_openai_api_key()
    except Exception as e:
        errors.append(str(e))
    try:
        get_azure_openai_endpoint()
    except Exception as e:
        errors.append(str(e))
    try:
        get_azure_openai_deployment()
    except Exception as e:
        errors.append(str(e))
    if errors:
        logger.error("Configuration validation failed: " + "; ".join(errors))
        raise RuntimeError("Agent configuration is incomplete. Please check environment variables.")

def get_api_headers() -> Dict[str, str]:
    try:
        return {
            "Authorization": f"Bearer {get_attendance_api_token()}",
            "Content-Type": "application/json"
        }
    except Exception as e:
        logger.error(f"API header generation failed: {e}")
        raise

def get_llm_settings() -> Dict[str, Any]:
    try:
        return {
            "api_key": get_azure_openai_api_key(),
            "endpoint": get_azure_openai_endpoint(),
            "deployment": get_azure_openai_deployment(),
            "config": LLM_CONFIG
        }
    except Exception as e:
        logger.error(f"LLM settings generation failed: {e}")
        raise

# --- Default Values and Fallbacks ---
DEFAULT_SHIFT_ID = os.getenv("DEFAULT_SHIFT_ID", "default")
DEFAULT_YEAR = os.getenv("DEFAULT_YEAR", "2024")
DEFAULT_ATTENDANCE_STATUS = "Absent"

# --- Exported Config Object ---
AGENT_CONFIG = {
    "agent_name": AGENT_NAME,
    "domain": DOMAIN,
    "llm_config": LLM_CONFIG,
    "policy_order": POLICY_ORDER,
    "attendance_statuses": ATTENDANCE_STATUSES,
    "api_endpoints": API_ENDPOINTS,
    "api_headers": API_HEADERS,
    "default_shift_id": DEFAULT_SHIFT_ID,
    "default_year": DEFAULT_YEAR,
    "default_attendance_status": DEFAULT_ATTENDANCE_STATUS,
    "validate_config": validate_config,
    "get_api_headers": get_api_headers,
    "get_llm_settings": get_llm_settings
}

# Validate config at import time
try:
    validate_config()
except Exception as e:
    logger.error(f"Agent config validation failed: {e}")
    # Optionally: exit or raise depending on deployment policy
    # raise

