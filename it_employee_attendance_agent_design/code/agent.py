try:
    from observability.observability_wrapper import (
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
except ImportError:  # observability module not available (e.g. isolated test env)
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):  # type: ignore[misc]
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass  # type: ignore[misc]
    def trace_tool_call(*_a, **_kw): pass  # type: ignore[misc]

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {'check_credentials_output': True,
 'check_jailbreak': True,
 'check_output': True,
 'check_pii_input': False,
 'check_toxic_code_output': True,
 'check_toxicity': True,
 'content_safety_enabled': True,
 'content_safety_severity_threshold': 4,
 'runtime_enabled': True,
 'sanitize_pii': True}


import os
import sys
import logging
import asyncio
import time as _time
from typing import Optional, Dict, Any, List, Union
from functools import lru_cache

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ValidationError, model_validator, constr
from dotenv import load_dotenv
import requests
import httpx

# Observability wrappers (trace_step, trace_step_sync, etc.) are injected by runtime

# Logging configuration
from loguru import logger

# Load .env if present
load_dotenv()

# Constants
BASE_URL = "https://attendance.example.com"
ATTENDANCE_RECORD_ENDPOINT = "/api/attendance/record"
LEAVE_ENDPOINT = "/api/leave"
CHECKIN_LOGS_ENDPOINT = "/api/checkin/logs"
SHIFT_RULES_ENDPOINT = "/api/shifts/rules"
HOLIDAY_CALENDAR_ENDPOINT = "/api/holidays/calendar"

# Allowed statuses
ATTENDANCE_STATUSES = ["Present", "Late Present", "Half Day", "Leave", "Absent", "Holiday"]

# --- Configuration Management ---

class Config:
    """
    Configuration loader for API keys, endpoints, and LLM credentials.
    """
    @staticmethod
    def get_attendance_api_token() -> Optional[str]:
        return os.getenv("ATTENDANCE_API_TOKEN")

    @staticmethod
    def get_azure_openai_api_key() -> Optional[str]:
        return os.getenv("AZURE_OPENAI_API_KEY")

    @staticmethod
    def get_azure_openai_endpoint() -> Optional[str]:
        return os.getenv("AZURE_OPENAI_ENDPOINT")

    @staticmethod
    def get_azure_openai_deployment() -> Optional[str]:
        return os.getenv("AZURE_OPENAI_DEPLOYMENT")

    @staticmethod
    def validate_attendance_api_token() -> bool:
        return bool(Config.get_attendance_api_token())

    @staticmethod
    @trace_agent(agent_name='IT Employee Attendance Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def validate_llm_config() -> bool:
        return all([
            Config.get_azure_openai_api_key(),
            Config.get_azure_openai_endpoint(),
            Config.get_azure_openai_deployment()
        ])

# --- Input Models and Validation ---

class AttendanceRequestModel(BaseModel):
    employee_id: constr(strip_whitespace=True, min_length=1, max_length=32)
    attendance_date: constr(strip_whitespace=True, min_length=8, max_length=10)

    @field_validator("employee_id")
    @classmethod
    def validate_employee_id(cls, v):
        if not v or not v.strip():
            raise ValueError("employee_id must not be empty.")
        if not v.isalnum():
            raise ValueError("employee_id must be alphanumeric.")
        return v.strip()

    @field_validator("attendance_date")
    @classmethod
    def validate_attendance_date(cls, v):
        import re
        # Accepts YYYY-MM-DD
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError("attendance_date must be in YYYY-MM-DD format.")
        return v

    @model_validator(mode="after")
    def check_length(self):
        if len(self.employee_id) > 32:
            raise ValueError("employee_id too long.")
        return self

class AttendanceResponseModel(BaseModel):
    success: bool
    status: Optional[str] = None
    explanation: Optional[str] = None
    api_result: Optional[dict] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    tips: Optional[str] = None

# --- Utility Functions ---

@with_content_safety(config=GUARDRAILS_CONFIG)
def mask_pii(text: str) -> str:
    """
    Masks PII such as employee IDs in logs and outputs.
    """
    import re
    # Mask employee IDs like E123 -> E***
    return re.sub(r"\bE\d+\b", lambda m: m.group(0)[0] + "***", text)

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_text(text: str) -> str:
    """
    Sanitizes input text for LLM and logs.
    """
    if not text:
        return ""
    return text.replace("\n", " ").replace("\r", " ").strip()

# --- Integration Layer: API Client ---

class AttendanceAPIClient:
    """
    Handles all API interactions for check-in logs, leave data, shift rules, holiday calendars, and attendance recording.
    """

    def __init__(self):
        self.base_url = BASE_URL

    def _get_auth_header(self) -> Dict[str, str]:
        token = Config.get_attendance_api_token()
        if not token:
            raise RuntimeError("Attendance API token not configured. Set ATTENDANCE_API_TOKEN in environment.")
        return {"Authorization": f"Bearer {token}"}

    @staticmethod
    def _handle_response(resp: requests.Response) -> Optional[dict]:
        try:
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            logger.error(f"API HTTP error: {e} - {resp.text}")
            return None
        except Exception as e:
            logger.error(f"API response parse error: {e}")
            return None

    @lru_cache(maxsize=32)
    def get_holiday_calendar(self, year: str) -> Optional[dict]:
        """
        Retrieves holiday calendar for a given year.
        """
        _t0 = _time.time()
        url = self.base_url + HOLIDAY_CALENDAR_ENDPOINT
        headers = self._get_auth_header()
        params = {"year": year}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=5)
            result = self._handle_response(resp)
            try:
                trace_tool_call(tool_name="AttendanceAPIClient.get_holiday_calendar",
                                latency_ms=int((_time.time() - _t0) * 1000),
                                output=str(result)[:200], status="success" if result else "fail")
            except Exception:
                pass
            return result
        except Exception as e:
            logger.error(f"Failed to get holiday calendar: {e}")
            return None

    @lru_cache(maxsize=64)
    def get_shift_rules(self, shift_id: str) -> Optional[dict]:
        """
        Retrieves shift rules for a given shift_id.
        """
        _t0 = _time.time()
        url = self.base_url + SHIFT_RULES_ENDPOINT
        headers = self._get_auth_header()
        params = {"shift_id": shift_id}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=5)
            result = self._handle_response(resp)
            try:
                trace_tool_call(tool_name="AttendanceAPIClient.get_shift_rules",
                                latency_ms=int((_time.time() - _t0) * 1000),
                                output=str(result)[:200], status="success" if result else "fail")
            except Exception:
                pass
            return result
        except Exception as e:
            logger.error(f"Failed to get shift rules: {e}")
            return None

    def get_leave_data(self, employee_id: str, date: str) -> Optional[dict]:
        """
        Retrieves leave data for an employee on a given date.
        """
        _t0 = _time.time()
        url = self.base_url + LEAVE_ENDPOINT
        headers = self._get_auth_header()
        params = {"employee_id": employee_id, "date": date}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=5)
            result = self._handle_response(resp)
            try:
                trace_tool_call(tool_name="AttendanceAPIClient.get_leave_data",
                                latency_ms=int((_time.time() - _t0) * 1000),
                                output=str(result)[:200], status="success" if result else "fail")
            except Exception:
                pass
            return result
        except Exception as e:
            logger.error(f"Failed to get leave data: {e}")
            return None

    def get_checkin_logs(self, employee_id: str, date: str) -> Optional[dict]:
        """
        Retrieves check-in logs for an employee on a given date.
        """
        _t0 = _time.time()
        url = self.base_url + CHECKIN_LOGS_ENDPOINT
        headers = self._get_auth_header()
        params = {"employee_id": employee_id, "date": date}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=5)
            result = self._handle_response(resp)
            try:
                trace_tool_call(tool_name="AttendanceAPIClient.get_checkin_logs",
                                latency_ms=int((_time.time() - _t0) * 1000),
                                output=str(result)[:200], status="success" if result else "fail")
            except Exception:
                pass
            return result
        except Exception as e:
            logger.error(f"Failed to get check-in logs: {e}")
            return None

    def record_attendance(self, employee_id: str, attendance_date: str, status: str) -> Optional[dict]:
        """
        Records attendance status for an employee.
        """
        _t0 = _time.time()
        url = self.base_url + ATTENDANCE_RECORD_ENDPOINT
        headers = self._get_auth_header()
        headers["Content-Type"] = "application/json"
        payload = {
            "employee_id": employee_id,
            "attendance_date": attendance_date,
            "status": status
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=5)
            result = self._handle_response(resp)
            try:
                trace_tool_call(tool_name="AttendanceAPIClient.record_attendance",
                                latency_ms=int((_time.time() - _t0) * 1000),
                                output=str(result)[:200], status="success" if result else "fail")
            except Exception:
                pass
            return result
        except Exception as e:
            logger.error(f"Failed to record attendance: {e}")
            return None

# --- Domain Layer: Data Validator ---

class AttendanceDataValidator:
    """
    Validates completeness and consistency of all required data sources.
    """
    def validate(self, data_bundle: dict) -> bool:
        # Required: checkin_logs, leave_data, shift_rules, holiday_calendar
        missing = []
        for key in ["checkin_logs", "leave_data", "shift_rules", "holiday_calendar"]:
            if key not in data_bundle or data_bundle[key] is None:
                missing.append(key)
        if missing:
            logger.warning(f"Attendance data missing: {missing}")
            return False
        return True

# --- Domain Layer: Policy Engine ---

class AttendancePolicyEngine:
    """
    Implements strict policy order and business rules for attendance classification.
    """
    def __init__(self, validator: AttendanceDataValidator):
        self.validator = validator

    def classify_attendance(self, data_bundle: dict) -> Union[str, dict]:
        """
        Applies strict policy order and business rules to classify attendance.
        Returns status string or error dict.
        """
        # Validate data
        if not self.validator.validate(data_bundle):
            return {"error": "ATTENDANCE_DATA_MISSING", "message": "Required data missing for classification."}

        # Extract data
        attendance_date = data_bundle.get("attendance_date")
        employee_id = data_bundle.get("employee_id")
        leave_data = data_bundle.get("leave_data")
        checkin_logs = data_bundle.get("checkin_logs")
        shift_rules = data_bundle.get("shift_rules")
        holiday_calendar = data_bundle.get("holiday_calendar")

        # 1. Holiday Check
        is_holiday = False
        try:
            holidays = holiday_calendar.get("holidays", [])
            is_holiday = attendance_date in [h.get("date") for h in holidays]
        except Exception:
            is_holiday = False

        if is_holiday:
            return "Holiday"

        # 2. Leave Check
        is_on_leave = False
        try:
            leaves = leave_data.get("leaves", [])
            is_on_leave = any(lv.get("date") == attendance_date and lv.get("status") == "Approved" for lv in leaves)
        except Exception:
            is_on_leave = False

        if is_on_leave:
            return "Leave"

        # 3. Check-in logic
        checkin_time = None
        try:
            checkin_time = checkin_logs.get("checkin_time")
        except Exception:
            checkin_time = None

        # Shift rules
        try:
            shift_start = shift_rules.get("shift_start")  # "09:00"
            grace_period = shift_rules.get("grace_period_minutes", 15)
            halfday_cutoff = shift_rules.get("halfday_cutoff_minutes", 120)
            absent_cutoff = shift_rules.get("absent_cutoff_minutes", 240)
        except Exception:
            return {"error": "ATTENDANCE_DATA_MISSING", "message": "Shift rules incomplete."}

        # Helper: convert time to minutes since midnight
        def time_to_minutes(tstr):
            try:
                h, m = map(int, tstr.split(":"))
                return h * 60 + m
            except Exception:
                return None

        shift_start_min = time_to_minutes(shift_start)
        checkin_min = time_to_minutes(checkin_time) if checkin_time else None

        if checkin_min is not None and shift_start_min is not None:
            diff = checkin_min - shift_start_min
            if diff <= grace_period:
                return "Present"
            elif grace_period < diff <= halfday_cutoff:
                return "Late Present"
            elif halfday_cutoff < diff <= absent_cutoff:
                return "Half Day"
            elif diff > absent_cutoff:
                return "Absent"
        else:
            # No check-in
            if not is_on_leave and not is_holiday:
                return "Absent"

        # Fallback
        return {"error": "POLICY_CONFLICT", "message": "Unable to classify attendance due to policy conflict."}

# --- Application Layer: Orchestrator ---

class AttendanceOrchestrator:
    """
    Coordinates the end-to-end attendance classification and recording process.
    """
    def __init__(self, api_client: AttendanceAPIClient, policy_engine: AttendancePolicyEngine,
                 validator: AttendanceDataValidator, notification_service, report_generator, llm_manager):
        self.api_client = api_client
        self.policy_engine = policy_engine
        self.validator = validator
        self.notification_service = notification_service
        self.report_generator = report_generator
        self.llm_manager = llm_manager

    async def classify_and_record_attendance(self, employee_id: str, attendance_date: str) -> dict:
        """
        Coordinates the classification and recording of attendance for a given employee and date.
        """
        async with trace_step(
            "gather_data", step_type="tool_call",
            decision_summary="Gather all required data for attendance classification",
            output_fn=lambda r: f"bundle_keys={list(r.keys()) if r else '?'}"
        ) as step:
            # Gather all data
            data_bundle = {
                "employee_id": employee_id,
                "attendance_date": attendance_date,
                "leave_data": self.api_client.get_leave_data(employee_id, attendance_date),
                "checkin_logs": self.api_client.get_checkin_logs(employee_id, attendance_date),
                "shift_rules": self.api_client.get_shift_rules("default"),  # Assume "default" shift for demo
                "holiday_calendar": self.api_client.get_holiday_calendar(attendance_date[:4])
            }
            step.capture(data_bundle)

        async with trace_step(
            "validate_data", step_type="process",
            decision_summary="Validate completeness and consistency of data sources",
            output_fn=lambda r: f"valid={r}"
        ) as step:
            valid = self.validator.validate(data_bundle)
            step.capture(valid)
            if not valid:
                self.notification_service.send_notification(employee_id, "Error", "Attendance data missing.")
                return {
                    "success": False,
                    "error_type": "ATTENDANCE_DATA_MISSING",
                    "error_message": "Required data missing for attendance classification.",
                    "tips": "Check if all data sources (leave, check-in, shift, holiday) are available."
                }

        async with trace_step(
            "classify_attendance", step_type="process",
            decision_summary="Apply attendance policy rules",
            output_fn=lambda r: f"status={r if isinstance(r,str) else r.get('error','?')}"
        ) as step:
            status = self.policy_engine.classify_attendance(data_bundle)
            step.capture(status)
            if isinstance(status, dict) and status.get("error"):
                self.notification_service.send_notification(employee_id, "Error", status.get("message", "Policy error."))
                return {
                    "success": False,
                    "error_type": status.get("error"),
                    "error_message": status.get("message"),
                    "tips": "Contact HR/admin for manual review."
                }

        async with trace_step(
            "record_attendance", step_type="tool_call",
            decision_summary="Record attendance status via API",
            output_fn=lambda r: f"api_result={str(r)[:100]}"
        ) as step:
            api_result = self.api_client.record_attendance(employee_id, attendance_date, status)
            step.capture(api_result)
            if not api_result:
                self.notification_service.send_notification(employee_id, "Error", "Failed to record attendance.")
                return {
                    "success": False,
                    "error_type": "API_ERROR",
                    "error_message": "Failed to record attendance via API.",
                    "tips": "Try again later or contact admin."
                }

        async with trace_step(
            "generate_explanation", step_type="llm_call",
            decision_summary="Generate explanation for attendance status",
            output_fn=lambda r: f"explanation_len={len(r) if r else 0}"
        ) as step:
            explanation = await self.llm_manager.generate_explanation(employee_id, attendance_date, status, data_bundle)
            step.capture(explanation)

        # Notification (async, fire-and-forget)
        asyncio.create_task(self.notification_service.send_notification(employee_id, status, explanation))

        return {
            "success": True,
            "status": status,
            "explanation": explanation,
            "api_result": api_result
        }

# --- Notification Service ---

class NotificationService:
    """
    Sends notifications to employees and HR/admin on attendance status or data issues.
    """
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def send_notification(self, employee_id: str, status: str, message: str) -> bool:
        # For demo: just log, in real use, integrate with email/SMS/Teams/Slack
        try:
            logger.info(f"Notify {mask_pii(employee_id)}: [{status}] {sanitize_text(message)}")
            try:
                trace_tool_call(tool_name="NotificationService.send_notification",
                                latency_ms=0,
                                output=f"{employee_id}:{status}", status="success")
            except Exception:
                pass
            return True
        except Exception as e:
            logger.error(f"Notification failed: {e}")
            return False

# --- Report Generator ---

class ReportGenerator:
    """
    Generates and exports attendance reports (daily/weekly/monthly).
    """
    def generate_report(self, params: dict) -> dict:
        # For demo: return dummy report
        try:
            report = {"report": "Attendance report generated.", "params": params}
            try:
                trace_tool_call(tool_name="ReportGenerator.generate_report",
                                latency_ms=0,
                                output=str(report)[:200], status="success")
            except Exception:
                pass
            return report
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {"error": str(e)}

# --- LLM Interaction Manager ---

class LLMInteractionManager:
    """
    Handles prompt construction, LLM calls, and response formatting.
    """
    def __init__(self, system_prompt: str, user_prompt_template: str, few_shot_examples: List[str]):
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.few_shot_examples = few_shot_examples
        self.model = "gpt-4.1"
        self.temperature = 0.7
        self.max_tokens = 2000

    def _get_llm_client(self):
        import openai
        api_key = Config.get_azure_openai_api_key()
        endpoint = Config.get_azure_openai_endpoint()
        deployment = Config.get_azure_openai_deployment()
        if not api_key or not endpoint or not deployment:
            raise ValueError("Azure OpenAI configuration missing. Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT.")
        return openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            azure_deployment=deployment
        )

    async def generate_explanation(self, employee_id: str, attendance_date: str, status: str, data_bundle: dict) -> str:
        """
        Generates a natural language explanation for attendance status.
        """
        user_prompt = self.user_prompt_template.format(employee_id=employee_id, attendance_date=attendance_date)
        # Add context from data_bundle
        context = f"Attendance status: {status}\n"
        context += f"Check-in logs: {data_bundle.get('checkin_logs')}\n"
        context += f"Leave data: {data_bundle.get('leave_data')}\n"
        context += f"Shift rules: {data_bundle.get('shift_rules')}\n"
        context += f"Holiday calendar: {data_bundle.get('holiday_calendar')}\n"

        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        for ex in self.few_shot_examples:
            messages.append({"role": "user", "content": ex.split("->")[0].strip()})
            messages.append({"role": "assistant", "content": ex.split("->")[1].strip()})
        messages.append({"role": "user", "content": f"{user_prompt}\n{context}"})

        _t0 = _time.time()
        try:
            client = self._get_llm_client()
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            content = response.choices[0].message.content
            try:
                trace_model_call(provider="azure", model_name=self.model,
                                 prompt_tokens=response.usage.prompt_tokens,
                                 completion_tokens=response.usage.completion_tokens,
                                 latency_ms=int((_time.time() - _t0) * 1000),
                                 response_summary=content[:200] if content else "")
            except Exception:
                pass
            return content
        except Exception as e:
            logger.error(f"LLM explanation failed: {e}")
            # Fallback to template
            return f"Employee {mask_pii(employee_id)} is marked as {status} on {attendance_date}. (Explanation unavailable due to system error.)"

# --- Main Agent ---

class AttendanceAgent:
    """
    Main agent class aggregating orchestrator, LLM manager, and tool integrations.
    """
    def __init__(self):
        self.api_client = AttendanceAPIClient()
        self.validator = AttendanceDataValidator()
        self.policy_engine = AttendancePolicyEngine(self.validator)
        self.notification_service = NotificationService()
        self.report_generator = ReportGenerator()
        self.llm_manager = LLMInteractionManager(
            system_prompt="You are an IT Employee Attendance Agent. Your job is to classify and record daily attendance for IT employees using check-in logs, leave data, shift rules, and holiday calendars. Always follow the strict policy order: Holiday > Leave > Present > Late Present > Half Day > Absent. Validate all data before making a decision and provide clear, casual explanations for each status.",
            user_prompt_template="Please provide the attendance details for {employee_id} on {attendance_date}.",
            few_shot_examples=[
                "What is the attendance status for employee E123 on 2024-06-10? -> Employee E123 is marked as Present on 2024-06-10 because their check-in was within the allowed grace period.",
                "Why is employee E456 marked as Late Present today? -> Employee E456 checked in after the grace period but before the half-day cutoff, so their status is Late Present."
            ]
        )
        self.orchestrator = AttendanceOrchestrator(
            self.api_client,
            self.policy_engine,
            self.validator,
            self.notification_service,
            self.report_generator,
            self.llm_manager
        )

    async def handle_attendance_request(self, employee_id: str, attendance_date: str) -> dict:
        """
        Handles the main attendance request.
        """
        async with trace_step(
            "handle_attendance_request", step_type="plan",
            decision_summary="Main entry for attendance classification and recording",
            output_fn=lambda r: f"success={r.get('success')}"
        ) as step:
            result = await self.orchestrator.classify_and_record_attendance(employee_id, attendance_date)
            step.capture(result)
            return result

# --- FastAPI App ---

app = FastAPI(
    title="IT Employee Attendance Agent",
    description="Automated agent for IT employee attendance classification and recording.",
    version="1.0.0"
)

# CORS (allow all for demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = AttendanceAgent()

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_type": "VALIDATION_ERROR",
            "error_message": str(exc),
            "tips": "Check your input fields and JSON formatting."
        }
    )

@app.exception_handler(HTTPException)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_type": "HTTP_ERROR",
            "error_message": exc.detail,
            "tips": "Check your request and try again."
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_type": "SERVER_ERROR",
            "error_message": "Internal server error.",
            "tips": "Contact support if the issue persists."
        }
    )

@app.post("/attendance/classify", response_model=AttendanceResponseModel)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def classify_attendance_endpoint(request: Request):
    """
    Endpoint to classify and record attendance for an employee.
    """
    try:
        data = await request.json()
    except Exception as e:
        logger.warning(f"Malformed JSON: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error_type": "MALFORMED_JSON",
                "error_message": "Malformed JSON in request body.",
                "tips": "Ensure your JSON is properly formatted (quotes, commas, braces)."
            }
        )
    try:
        model = AttendanceRequestModel(**data)
    except ValidationError as ve:
        logger.warning(f"Input validation failed: {ve}")
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error_type": "VALIDATION_ERROR",
                "error_message": str(ve),
                "tips": "Check employee_id and attendance_date fields."
            }
        )
    # Input size check
    if len(model.employee_id) > 32 or len(model.attendance_date) > 10:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error_type": "INPUT_TOO_LARGE",
                "error_message": "Input fields too large.",
                "tips": "employee_id max 32 chars, attendance_date max 10 chars."
            }
        )
    # Main logic
    result = await agent.handle_attendance_request(model.employee_id, model.attendance_date)
    # Mask PII in output
    if result.get("explanation"):
        result["explanation"] = mask_pii(result["explanation"])
    if result.get("status"):
        result["status"] = result["status"]
    return JSONResponse(
        status_code=200 if result.get("success") else 400,
        content=result
    )

@app.get("/health")
async def health_check():
    return {"success": True, "status": "ok"}

# --- Main Entrypoint ---



async def _run_with_eval_service():
    """Entrypoint: initialises observability then runs the agent."""
    import logging as _obs_log
    _obs_logger = _obs_log.getLogger(__name__)
    # ── 1. Observability DB schema ─────────────────────────────────────
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401 – register ORM models
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
    except Exception as _e:
        _obs_logger.warning('Observability DB init skipped: %s', _e)
    # ── 2. OpenTelemetry tracer ────────────────────────────────────────
    try:
        from observability.instrumentation import initialize_tracer
        initialize_tracer()
    except Exception as _e:
        _obs_logger.warning('Tracer init skipped: %s', _e)
    # ── 3. Evaluation background worker ───────────────────────────────
    _stop_eval = None
    try:
        from observability.evaluation_background_service import (
            start_evaluation_worker as _start_eval,
            stop_evaluation_worker as _stop_eval_fn,
        )
        await _start_eval()
        _stop_eval = _stop_eval_fn
    except Exception as _e:
        _obs_logger.warning('Evaluation worker start skipped: %s', _e)
    # ── 4. Run the agent ───────────────────────────────────────────────
    try:
        import uvicorn
        logger.info("Starting IT Employee Attendance Agent...")
        uvicorn.run("agent:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
        pass  # TODO: run your agent here
    finally:
        if _stop_eval is not None:
            try:
                await _stop_eval()
            except Exception:
                pass


if __name__ == "__main__":
    import asyncio as _asyncio
    _asyncio.run(_run_with_eval_service())