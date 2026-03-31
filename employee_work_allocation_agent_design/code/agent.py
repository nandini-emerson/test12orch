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
import asyncio
import logging
import time as _time
from typing import List, Optional, Dict, Any, Tuple, Union
from functools import lru_cache

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ValidationError, model_validator, ConfigDict
from dotenv import load_dotenv
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger

# Observability wrappers are injected by runtime (do not import/define trace_agent, trace_step, etc.)

# Load .env if present
load_dotenv()

# --- Logging Configuration ---
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")

# --- Configuration Management ---
class Config:
    """Configuration loader for API keys, endpoints, and LLM settings."""
    @staticmethod
    def get_oauth_token() -> Optional[str]:
        return os.getenv("WORKFORCE_API_OAUTH_TOKEN")

    @staticmethod
    def get_azure_openai_key() -> Optional[str]:
        return os.getenv("AZURE_OPENAI_API_KEY")

    @staticmethod
    def get_azure_openai_endpoint() -> Optional[str]:
        return os.getenv("AZURE_OPENAI_ENDPOINT")

    @staticmethod
    def get_azure_openai_deployment() -> Optional[str]:
        return os.getenv("AZURE_OPENAI_DEPLOYMENT")

    @staticmethod
    def validate():
        missing = []
        if not Config.get_oauth_token():
            missing.append("WORKFORCE_API_OAUTH_TOKEN")
        if not Config.get_azure_openai_key():
            missing.append("AZURE_OPENAI_API_KEY")
        if not Config.get_azure_openai_endpoint():
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not Config.get_azure_openai_deployment():
            missing.append("AZURE_OPENAI_DEPLOYMENT")
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

# --- Constants ---
BASE_URL = "https://workforce.example.com/api/v1"
HEADERS = {
    "Content-Type": "application/json"
}

# --- Utility Functions ---
@with_content_safety(config=GUARDRAILS_CONFIG)
def mask_pii(data: Any) -> Any:
    """Mask employee_id and other sensitive fields in logs."""
    if isinstance(data, dict):
        return {k: ("***" if "employee_id" in k else mask_pii(v)) for k, v in data.items()}
    elif isinstance(data, list):
        return [mask_pii(item) for item in data]
    return data

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_text(text: str) -> str:
    """Sanitize input text for LLM and logs."""
    return text.strip().replace('\x00', '').replace('\r', '').replace('\n', ' ').replace('\t', ' ')

@with_content_safety(config=GUARDRAILS_CONFIG)
def redact_sensitive(text: str) -> str:
    """Redact sensitive info from output."""
    # For now, just mask employee IDs (simple pattern)
    import re
    return re.sub(r'\b(emp|employee)[-_]?\d+\b', '***', text, flags=re.IGNORECASE)

# --- Pydantic Models ---
class Employee(BaseModel):
    employee_id: str = Field(..., min_length=1, max_length=64)
    name: Optional[str] = None
    attendance_status: Optional[str] = None
    skills: Optional[List[str]] = None
    capacity: Optional[float] = None  # e.g., 8.0 for 8 hours
    model_config = ConfigDict(extra="ignore")

    @field_validator("employee_id")
    @classmethod
    def validate_employee_id(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("employee_id cannot be empty")
        return v

class Task(BaseModel):
    task_id: str = Field(..., min_length=1, max_length=64)
    name: Optional[str] = None
    required_skills: Optional[List[str]] = None
    priority: Optional[str] = None
    due_date: Optional[str] = None  # ISO date
    dependencies: Optional[List[str]] = None
    model_config = ConfigDict(extra="ignore")

    @field_validator("task_id")
    @classmethod
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def validate_task_id(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("task_id cannot be empty")
        return v

class Allocation(BaseModel):
    employee_id: str
    task_id: str
    allocation_percentage: float

class AllocationRequestModel(BaseModel):
    tasks: List[Task]
    employees: List[Employee]
    date: str = Field(..., min_length=8, max_length=10)  # e.g., '2024-06-10'

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        import re
        if not re.match(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError("date must be in YYYY-MM-DD format")
        return v

    @model_validator(mode="after")
    def check_lists(self):
        if not self.tasks or not self.employees:
            raise ValueError("Both tasks and employees lists must be provided and non-empty.")
        return self

class AllocationResultModel(BaseModel):
    success: bool
    allocations: Optional[List[Allocation]] = None
    errors: Optional[List[str]] = None
    explanation: Optional[str] = None

class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None

class NotificationResult(BaseModel):
    success: bool
    notified: List[str] = []
    failed: List[str] = []
    error: Optional[str] = None

class LogEntry(BaseModel):
    timestamp: float
    event: str
    details: Dict[str, Any]

class AllocationEvent(BaseModel):
    event_type: str
    allocations: Optional[List[Allocation]] = None
    error: Optional[str] = None
    user: Optional[str] = None
    timestamp: Optional[float] = None

# --- Infrastructure Layer: API Clients ---
class BaseAPIClient:
    def __init__(self):
        self._token = None

    def get_token(self) -> str:
        if not self._token:
            self._token = Config.get_oauth_token()
            if not self._token:
                raise RuntimeError("WORKFORCE_API_OAUTH_TOKEN not configured")
        return self._token

    def get_headers(self) -> Dict[str, str]:
        headers = HEADERS.copy()
        headers["Authorization"] = f"Bearer {self.get_token()}"
        return headers

    @property
    def client(self):
        # Use a shared httpx.AsyncClient for connection pooling
        if not hasattr(self, "_client"):
            self._client = httpx.AsyncClient(timeout=10)
        return self._client

class AttendanceAPIClient(BaseAPIClient):
    """Fetches employee attendance status."""
    ENDPOINT = f"{BASE_URL}/attendance/status"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=1, max=5),
           retry=retry_if_exception_type(httpx.RequestError))
    async def get_attendance(self, employee_ids: Optional[List[str]] = None, date: Optional[str] = None) -> Dict[str, Any]:
        params = {}
        if employee_ids:
            params["employee_id"] = ",".join(employee_ids)
        if date:
            params["date"] = date
        async with trace_step(
            "attendance_api_call", step_type="tool_call",
            decision_summary="Fetch attendance status for employees",
            output_fn=lambda r: f"attendance keys={list(r.keys()) if isinstance(r, dict) else '?'}"
        ) as step:
            try:
                _obs_t0 = _time.time()
                resp = await self.client.get(
                    self.ENDPOINT,
                    headers=self.get_headers(),
                    params=params
                )
                try:
                    trace_tool_call(
                        tool_name='client.get',
                        latency_ms=int((_time.time() - _obs_t0) * 1000),
                        output=str(resp)[:200] if resp is not None else None,
                        status="success",
                    )
                except Exception:
                    pass
                resp.raise_for_status()
                data = resp.json()
                step.capture(data)
                return data
            except Exception as e:
                logger.error(f"AttendanceAPIClient error: {e}")
                raise

class SkillsAPIClient(BaseAPIClient):
    """Fetches employee skills."""
    ENDPOINT = f"{BASE_URL}/employees/skills"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=1, max=5),
           retry=retry_if_exception_type(httpx.RequestError))
    async def get_skills(self, employee_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        params = {}
        if employee_ids:
            params["employee_id"] = ",".join(employee_ids)
        async with trace_step(
            "skills_api_call", step_type="tool_call",
            decision_summary="Fetch skills for employees",
            output_fn=lambda r: f"skills keys={list(r.keys()) if isinstance(r, dict) else '?'}"
        ) as step:
            try:
                _obs_t0 = _time.time()
                resp = await self.client.get(
                    self.ENDPOINT,
                    headers=self.get_headers(),
                    params=params
                )
                try:
                    trace_tool_call(
                        tool_name='client.get',
                        latency_ms=int((_time.time() - _obs_t0) * 1000),
                        output=str(resp)[:200] if resp is not None else None,
                        status="success",
                    )
                except Exception:
                    pass
                resp.raise_for_status()
                data = resp.json()
                step.capture(data)
                return data
            except Exception as e:
                logger.error(f"SkillsAPIClient error: {e}")
                raise

class CapacityAPIClient(BaseAPIClient):
    """Fetches employee capacity."""
    ENDPOINT = f"{BASE_URL}/employees/capacity"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=1, max=5),
           retry=retry_if_exception_type(httpx.RequestError))
    async def get_capacity(self, employee_ids: Optional[List[str]] = None, date: Optional[str] = None) -> Dict[str, Any]:
        params = {}
        if employee_ids:
            params["employee_id"] = ",".join(employee_ids)
        if date:
            params["date"] = date
        async with trace_step(
            "capacity_api_call", step_type="tool_call",
            decision_summary="Fetch capacity for employees",
            output_fn=lambda r: f"capacity keys={list(r.keys()) if isinstance(r, dict) else '?'}"
        ) as step:
            try:
                _obs_t0 = _time.time()
                resp = await self.client.get(
                    self.ENDPOINT,
                    headers=self.get_headers(),
                    params=params
                )
                try:
                    trace_tool_call(
                        tool_name='client.get',
                        latency_ms=int((_time.time() - _obs_t0) * 1000),
                        output=str(resp)[:200] if resp is not None else None,
                        status="success",
                    )
                except Exception:
                    pass
                resp.raise_for_status()
                data = resp.json()
                step.capture(data)
                return data
            except Exception as e:
                logger.error(f"CapacityAPIClient error: {e}")
                raise

class TaskAPIClient(BaseAPIClient):
    """Fetches task priority, due date, and dependencies."""
    PRIORITY_ENDPOINT = f"{BASE_URL}/tasks/priority"
    DUE_DATE_ENDPOINT = f"{BASE_URL}/tasks/due-date"
    DEPENDENCIES_ENDPOINT = f"{BASE_URL}/tasks/dependencies"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=1, max=5),
           retry=retry_if_exception_type(httpx.RequestError))
    async def get_priority(self, task_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        params = {}
        if task_ids:
            params["task_id"] = ",".join(task_ids)
        async with trace_step(
            "task_priority_api_call", step_type="tool_call",
            decision_summary="Fetch task priorities",
            output_fn=lambda r: f"priority keys={list(r.keys()) if isinstance(r, dict) else '?'}"
        ) as step:
            try:
                _obs_t0 = _time.time()
                resp = await self.client.get(
                    self.PRIORITY_ENDPOINT,
                    headers=self.get_headers(),
                    params=params
                )
                try:
                    trace_tool_call(
                        tool_name='client.get',
                        latency_ms=int((_time.time() - _obs_t0) * 1000),
                        output=str(resp)[:200] if resp is not None else None,
                        status="success",
                    )
                except Exception:
                    pass
                resp.raise_for_status()
                data = resp.json()
                step.capture(data)
                return data
            except Exception as e:
                logger.error(f"TaskAPIClient.get_priority error: {e}")
                raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=1, max=5),
           retry=retry_if_exception_type(httpx.RequestError))
    async def get_due_dates(self, task_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        params = {}
        if task_ids:
            params["task_id"] = ",".join(task_ids)
        async with trace_step(
            "task_due_date_api_call", step_type="tool_call",
            decision_summary="Fetch task due dates",
            output_fn=lambda r: f"due_date keys={list(r.keys()) if isinstance(r, dict) else '?'}"
        ) as step:
            try:
                _obs_t0 = _time.time()
                resp = await self.client.get(
                    self.DUE_DATE_ENDPOINT,
                    headers=self.get_headers(),
                    params=params
                )
                try:
                    trace_tool_call(
                        tool_name='client.get',
                        latency_ms=int((_time.time() - _obs_t0) * 1000),
                        output=str(resp)[:200] if resp is not None else None,
                        status="success",
                    )
                except Exception:
                    pass
                resp.raise_for_status()
                data = resp.json()
                step.capture(data)
                return data
            except Exception as e:
                logger.error(f"TaskAPIClient.get_due_dates error: {e}")
                raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=1, max=5),
           retry=retry_if_exception_type(httpx.RequestError))
    async def get_dependencies(self, task_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        params = {}
        if task_ids:
            params["task_id"] = ",".join(task_ids)
        async with trace_step(
            "task_dependencies_api_call", step_type="tool_call",
            decision_summary="Fetch task dependencies",
            output_fn=lambda r: f"dependencies keys={list(r.keys()) if isinstance(r, dict) else '?'}"
        ) as step:
            try:
                _obs_t0 = _time.time()
                resp = await self.client.get(
                    self.DEPENDENCIES_ENDPOINT,
                    headers=self.get_headers(),
                    params=params
                )
                try:
                    trace_tool_call(
                        tool_name='client.get',
                        latency_ms=int((_time.time() - _obs_t0) * 1000),
                        output=str(resp)[:200] if resp is not None else None,
                        status="success",
                    )
                except Exception:
                    pass
                resp.raise_for_status()
                data = resp.json()
                step.capture(data)
                return data
            except Exception as e:
                logger.error(f"TaskAPIClient.get_dependencies error: {e}")
                raise

class AllocationAPIClient(BaseAPIClient):
    """Submits work allocations."""
    ENDPOINT = f"{BASE_URL}/allocations"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=1, max=5),
           retry=retry_if_exception_type(httpx.RequestError))
    async def submit_allocations(self, allocations: List[Allocation]) -> Dict[str, Any]:
        payload = {
            "allocations": [a.model_dump() for a in allocations]
        }
        async with trace_step(
            "allocation_api_call", step_type="tool_call",
            decision_summary="Submit allocations",
            output_fn=lambda r: f"allocation response keys={list(r.keys()) if isinstance(r, dict) else '?'}"
        ) as step:
            try:
                _obs_t0 = _time.time()
                resp = await self.client.post(
                    self.ENDPOINT,
                    headers=self.get_headers(),
                    json=payload
                )
                try:
                    trace_tool_call(
                        tool_name='client.post',
                        latency_ms=int((_time.time() - _obs_t0) * 1000),
                        output=str(resp)[:200] if resp is not None else None,
                        status="success",
                    )
                except Exception:
                    pass
                resp.raise_for_status()
                data = resp.json()
                step.capture(data)
                return data
            except Exception as e:
                logger.error(f"AllocationAPIClient error: {e}")
                raise

# --- Notification and Audit Logging ---
class NotificationService:
    """Sends notifications to employees and managers."""
    async def notify(self, allocations: List[Allocation]) -> NotificationResult:
        # Placeholder: In production, integrate with email/SMS/Slack/etc.
        notified = []
        failed = []
        for alloc in allocations:
            try:
                logger.info(f"Notify employee {alloc.employee_id} about task {alloc.task_id}")
                notified.append(alloc.employee_id)
            except Exception as e:
                failed.append(alloc.employee_id)
                logger.error(f"Notification failed for {alloc.employee_id}: {e}")
        return NotificationResult(success=(len(failed) == 0), notified=notified, failed=failed)

class AuditLogger:
    """Logs all allocation actions and errors."""
    def log(self, event: AllocationEvent) -> LogEntry:
        ts = event.timestamp or _time.time()
        details = mask_pii(event.model_dump())
        logger.info(f"Audit log: {details}")
        return LogEntry(timestamp=ts, event=event.event_type, details=details)

# --- Domain Layer: Rules Engine ---
class RulesEngine:
    """Evaluates business rules for eligibility, capacity, skills, and dependencies."""
    def validate_employee_eligibility(self, employees: List[Employee], attendance: Dict[str, Any], skills: Dict[str, Any], tasks: List[Task]) -> List[Employee]:
        eligible = []
        for emp in employees:
            att_status = attendance.get(emp.employee_id, {}).get("attendance_status", None)
            emp_skills = skills.get(emp.employee_id, {}).get("skills", [])
            if att_status in ("Present", "Half-day"):
                for task in tasks:
                    req_skills = task.required_skills or []
                    if all(skill in emp_skills for skill in req_skills):
                        eligible.append(emp)
                        break
        return eligible

    def adjust_employee_capacity(self, employees: List[Employee], attendance: Dict[str, Any], capacity: Dict[str, Any]) -> List[Employee]:
        adjusted = []
        for emp in employees:
            att_status = attendance.get(emp.employee_id, {}).get("attendance_status", None)
            emp_capacity = capacity.get(emp.employee_id, {}).get("capacity", 0)
            if att_status == "Present":
                emp.capacity = emp_capacity
            elif att_status == "Half-day":
                emp.capacity = emp_capacity * 0.5
            else:
                emp.capacity = 0
            adjusted.append(emp)
        return adjusted

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def enforce_task_dependencies(self, tasks: List[Task], dependencies: Dict[str, Any]) -> List[Task]:
        # Only assign tasks whose dependencies are completed or not present
        dep_map = {t.task_id: dependencies.get(t.task_id, {}).get("dependencies", []) for t in tasks}
        completed = set()
        ordered = []
        # Simple topological sort (no cycles assumed)
        def visit(tid, visited, stack):
            if tid in visited:
                return
            visited.add(tid)
            for dep in dep_map.get(tid, []):
                visit(dep, visited, stack)
            stack.append(tid)
        visited = set()
        stack = []
        for t in tasks:
            visit(t.task_id, visited, stack)
        stack.reverse()
        id_to_task = {t.task_id: t for t in tasks}
        for tid in stack:
            if tid in id_to_task:
                ordered.append(id_to_task[tid])
        return ordered

    def match_skills(self, employees: List[Employee], skills: Dict[str, Any], task: Task) -> List[Employee]:
        req_skills = set(task.required_skills or [])
        matched = []
        for emp in employees:
            emp_skills = set(skills.get(emp.employee_id, {}).get("skills", []))
            if req_skills.issubset(emp_skills):
                matched.append(emp)
        return matched

# --- Application Layer: Work Allocation Coordinator ---
class WorkAllocationCoordinator:
    """Orchestrates the allocation process."""
    def __init__(
        self,
        attendance_client: AttendanceAPIClient,
        skills_client: SkillsAPIClient,
        capacity_client: CapacityAPIClient,
        task_client: TaskAPIClient,
        allocation_client: AllocationAPIClient,
        rules_engine: RulesEngine,
        notification_service: NotificationService,
        audit_logger: AuditLogger
    ):
        self.attendance_client = attendance_client
        self.skills_client = skills_client
        self.capacity_client = capacity_client
        self.task_client = task_client
        self.allocation_client = allocation_client
        self.rules_engine = rules_engine
        self.notification_service = notification_service
        self.audit_logger = audit_logger

    async def allocate_work(self, tasks: List[Task], employees: List[Employee], date: str) -> AllocationResultModel:
        errors = []
        allocations: List[Allocation] = []
        explanation = ""
        async with trace_step(
            "fetch_external_data", step_type="tool_call",
            decision_summary="Fetch all required data for allocation",
            output_fn=lambda r: f"attendance, skills, capacity, priority, due_date, dependencies fetched"
        ) as step:
            try:
                employee_ids = [e.employee_id for e in employees]
                task_ids = [t.task_id for t in tasks]
                attendance, skills, capacity, priority, due_date, dependencies = await asyncio.gather(
                    self.attendance_client.get_attendance(employee_ids, date),
                    self.skills_client.get_skills(employee_ids),
                    self.capacity_client.get_capacity(employee_ids, date),
                    self.task_client.get_priority(task_ids),
                    self.task_client.get_due_dates(task_ids),
                    self.task_client.get_dependencies(task_ids)
                )
                step.capture({
                    "attendance": list(attendance.keys()),
                    "skills": list(skills.keys()),
                    "capacity": list(capacity.keys()),
                    "priority": list(priority.keys()),
                    "due_date": list(due_date.keys()),
                    "dependencies": list(dependencies.keys())
                })
            except Exception as e:
                logger.error(f"Error fetching external data: {e}")
                errors.append(f"Failed to fetch external data: {e}")
                self.audit_logger.log(AllocationEvent(
                    event_type="fetch_error",
                    error=str(e),
                    timestamp=_time.time()
                ))
                return AllocationResultModel(success=False, errors=errors, explanation="Failed to fetch required data.")

        async with trace_step(
            "validate_eligibility", step_type="process",
            decision_summary="Filter eligible employees",
            output_fn=lambda r: f"eligible={len(r)}"
        ) as step:
            eligible_employees = self.rules_engine.validate_employee_eligibility(employees, attendance, skills, tasks)
            if not eligible_employees:
                errors.append("No eligible employees found (attendance/skills).")
                self.audit_logger.log(AllocationEvent(
                    event_type="eligibility_error",
                    error="ERR_NO_AVAILABLE_EMPLOYEE",
                    timestamp=_time.time()
                ))
                return AllocationResultModel(success=False, errors=errors, explanation="No eligible employees found.")
            step.capture(eligible_employees)

        async with trace_step(
            "adjust_capacity", step_type="process",
            decision_summary="Adjust employee capacity",
            output_fn=lambda r: f"adjusted={len(r)}"
        ) as step:
            adjusted_employees = self.rules_engine.adjust_employee_capacity(eligible_employees, attendance, capacity)
            if not adjusted_employees:
                errors.append("No employees with sufficient capacity.")
                self.audit_logger.log(AllocationEvent(
                    event_type="capacity_error",
                    error="ERR_INSUFFICIENT_CAPACITY",
                    timestamp=_time.time()
                ))
                return AllocationResultModel(success=False, errors=errors, explanation="No employees with sufficient capacity.")
            step.capture(adjusted_employees)

        async with trace_step(
            "enforce_dependencies", step_type="process",
            decision_summary="Order tasks by dependencies",
            output_fn=lambda r: f"tasks={len(r)}"
        ) as step:
            ordered_tasks = self.rules_engine.enforce_task_dependencies(tasks, dependencies)
            step.capture(ordered_tasks)

        async with trace_step(
            "assign_tasks", step_type="process",
            decision_summary="Assign tasks to employees",
            output_fn=lambda r: f"allocations={len(r)}"
        ) as step:
            try:
                allocations = await self.assign_tasks(ordered_tasks, adjusted_employees, skills, priority, due_date)
                if not allocations:
                    errors.append("No allocations could be made.")
                    self.audit_logger.log(AllocationEvent(
                        event_type="allocation_error",
                        error="No allocations",
                        timestamp=_time.time()
                    ))
                    return AllocationResultModel(success=False, errors=errors, explanation="No allocations could be made.")
                step.capture(allocations)
            except Exception as e:
                logger.error(f"Error in assign_tasks: {e}")
                errors.append(f"Task assignment failed: {e}")
                self.audit_logger.log(AllocationEvent(
                    event_type="assignment_error",
                    error=str(e),
                    timestamp=_time.time()
                ))
                return AllocationResultModel(success=False, errors=errors, explanation="Task assignment failed.")

        async with trace_step(
            "submit_allocations", step_type="tool_call",
            decision_summary="Submit allocations to external API",
            output_fn=lambda r: f"api_response={r.get('success', False)}"
        ) as step:
            try:
                api_response = await self.allocation_client.submit_allocations(allocations)
                step.capture(api_response)
                if not api_response.get("success", True):
                    errors.append("Failed to submit allocations.")
                    self.audit_logger.log(AllocationEvent(
                        event_type="submit_error",
                        error="Failed to submit allocations",
                        allocations=allocations,
                        timestamp=_time.time()
                    ))
                    return AllocationResultModel(success=False, allocations=allocations, errors=errors, explanation="Failed to submit allocations.")
            except Exception as e:
                logger.error(f"Error submitting allocations: {e}")
                errors.append(f"Failed to submit allocations: {e}")
                self.audit_logger.log(AllocationEvent(
                    event_type="submit_error",
                    error=str(e),
                    allocations=allocations,
                    timestamp=_time.time()
                ))
                return AllocationResultModel(success=False, allocations=allocations, errors=errors, explanation="Failed to submit allocations.")

        async with trace_step(
            "notify_employees", step_type="tool_call",
            decision_summary="Notify employees about allocations",
            output_fn=lambda r: f"notified={len(r.notified)} failed={len(r.failed)}"
        ) as step:
            notification_result = await self.notification_service.notify(allocations)
            step.capture(notification_result)
            if not notification_result.success:
                errors.append(f"Notification failures: {notification_result.failed}")

        self.audit_logger.log(AllocationEvent(
            event_type="allocation_success",
            allocations=allocations,
            timestamp=_time.time()
        ))

        # LLM explanation
        async with trace_step(
            "llm_explanation", step_type="llm_call",
            decision_summary="Generate allocation explanation",
            output_fn=lambda r: f"explanation_len={len(r) if r else 0}"
        ) as step:
            explanation = await EmployeeWorkAllocationAgent.llm_explanation(
                allocations=allocations,
                employees=adjusted_employees,
                tasks=ordered_tasks
            )
            step.capture(explanation)

        return AllocationResultModel(
            success=True,
            allocations=allocations,
            errors=errors if errors else None,
            explanation=explanation
        )

    @trace_agent(agent_name='Employee Work Allocation Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def assign_tasks(
        self,
        tasks: List[Task],
        employees: List[Employee],
        skills: Dict[str, Any],
        priority: Dict[str, Any],
        due_date: Dict[str, Any]
    ) -> List[Allocation]:
        # Simple greedy assignment: assign highest priority tasks first to employees with capacity and skills
        allocations = []
        emp_capacity = {e.employee_id: e.capacity or 0 for e in employees}
        # Sort tasks by priority and due date
        def priority_value(p):
            if isinstance(p, str):
                return {"High": 1, "Medium": 2, "Low": 3}.get(p, 4)
            return 4
        task_priority_map = {tid: priority.get(tid, {}).get("priority", "Medium") for tid in [t.task_id for t in tasks]}
        task_due_map = {tid: due_date.get(tid, {}).get("due_date", "9999-12-31") for tid in [t.task_id for t in tasks]}
        sorted_tasks = sorted(
            tasks,
            key=lambda t: (priority_value(task_priority_map.get(t.task_id)), task_due_map.get(t.task_id))
        )
        for task in sorted_tasks:
            candidates = self.rules_engine.match_skills(employees, skills, task)
            candidates = [e for e in candidates if emp_capacity.get(e.employee_id, 0) > 0]
            if not candidates:
                continue
            # Assign to employee with most available capacity
            candidates.sort(key=lambda e: -emp_capacity.get(e.employee_id, 0))
            chosen = candidates[0]
            alloc_pct = min(100.0, emp_capacity[chosen.employee_id] * 12.5)  # 1 capacity unit = 12.5% (for 8h day)
            allocations.append(Allocation(
                employee_id=chosen.employee_id,
                task_id=task.task_id,
                allocation_percentage=alloc_pct
            ))
            emp_capacity[chosen.employee_id] -= alloc_pct / 12.5
        return allocations

# --- LLM Integration (Azure OpenAI) ---
class EmployeeWorkAllocationAgent:
    """Main agent class implementing IAgent interface."""
    # LLM config
    provider = "azure"
    model = "gpt-4.1"
    temperature = 0.7
    max_tokens = 2000
    system_prompt = (
        "You are the Employee Work Allocation Agent. Assign daily tasks only to employees who are present or on half-day, "
        "considering their skills, capacity, task priority, due dates, and dependencies. Exclude absent or on-leave employees "
        "and adjust capacity for half-day status. Ensure all allocations are fair, balanced, and compliant with business rules."
    )
    user_prompt_template = (
        "Please provide the list of tasks and employees for today's allocation. The agent will ensure only eligible employees are assigned work based on attendance, skills, and capacity."
    )
    few_shot_examples = [
        "Allocating tasks for June 10, 2024. Only present and half-day employees will be considered. Please provide the task list.",
        "John Doe was not assigned tasks because his attendance status is marked as 'Absent' for today."
    ]

    def __init__(self):
        self.coordinator = WorkAllocationCoordinator(
            attendance_client=AttendanceAPIClient(),
            skills_client=SkillsAPIClient(),
            capacity_client=CapacityAPIClient(),
            task_client=TaskAPIClient(),
            allocation_client=AllocationAPIClient(),
            rules_engine=RulesEngine(),
            notification_service=NotificationService(),
            audit_logger=AuditLogger()
        )

    @staticmethod
    @trace_agent(agent_name='Employee Work Allocation Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def get_llm_client():
        # Lazy import and client creation
        try:
            import openai
        except ImportError:
            raise RuntimeError("openai package not installed. Please install openai>=1.0.0")
        api_key = Config.get_azure_openai_key()
        endpoint = Config.get_azure_openai_endpoint()
        deployment = Config.get_azure_openai_deployment()
        if not api_key or not endpoint or not deployment:
            raise ValueError("Azure OpenAI configuration missing (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT)")
        client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_version="2024-02-15-preview"
        )
        return client

    @classmethod
    @trace_agent(agent_name='Employee Work Allocation Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def llm_explanation(cls, allocations: List[Allocation], employees: List[Employee], tasks: List[Task]) -> str:
        """Generate an explanation for the allocation using LLM."""
        try:
            async with trace_step(
                "llm_explanation_call", step_type="llm_call",
                decision_summary="Call LLM for allocation explanation",
                output_fn=lambda r: f"explanation_len={len(r) if r else 0}"
            ) as step:
                client = cls.get_llm_client()
                # Prepare context
                alloc_map = {}
                for alloc in allocations:
                    alloc_map.setdefault(alloc.employee_id, []).append(alloc.task_id)
                emp_map = {e.employee_id: e.name or e.employee_id for e in employees}
                task_map = {t.task_id: t.name or t.task_id for t in tasks}
                alloc_summary = []
                for eid, tids in alloc_map.items():
                    alloc_summary.append(f"{emp_map.get(eid, eid)}: {', '.join([task_map.get(tid, tid) for tid in tids])}")
                alloc_text = "\n".join(alloc_summary)
                user_prompt = (
                    f"Today's allocations:\n{alloc_text}\n"
                    "Explain why each employee was assigned their tasks, and note any employees who were not assigned due to attendance or skills."
                )
                messages = [
                    {"role": "system", "content": cls.system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                for ex in cls.few_shot_examples:
                    messages.append({"role": "user", "content": ex})
                _t0 = _time.time()
                response = await cls.get_llm_client().chat.completions.create(
                    model=cls.model,
                    messages=messages,
                    temperature=cls.temperature,
                    max_tokens=cls.max_tokens
                )
                content = response.choices[0].message.content
                try:
                    trace_model_call(
                        provider="azure",
                        model_name=cls.model,
                        prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
                        completion_tokens=getattr(response.usage, "completion_tokens", 0),
                        latency_ms=int((_time.time() - _t0) * 1000),
                        response_summary=content[:200] if content else ""
                    )
                except Exception:
                    pass
                step.capture(content)
                return redact_sensitive(content)
        except Exception as e:
            logger.error(f"LLM explanation error: {e}")
            return "Could not generate explanation due to LLM error."

    @trace_agent(agent_name='Employee Work Allocation Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def allocate(self, request: AllocationRequestModel) -> AllocationResultModel:
        """Main entry point for allocation."""
        async with trace_step(
            "main_allocate", step_type="plan",
            decision_summary="Start allocation process",
            output_fn=lambda r: f"success={r.success}"
        ) as step:
            result = await self.coordinator.allocate_work(
                tasks=request.tasks,
                employees=request.employees,
                date=request.date
            )
            step.capture(result)
            return result

# --- Presentation Layer: FastAPI App ---
app = FastAPI(
    title="Employee Work Allocation Agent",
    description="Automated agent for fair and compliant employee work allocation.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Input validation failed",
            "details": exc.errors(),
            "tips": [
                "Ensure all required fields are present and correctly formatted.",
                "Check for missing commas, quotes, or brackets in your JSON.",
                "Limit text fields to 50,000 characters."
            ]
        }
    )

@app.exception_handler(HTTPException)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "tips": [
                "Check your request and try again.",
                "Contact support if the issue persists."
            ]
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "tips": [
                "Ensure your JSON is valid.",
                "Contact support if the issue persists."
            ]
        }
    )

@app.post("/allocate", response_model=AllocationResultModel)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def allocate_work_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        logger.warning(f"Malformed JSON: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Malformed JSON in request body.",
                "tips": [
                    "Ensure your JSON is valid (check quotes, commas, brackets).",
                    "Limit text fields to 50,000 characters.",
                    "See API docs for correct schema."
                ]
            }
        )
    try:
        req_model = AllocationRequestModel(**body)
    except ValidationError as ve:
        logger.warning(f"Validation error: {ve}")
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": "Input validation failed.",
                "details": ve.errors(),
                "tips": [
                    "Ensure all required fields are present and correctly formatted.",
                    "Check for missing commas, quotes, or brackets in your JSON.",
                    "Limit text fields to 50,000 characters."
                ]
            }
        )
    agent = EmployeeWorkAllocationAgent()
    result = await agent.allocate(req_model)
    return result

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
        logger.info("Starting Employee Work Allocation Agent...")
        uvicorn.run("agent:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), reload=False)
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