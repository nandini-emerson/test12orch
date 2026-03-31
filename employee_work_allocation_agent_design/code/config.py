
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables from .env file if present
load_dotenv()

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class Config:
    # --- API Base URL ---
    BASE_URL = "https://workforce.example.com/api/v1"

    # --- API Endpoints ---
    ENDPOINTS = {
        "attendance": f"{BASE_URL}/attendance/status",
        "skills": f"{BASE_URL}/employees/skills",
        "capacity": f"{BASE_URL}/employees/capacity",
        "task_priority": f"{BASE_URL}/tasks/priority",
        "task_due_date": f"{BASE_URL}/tasks/due-date",
        "task_dependencies": f"{BASE_URL}/tasks/dependencies",
        "allocations": f"{BASE_URL}/allocations"
    }

    # --- Required Environment Variables ---
    REQUIRED_ENV_VARS = [
        "WORKFORCE_API_OAUTH_TOKEN",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT"
    ]

    # --- LLM Configuration ---
    LLM_CONFIG = {
        "provider": "azure",
        "model": "gpt-4.1",
        "temperature": 0.7,
        "max_tokens": 2000,
        "system_prompt": (
            "You are the Employee Work Allocation Agent. Assign daily tasks only to employees who are present or on half-day, "
            "considering their skills, capacity, task priority, due dates, and dependencies. Exclude absent or on-leave employees "
            "and adjust capacity for half-day status. Ensure all allocations are fair, balanced, and compliant with business rules."
        ),
        "user_prompt_template": (
            "Please provide the list of tasks and employees for today's allocation. The agent will ensure only eligible employees are assigned work based on attendance, skills, and capacity."
        ),
        "few_shot_examples": [
            "Allocating tasks for June 10, 2024. Only present and half-day employees will be considered. Please provide the task list.",
            "John Doe was not assigned tasks because his attendance status is marked as 'Absent' for today."
        ]
    }

    # --- Domain-Specific Settings ---
    DOMAIN = "general"
    AGENT_NAME = "Employee Work Allocation Agent"
    ALLOCATION_RULES = {
        "attendance_status": ["Present", "Half-day"],
        "capacity_adjustment": {"Present": 1.0, "Half-day": 0.5},
        "excluded_status": ["Absent", "On Leave"]
    }

    # --- API Key Management ---
    @classmethod
    def get_api_key(cls, key_name: str) -> str:
        value = os.getenv(key_name)
        if not value:
            raise ConfigError(f"Missing required API key: {key_name}")
        return value

    @classmethod
    def get_oauth_token(cls) -> str:
        return cls.get_api_key("WORKFORCE_API_OAUTH_TOKEN")

    @classmethod
    def get_azure_openai_key(cls) -> str:
        return cls.get_api_key("AZURE_OPENAI_API_KEY")

    @classmethod
    def get_azure_openai_endpoint(cls) -> str:
        return cls.get_api_key("AZURE_OPENAI_ENDPOINT")

    @classmethod
    def get_azure_openai_deployment(cls) -> str:
        return cls.get_api_key("AZURE_OPENAI_DEPLOYMENT")

    # --- Validation and Error Handling ---
    @classmethod
    def validate(cls):
        missing = []
        for var in cls.REQUIRED_ENV_VARS:
            if not os.getenv(var):
                missing.append(var)
        if missing:
            raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")

    # --- Default Values and Fallbacks ---
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        config = cls.LLM_CONFIG.copy()
        # Allow override via env vars if needed
        config["model"] = os.getenv("AZURE_OPENAI_MODEL", config["model"])
        config["temperature"] = float(os.getenv("LLM_TEMPERATURE", config["temperature"]))
        config["max_tokens"] = int(os.getenv("LLM_MAX_TOKENS", config["max_tokens"]))
        return config

    @classmethod
    def get_headers(cls) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {cls.get_oauth_token()}",
            "Content-Type": "application/json"
        }

    @classmethod
    def get_endpoint(cls, name: str) -> str:
        if name not in cls.ENDPOINTS:
            raise ConfigError(f"Unknown endpoint requested: {name}")
        return cls.ENDPOINTS[name]

    @classmethod
    def get_domain(cls) -> str:
        return cls.DOMAIN

    @classmethod
    def get_agent_name(cls) -> str:
        return cls.AGENT_NAME

# --- Validate configuration on import ---
try:
    Config.validate()
except ConfigError as e:
    # comment: print(f"Configuration error: {e}")
    raise

# comment: # Usage example:
# comment: # headers = Config.get_headers()
# comment: # llm_config = Config.get_llm_config()
# comment: # endpoint = Config.get_endpoint("attendance")
