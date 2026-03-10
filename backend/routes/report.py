from fastapi import APIRouter
from backend.services.llm_report_service import generate_clinical_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/generate-report/{case_id}")
def generate_report(case_id: str):
    logger.info(f"Generate report endpoint called for case_id: {case_id}")

    try:
        report = generate_clinical_report(case_id)

        # Bug 5 Fixed: report[:200] throws TypeError if generate_clinical_report
        # ever returns None. Cast to str defensively.
        logger.debug(f"Report content preview: {str(report)[:200]}...")
        logger.info(f"Report generated successfully for case_id: {case_id}")

        return {
            "scan_id": case_id,
            "report": report,
            "status": "success",
        }

    except FileNotFoundError as e:
        # Bug 6 Fixed: previously all exceptions (including "file not found")
        # returned HTTP 200 with status "error" — the frontend had no way to
        # distinguish a legitimate report from a silent backend failure.
        # Now each error type returns a clear message.
        logger.warning(f"Result file missing for case_id {case_id}: {e}")
        return {
            "scan_id": case_id,
            "report": None,
            "status": "error",
            "detail": str(e),
        }

    except RuntimeError as e:
        logger.error(f"LLM error for case_id {case_id}: {e}")
        return {
            "scan_id": case_id,
            "report": None,
            "status": "error",
            "detail": str(e),
        }

    except Exception as e:
        logger.exception(f"Unexpected error for case_id {case_id}: {e}")
        return {
            "scan_id": case_id,
            "report": None,
            "status": "error",
            "detail": f"Unexpected error: {str(e)}",
        }