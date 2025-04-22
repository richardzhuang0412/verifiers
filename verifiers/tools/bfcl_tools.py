from typing import List, Dict
from huanzhi_utils import load_file
import json

INVOLVED_CLASS_TO_FUNC_DOC_PATH = {
    "GorillaFileSystem": "verifiers/tools/bfcl_tools/gorilla_file_system.json",
    "MathAPI": "verifiers/tools/bfcl_tools/math_api.json",
    "MessageAPI": "verifiers/tools/bfcl_tools/message_api.json",
    "TwitterAPI": "verifiers/tools/bfcl_tools/posting_api.json",
    "TicketAPI": "verifiers/tools/bfcl_tools/ticket_api.json",
    "TradingBot": "verifiers/tools/bfcl_tools/trading_bot.json",
    "TravelAPI": "verifiers/tools/bfcl_tools/travel_booking.json",
    "VehicleControlAPI": "verifiers/tools/bfcl_tools/vehicle_control.json",
}

def construct_tools_from_involved_classes(involved_classes: List[str]) -> str:
    tools = []
    for class_name in involved_classes:
        func_doc = load_file(INVOLVED_CLASS_TO_FUNC_DOC_PATH[class_name])
        for func in func_doc:
            func["description"] = func["description"].split("Tool description: ")[1]
        func_doc = [json.dumps(func) for func in func_doc]
        tools.extend(func_doc)
    return "\n".join(tools)

def mean(numbers: List[float]) -> Dict[str, float]:
    """
    Calculate the mean of a list of numbers.

    Args:
        numbers (List[float]): List of numbers to calculate the mean of.

    Returns:
        result (float): Mean of the numbers.
    """
    if not numbers:
        return {"error": "Cannot calculate mean of an empty list"}
    try:
        return {"result": sum(numbers) / len(numbers)}
    except TypeError:
        return {"error": "All elements in the list must be numbers"}