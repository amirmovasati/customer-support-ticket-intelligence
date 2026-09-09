# region: Priority Mapping
# Business-defined mapping from intent to base priority level.
# Rule-based (not ML-trained) since the dataset has no ground-truth priority labels.

PRIORITY_MAP = {
    # High: financial risk or customer-churn risk, needs fast handling
    "complaint": "High",
    "contact_human_agent": "High",
    "payment_issue": "High",
    "cancel_order": "High",
    "delete_account": "High",
    "get_refund": "High",
    # Medium: time-sensitive or user-blocking, but not urgent
    "track_order": "Medium",
    "track_refund": "Medium",
    "change_order": "Medium",
    "check_cancellation_fee": "Medium",
    "registration_problems": "Medium",
    "recover_password": "Medium",
    "change_shipping_address": "Medium",
    "check_invoice": "Medium",
    "get_invoice": "Medium",
    "check_refund_policy": "Medium",
    "contact_customer_service": "Medium",
    # Low: informational, no real urgency
    "create_account": "Low",
    "edit_account": "Low",
    "switch_account": "Low",
    "delivery_options": "Low",
    "delivery_period": "Low",
    "review": "Low",
    "check_payment_methods": "Low",
    "set_up_shipping_address": "Low",
    "newsletter_subscription": "Low",
    "place_order": "Low",
}

ESCALATION_FLAGS = {"W", "N"}  # offensive language, negation
PRIORITY_ORDER = ["Low", "Medium", "High"]
# endregion


# region: Priority Assignment Function
# Looks up the base priority for an intent, then escalates one level if the
# ticket's flags signal offensive language or negation.

def assign_priority(intent: str, flags: str) -> str:
    base_priority = PRIORITY_MAP.get(intent, "Medium")  # unseen intent -> safe default

    has_escalation_signal = any(flag in flags for flag in ESCALATION_FLAGS)
    if has_escalation_signal:
        current_index = PRIORITY_ORDER.index(base_priority)
        escalated_index = min(current_index + 1, len(PRIORITY_ORDER) - 1)
        return PRIORITY_ORDER[escalated_index]

    return base_priority
# endregion