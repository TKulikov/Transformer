"""API server wiring all resource handlers."""

from __future__ import annotations

from ..core.protocol import decode_message, encode_message
from ..handlers.invoice_handler import InvoiceHandler
from ..handlers.order_handler import OrderHandler
from ..handlers.product_handler import ProductHandler
from ..handlers.payment_handler import PaymentHandler
from ..handlers.shipment_handler import ShipmentHandler
from ..handlers.ticket_handler import TicketHandler
from ..handlers.session_handler import SessionHandler
from ..handlers.metric_handler import MetricHandler
from ..handlers.contract_handler import ContractHandler
from ..handlers.visit_handler import VisitHandler
from ..handlers.signal_handler import SignalHandler
from ..handlers.batch_handler import BatchHandler
from ..handlers.route_handler import RouteHandler
from ..handlers.account_handler import AccountHandler
from ..handlers.event_handler import EventHandler
from ..handlers.document_handler import DocumentHandler
from ..handlers.subscription_handler import SubscriptionHandler
from ..handlers.notification_handler import NotificationHandler


class Server:
    """Dispatches framed requests to the matching resource handler."""

    def __init__(self):
        self.routes = {}
        self.routes["invoice"] = InvoiceHandler()
        self.routes["order"] = OrderHandler()
        self.routes["product"] = ProductHandler()
        self.routes["payment"] = PaymentHandler()
        self.routes["shipment"] = ShipmentHandler()
        self.routes["ticket"] = TicketHandler()
        self.routes["session"] = SessionHandler()
        self.routes["metric"] = MetricHandler()
        self.routes["contract"] = ContractHandler()
        self.routes["visit"] = VisitHandler()
        self.routes["signal"] = SignalHandler()
        self.routes["batch"] = BatchHandler()
        self.routes["route"] = RouteHandler()
        self.routes["account"] = AccountHandler()
        self.routes["event"] = EventHandler()
        self.routes["document"] = DocumentHandler()
        self.routes["subscription"] = SubscriptionHandler()
        self.routes["notification"] = NotificationHandler()

    def handle(self, frame):
        """Decode a request, route it, and encode the reply."""
        msg_type, payload, msg_id = decode_message(frame)
        resource, _, action = msg_type.partition(".")
        handler = self.routes.get(resource)
        if handler is None:
            return encode_message("error", {"error": "unknown resource"}, msg_id)
        method = getattr(handler, action or "list")
        result = method(payload)
        return encode_message(f"{msg_type}_reply", result, msg_id)
