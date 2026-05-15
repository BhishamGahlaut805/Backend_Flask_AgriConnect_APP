from flask import Blueprint, request, jsonify
import json
import logging
from .market_price import get_agmarknet_data  # adjust import based on your structure

logger = logging.getLogger(__name__)

# Create blueprint
market_prices_bp = Blueprint("market_prices", __name__)

@market_prices_bp.route("/market-prices", methods=["POST"])
def market_prices():
    """
    POST endpoint to fetch market price data.
    Expected JSON body:
    {
        "commodity": "Rice",
        "state": "Tamil Nadu",
        "market": "Chennai"
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        commodity = (data.get("commodity") or "").strip()
        state = (data.get("state") or "").strip()
        market = (data.get("market") or "").strip()

        # Validate required fields
        if not commodity or not state or not market:
            return jsonify({
                "error": "Missing required fields: commodity, state, market",
                "usage": {
                    "commodity": "string (e.g., 'Rice')",
                    "state": "string (e.g., 'Tamil Nadu')",
                    "market": "string (e.g., 'Chennai')"
                }
            }), 400

        logger.info(f"Processing market prices for commodity={commodity}, state={state}, market={market}")

        # Fetch data
        result = get_agmarknet_data(state, commodity, market)

        # Handle if scraper returns an error dict
        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 500

        return jsonify({
            "commodity": commodity,
            "state": state,
            "market": market,
            "data": result,
            "source": "agmarknet/scraper"
        })

    except Exception as e:
        logger.exception("Error processing market prices request")
        return jsonify({"error": str(e)}), 500
