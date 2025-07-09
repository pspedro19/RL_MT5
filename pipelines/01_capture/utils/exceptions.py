class MT5InitializationError(Exception):
    """Raised when MetaTrader5 fails to initialize or symbol is unavailable."""

class DataRetrievalError(Exception):
    """Raised when no data could be retrieved from MetaTrader5."""

