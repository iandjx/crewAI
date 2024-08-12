
class AgentExecutionStoppedException(Exception):
    """
    Exception raised when agent execution is stopped. We use this to effect chain termination which
    may not be the best method but will do until there's a proper langchain way to do it.
    """
    def __init__(self, message: str = None):
        self.message = message or "Agent execution terminated by user"
        super().__init__(self.message)
