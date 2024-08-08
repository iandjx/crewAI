import uuid
from datetime import datetime
from typing import Any, Optional, Union, Callable

from ..agentcloud.socket_io import AgentCloudSocketIO
from ..tools.cache_tools import CacheTools
from ..tools.tool_calling import InstructorToolCalling, ToolCalling
from .cache.cache_handler import CacheHandler


class ToolsHandler:
    """Callback handler for tool usage."""

    last_used_tool: ToolCalling = {}  # type: ignore # BUG?: Incompatible types in assignment (expression has type "Dict[...]", variable has type "ToolCalling")
    cache: Optional[CacheHandler]
    send_to_socket: Callable

    def __init__(self, socket_io: AgentCloudSocketIO, cache: Optional[CacheHandler] = None):
        """Initialize the callback handler."""
        self.cache = cache
        self.last_used_tool = {}  # type: ignore # BUG?: same as above
        self.socket_io = socket_io
        self.tool_chunkId = None

    def on_tool_use(
            self,
            calling: Union[ToolCalling, InstructorToolCalling],
            output: str,
            should_cache: bool = True,
    ) -> Any:
        """Run when tool ends running."""
        self.last_used_tool = calling  # type: ignore # BUG?: Incompatible types in assignment (expression has type "Union[ToolCalling, InstructorToolCalling]", variable has type "ToolCalling")
        if self.cache and should_cache and calling.tool_name != CacheTools().name:
            self.cache.add(
                tool=calling.tool_name,
                input=calling.arguments,
                output=output,
            )

    def on_tool_start(self, tool_name: str):
        self.tool_chunkId = str(uuid.uuid4())
        self.socket_io.send_to_socket(
            text=f"Using tool: {tool_name.capitalize()}",
            event="message",
            first=True,
            chunk_id=self.tool_chunkId,
            timestamp=datetime.now().timestamp() * 1000,
            display_type="inline"
        )

    def on_tool_end(self, tool_name: str):
        self.socket_io.send_to_socket(
            text=f"Finished using tool: {tool_name.capitalize()}",
            event="message",
            first=True,
            chunk_id=self.tool_chunkId,
            timestamp=datetime.now().timestamp() * 1000,
            display_type="inline",
            overwrite=True
        )

    def on_tool_error(self, error_msg: str):
        self.socket_io.send_to_socket(
            text=f"""Tool usage failed:
```
{error_msg}
```
""",
            event="message",
            first=True,
            chunk_id=self.tool_chunkId,
            timestamp=datetime.now().timestamp() * 1000,
            display_type="bubble",
            overwrite=True
        )
