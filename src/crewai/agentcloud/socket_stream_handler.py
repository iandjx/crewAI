import uuid
from datetime import datetime

from langchain_core.agents import AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict, List, Optional, Union, Callable
from uuid import UUID

from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult

from crewai.agentcloud.socket_io import AgentCloudSocketIO


class SocketStreamHandler(BaseCallbackHandler):
    def __init__(self, socket_io: AgentCloudSocketIO, agent_name: str, task_name: str, tools_names: str,
                 stream_only_final_output: bool = False):
        self.socket_io = socket_io
        self.agent_name = agent_name
        self.chunkId = str(uuid.uuid4())
        self.task_chunkId = str(uuid.uuid4())
        self.first = True
        self.socket_io.send_to_socket(
            text=f"""**Running task**: {task_name} **Available tools**: {tools_names}""",
            event="message",
            first=self.first,
            chunk_id=self.task_chunkId,
            timestamp=datetime.now().timestamp() * 1000,
            display_type="inline"
        )
        self.stream_only_final_output = stream_only_final_output

    def on_chain_end(
            self,
            outputs: Dict[str, Any],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> None:
        self.chunkId = str(uuid.uuid4())
        self.first = True

    def on_llm_end(
            self,
            response: LLMResult,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> None:
        self.socket_io.send_to_socket(
            text="",
            event="terminate"
        )
        self.first = True

    def on_llm_new_token(
            self,
            token: str,
            *,
            chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> None:
        if token and not (self.stream_only_final_output and any('seq:step' in x for x in tags)):
            self.socket_io.send_to_socket(
                text=token,
                event="message",
                first=self.first,
                chunk_id=self.chunkId,
                timestamp=datetime.now().timestamp() * 1000,
                display_type="bubble",
                author_name=self.agent_name
            )
            self.first = False

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        self.socket_io.send_to_socket(
            text=finish.return_values["output"],
            event="message",
            first=True,
            chunk_id=self.chunkId,
            timestamp=datetime.now().timestamp() * 1000,
            display_type="bubble",
            author_name=self.agent_name
        )
