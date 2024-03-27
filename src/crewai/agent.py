import os
import uuid
import json
import asyncio
import threading
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from langchain.agents.agent import RunnableAgent
from langchain.agents.tools import tool as LangChainTool
from langchain.memory import ConversationSummaryMemory
from langchain.tools.render import render_text_description
from langchain_core.agents import AgentAction
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from queue import Queue
from crewai.agents import CacheHandler, CrewAgentExecutor, CrewAgentParser, ToolsHandler
from crewai.utilities import I18N, Logger, Prompts, RPMController
from crewai.utilities.token_counter_callback import TokenCalcHandler, TokenProcess


class Agent(BaseModel):
    """Represents an agent in a system.

    Each agent has a role, a goal, a backstory, and an optional language model (llm).
    The agent can also have memory, can operate in verbose mode, and can delegate tasks to other agents.

    Attributes:
            agent_executor: An instance of the CrewAgentExecutor class.
            role: The role of the agent.
            goal: The objective of the agent.
            backstory: The backstory of the agent.
            config: Dict representation of agent configuration.
            llm: The language model that will run the agent.
            function_calling_llm: The language model that will the tool calling for this agent, it overrides the crew function_calling_llm.
            max_iter: Maximum number of iterations for an agent to execute a task.
            memory: Whether the agent should have memory or not.
            max_rpm: Maximum number of requests per minute for the agent execution to be respected.
            verbose: Whether the agent execution should be in verbose mode.
            allow_delegation: Whether the agent is allowed to delegate tasks to other agents.
            tools: Tools at agents disposal
            step_callback: Callback to be executed after each step of the agent execution.
            callbacks: A list of callback functions from the langchain library that are triggered during the agent's execution process
    """

    __hash__ = object.__hash__  # type: ignore
    _logger: Logger = PrivateAttr()
    _rpm_controller: RPMController = PrivateAttr(default=None)
    _request_within_rpm_limit: Any = PrivateAttr(default=None)
    _token_process: TokenProcess = TokenProcess()

    formatting_errors: int = 0
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: UUID4 = Field(
        default_factory=uuid.uuid4,
        frozen=True,
        description="Unique identifier for the object, not set by user.",
    )
    name: str = Field(description="Name of the agent")
    role: str = Field(description="Role of the agent")
    goal: str = Field(description="Objective of the agent")
    backstory: str = Field(description="Backstory of the agent")
    config: Optional[Dict[str, Any]] = Field(
        description="Configuration for the agent",
        default=None,
    )
    max_rpm: Optional[int] = Field(
        default=None,
        description="Maximum number of requests per minute for the agent execution to be respected.",
    )
    memory: bool = Field(
        default=False, description="Whether the agent should have memory or not"
    )
    verbose: bool = Field(
        default=False, description="Verbose mode for the Agent Execution"
    )
    allow_delegation: bool = Field(
        default=True, description="Allow delegation of tasks to agents"
    )
    tools: Optional[List[Any]] = Field(
        default_factory=list, description="Tools at agents disposal"
    )
    max_iter: Optional[int] = Field(
        default=15, description="Maximum iterations for an agent to execute a task"
    )
    agent_executor: InstanceOf[CrewAgentExecutor] = Field(
        default=None, description="An instance of the CrewAgentExecutor class."
    )
    tools_handler: InstanceOf[ToolsHandler] = Field(
        default=None, description="An instance of the ToolsHandler class."
    )
    cache_handler: InstanceOf[CacheHandler] = Field(
        default=CacheHandler(), description="An instance of the CacheHandler class."
    )
    step_callback: Optional[Any] = Field(
        default=None,
        description="Callback to be executed after each step of the agent execution.",
    )
    i18n: I18N = Field(default=I18N(), description="Internationalization settings.")
    llm: Any = Field(
        default_factory=lambda: ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4")
        ),
        description="Language model that will run the agent.",
    )
    function_calling_llm: Optional[Any] = Field(
        description="Language model that will run the agent.", default=None
    )
    callbacks: Optional[List[InstanceOf[BaseCallbackHandler]]] = Field(
        default=None, description="Callback to be executed"
    )

    _original_role: str | None = None
    _original_goal: str | None = None
    _original_backstory: str | None = None

    def __init__(__pydantic_self__, **data):
        config = data.pop("config", {})
        super().__init__(**config, **data)

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        if v:
            raise PydanticCustomError(
                "may_not_set_field", "This field is not to be set by the user.", {}
            )

    @model_validator(mode="after")
    def set_attributes_based_on_config(self) -> "Agent":
        """Set attributes based on the agent configuration."""
        if self.config:
            for key, value in self.config.items():
                setattr(self, key, value)
        return self

    @model_validator(mode="after")
    def set_private_attrs(self):
        """Set private attributes."""
        self._logger = Logger(self.verbose)
        if self.max_rpm and not self._rpm_controller:
            self._rpm_controller = RPMController(
                max_rpm=self.max_rpm, logger=self._logger
            )
        return self

    @model_validator(mode="after")
    def set_agent_executor(self) -> "Agent":
        """set agent executor is set."""
        if hasattr(self.llm, "model_name"):
            self.llm.callbacks = [
                TokenCalcHandler(self.llm.model_name, self._token_process)
            ]
        if not self.agent_executor:
            self.set_cache_handler(self.cache_handler)
        return self

    def execute_task(
        self,
        task: Any,
        context: Optional[str] = None,
        tools: Optional[List[Any]] = None,
    ) -> str:
        """Execute a task with the agent.

        Args:
            task: Task to execute.
            context: Context to execute the task in.
            tools: Tools to use for the task.

        Returns:
            Output of the agent
        """
        self.tools_handler.last_used_tool = {}

        task_prompt = task.prompt()

        if context:
            task_prompt = self.i18n.slice("task_with_context").format(
                task=task_prompt, context=context
            )

        tools = self._parse_tools(tools or self.tools)
        self.create_agent_executor(tools=tools)
        self.agent_executor.tools = tools
        self.agent_executor.task = task

        self.agent_executor.tools_description = render_text_description(tools)
        self.agent_executor.tools_names = self.__tools_names(tools)

        if self.step_callback:
            result_queue = Queue()
            _thread = threading.Thread(target=self.wrap_async_func, args=(task_prompt, result_queue))
            _thread.start()
            _thread.join()
            result = result_queue.get()
        else:
            result = self.agent_executor.invoke(
                {
                    "input": task_prompt,
                    "tool_names": self.agent_executor.tools_names,
                    "tools": self.agent_executor.tools_description,
                }
            )["output"]

        if self.max_rpm:
            self._rpm_controller.stop_rpm_counter()

        return result

    def wrap_async_func(self, args, queue):
        asyncio.run(self.stream_execute(args, queue))

    async def stream_execute(self, task_prompt, result_queue):
        result = ""
        acc = ""
        chunkId = str(uuid.uuid4())
        tool_chunkId = str(uuid.uuid4())
        first = True

        #state machine used by us to decide when to do certain custom actions based on strem events
        # state = None
        try:
            async for event in self.agent_executor.astream_events(
                {
                    "input": task_prompt,
                    "tool_names": self.agent_executor.tools_names,
                    "tools": self.agent_executor.tools_description,
                },
                version="v1",
            ):
                    kind = event["event"]
                    # print(f"{kind}:\n{event}", flush=True)
                    match kind:

                        # message chunk
                        case "on_chat_model_stream":
                            content = event['data']['chunk'].content
                            chunk = repr(content)
                            self.step_callback(content, "message", first, chunkId, datetime.now().timestamp() * 1000, "bubble", event["name"])
                            first = False
                            print(f"Text chunkId ({chunkId}): {chunk}", flush=True)
                            acc += content
                            result += chunk
                            # if acc.strip().endswith('Action Input'):
                            #     state = 'STREAMING_ACTION_INPUT' #TODO: enum

                        # praser chunk
                        case "on_parser_stream":
                            print(f"Parser chunk ({kind}): {event['data']['chunk']}", flush=True)

                        # all done
                        case "on_llm_end":
                            # print(f"{kind}:\n{event}", flush=True)
                            self.step_callback("", "terminate")

                        # tool chat message finished
                        case "on_chain_end":
                            # print(f"{kind}:\n{event}", flush=True)
                            chunkId = str(uuid.uuid4())
                            first = True
                            # state = None
                            self.step_callback(acc, "message_complete", True, chunkId, datetime.now().timestamp() * 1000, "bubble", event["name"])

                        # tool started being used
                        case "on_tool_start":
                            print(f"{kind}:\n{event}", flush=True)
                            # if state != 'RUNNING_TOOL':
                            #     tool_input_json = event.get('data', {}).get('input', '')
                            #     if len(tool_input_json) > 0:
                            #         try:
                            #             tool_input = json.loads(tool_input_json)
                            #             self.step_callback(tool_input.get('message') or tool_input, "message_complete", False, chunkId, datetime.now().timestamp() * 1000)
                            #             state = 'RUNNING_TOOL'
                            #         except:
                            #             #ignore, bad tool output
                            #             pass
                            tool_chunkId = str(uuid.uuid4()) #TODO:
                            tool_name = event.get('name').replace('_', ' ').capitalize()
                            self.step_callback(f"🛠️  Using tool: {tool_name}", "message", True, tool_chunkId, datetime.now().timestamp() * 1000, "inline", event["name"])
                            # self.step_callback("", "message", False, tool_chunkId, datetime.now().timestamp() * 1000, f"")

                        # tool finished being used
                        case "on_tool_end":
                            print(f"{kind}:\n{event}", flush=True)
                            tool_chunkId = str(uuid.uuid4()) #TODO:
                            tool_name = event.get('name').replace('_', ' ').capitalize()
                            self.step_callback(f"✅ Finished using tool: {tool_name}", "message", False, tool_chunkId, datetime.now().timestamp() * 1000, "inline", event["name"])

                        # see https://python.langchain.com/docs/expression_language/streaming#event-reference
                        case _:
                            print(f"unhandled {kind} event", flush=True)
        except:
            self.step_callback(f"⛔ An unexpected error occurred.", "message", True, tool_chunkId, datetime.now().timestamp() * 1000, "inline")
            pass
        # self.step_callback("", "terminate") # is this correct?
        result_queue.put(acc)
        # result_queue.put(result)

    def set_cache_handler(self, cache_handler: CacheHandler) -> None:
        """Set the cache handler for the agent.

        Args:
            cache_handler: An instance of the CacheHandler class.
        """
        self.cache_handler = cache_handler
        self.tools_handler = ToolsHandler(cache=self.cache_handler)
        self.create_agent_executor()

    def set_rpm_controller(self, rpm_controller: RPMController) -> None:
        """Set the rpm controller for the agent.

        Args:
            rpm_controller: An instance of the RPMController class.
        """
        if not self._rpm_controller:
            self._rpm_controller = rpm_controller
            self.create_agent_executor()

    def create_agent_executor(self, tools=None) -> None:
        """Create an agent executor for the agent.

        Returns:
            An instance of the CrewAgentExecutor class.
        """
        tools = tools or self.tools

        agent_args = {
            "input": lambda x: x["input"],
            "tools": lambda x: x["tools"],
            "tool_names": lambda x: x["tool_names"],
            "agent_scratchpad": lambda x: self.format_log_to_str(
                x["intermediate_steps"]
            ),
        }

        executor_args = {
            "llm": self.llm,
            "i18n": self.i18n,
            "tools": self._parse_tools(tools),
            "verbose": self.verbose,
            "handle_parsing_errors": True,
            "max_iterations": self.max_iter,
            "step_callback": self.step_callback,
            "tools_handler": self.tools_handler,
            "function_calling_llm": self.function_calling_llm,
            "callbacks": self.callbacks,
        }

        if self._rpm_controller:
            executor_args[
                "request_within_rpm_limit"
            ] = self._rpm_controller.check_or_wait

        if self.memory:
            summary_memory = ConversationSummaryMemory(
                llm=self.llm, input_key="input", memory_key="chat_history"
            )
            executor_args["memory"] = summary_memory
            agent_args["chat_history"] = lambda x: x["chat_history"]
            prompt = Prompts(i18n=self.i18n, tools=tools).task_execution_with_memory()
        else:
            prompt = Prompts(i18n=self.i18n, tools=tools).task_execution()

        execution_prompt = prompt.partial(
            goal=self.goal,
            role=self.role,
            backstory=self.backstory,
        )

        bind = self.llm.bind(stop=[self.i18n.slice("observation")])
        inner_agent = agent_args | execution_prompt | bind | CrewAgentParser(agent=self)
        self.agent_executor = CrewAgentExecutor(
            agent=RunnableAgent(runnable=inner_agent), **executor_args
        )

    def interpolate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Interpolate inputs into the agent description and backstory."""
        if self._original_role is None:
            self._original_role = self.role
        if self._original_goal is None:
            self._original_goal = self.goal
        if self._original_backstory is None:
            self._original_backstory = self.backstory

        if inputs:
            self.role = self._original_role.format(**inputs)
            self.goal = self._original_goal.format(**inputs)
            self.backstory = self._original_backstory.format(**inputs)

    def increment_formatting_errors(self) -> None:
        """Count the formatting errors of the agent."""
        self.formatting_errors += 1

    def format_log_to_str(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        observation_prefix: str = "Observation: ",
        llm_prefix: str = "",
    ) -> str:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
        return thoughts

    def _parse_tools(self, tools: List[Any]) -> List[LangChainTool]:
        """Parse tools to be used for the task."""
        # tentatively try to import from crewai_tools import BaseTool as CrewAITool
        tools_list = []
        try:
            from crewai_tools import BaseTool as CrewAITool

            for tool in tools:
                if isinstance(tool, CrewAITool):
                    tools_list.append(tool.to_langchain())
                else:
                    tools_list.append(tool)
        except ModuleNotFoundError:
            for tool in tools:
                tools_list.append(tool)
        return tools_list

    @staticmethod
    def __tools_names(tools) -> str:
        return ", ".join([t.name for t in tools])

    def __repr__(self):
        return f"Agent(role={self.role}, goal={self.goal}, backstory={self.backstory})"
