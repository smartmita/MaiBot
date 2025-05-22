from src.chat.models.utils_model import LLMRequest
from src.config.config import global_config
import json
from src.common.logger_manager import get_logger
from src.tools.tool_can_use import get_all_tool_definitions, get_tool_instance
from typing import Optional
import traceback
from src.chat.person_info.relationship_manager import relationship_manager
from src.chat.utils.utils import parse_text_timestamps
from src.chat.message_receive.chat_stream import ChatStream
from src.chat.heart_flow.observation.chatting_observation import ChattingObservation
import inspect

logger = get_logger("tool_use")


class ToolUser:
    def __init__(self):
        self.llm_model_tool = LLMRequest(
            model=global_config.model.tool_use, temperature=0.2, max_tokens=1000, request_type="tool_use"
        )

    @staticmethod
    async def _build_tool_prompt(
        message_txt: str, chat_stream: ChatStream = None, observation: ChattingObservation = None
    ):
        """构建工具使用的提示词

        Args:
            message_txt: 用户消息文本
            subheartflow: 子心流对象

        Returns:
            str: 构建好的提示词
        """

        if observation:
            mid_memory_info = observation.mid_memory_info
            # print(f"intol111111111111111111111111111111111222222222222mid_memory_info：{mid_memory_info}")

        # 这些信息应该从调用者传入，而不是从self获取
        bot_name = global_config.bot.nickname
        prompt = ""
        prompt += mid_memory_info
        prompt += "你正在思考如何回复群里的消息。\n"
        prompt += "之前群里进行了如下讨论:\n"
        prompt += message_txt
        # prompt += f"你注意到{sender_name}刚刚说：{message_txt}\n"
        prompt += f"注意你就是{bot_name}，{bot_name}是你的名字。根据之前的聊天记录补充问题信息，搜索时避开你的名字。\n"
        # prompt += "必须调用 'lpmm_get_knowledge' 工具来获取知识。\n"
        prompt += "你现在需要对群里的聊天内容进行回复，请你思考应该使用什么工具，然后选择工具来对消息和你的回复进行处理，你是否需要额外的信息，比如回忆或者搜寻已有的知识，改变关系和情感，或者了解你现在正在做什么。"

        # prompt = await relationship_manager.convert_all_person_sign_to_person_name(prompt) <- ?
        prompt = parse_text_timestamps(prompt, mode="lite")

        return prompt

    @staticmethod
    def _define_tools():
        """获取所有已注册工具的定义

        Returns:
            list: 工具定义列表
        """
        return get_all_tool_definitions()

    @staticmethod
    async def _execute_tool_call(tool_call, chat_stream: Optional[ChatStream] = None):
        """执行特定的工具调用

        Args:
            tool_call: 工具调用对象
            message_txt: 原始消息文本

        Returns:
            dict: 工具调用结果
        """
        try:
            function_name = tool_call["function"]["name"]
            # 确保 function_args 是字典
            raw_function_args = tool_call["function"]["arguments"]
            if isinstance(raw_function_args, str):
                function_args = json.loads(raw_function_args)
            elif isinstance(raw_function_args, dict):
                function_args = raw_function_args
            else:
                logger.error(f"工具参数格式未知: {type(raw_function_args)}")
                return None


            tool_instance = get_tool_instance(function_name)
            if not tool_instance:
                logger.warning(f"未知工具名称: {function_name}")
                return None

            tool_execute_params = inspect.signature(tool_instance.execute).parameters
            if 'chat_stream' in tool_execute_params:
                # 特别注意：这里传递的是 chat_stream 对象本身
                result = await tool_instance.execute(function_args, chat_stream=chat_stream)
            else:
                result = await tool_instance.execute(function_args) 

            if result: # 确保 result 是一个字典
                tool_type = function_name 
                return {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": function_name,
                    "type": result.get("type", tool_type), # 优先使用工具返回的type
                    "content": result.get("content", "工具执行完成但未返回主要内容。"),
                    "id": result.get("id", function_args.get("user_id", function_name)) 
                }
            return None
        except json.JSONDecodeError as json_err:
            logger.error(f"解析工具参数 JSON 失败: {json_err}. 原始参数: {tool_call['function']['arguments']}")
            return None
        except Exception as e:
            logger.error(f"执行工具调用 {tool_call.get('function',{}).get('name','UnknownTool')} 时发生错误: {str(e)}", exc_info=True)
            return None

    async def use_tool(self, message_txt: str, chat_stream: ChatStream = None, observation: ChattingObservation = None):
        """使用工具辅助思考，判断是否需要额外信息

        Args:
            message_txt: 用户消息文本
            chat_stream: 聊天流对象
            observation: 观察对象（可选）

        Returns:
            dict: 工具使用结果，包含结构化的信息
        """
        try:
            # 构建提示词
            prompt = await self._build_tool_prompt(
                message_txt=message_txt,
                chat_stream=chat_stream,
                observation=observation,
            )

            # 定义可用工具
            tools = self._define_tools()
            logger.trace(f"工具定义: {tools}")

            # 使用llm_model_tool发送带工具定义的请求
            payload = {
                "model": self.llm_model_tool.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "tools": tools,
                "temperature": 0.2,
            }

            logger.trace(f"发送工具调用请求，模型: {self.llm_model_tool.model_name}")
            # 发送请求获取模型是否需要调用工具
            response = await self.llm_model_tool._execute_request(
                endpoint="/chat/completions", payload=payload, prompt=prompt
            )

            # 根据返回值数量判断是否有工具调用
            if len(response) == 3:
                content, reasoning_content, tool_calls = response
                # logger.info(f"工具思考: {tool_calls}")
                # logger.debug(f"工具思考: {content}")

                # 检查响应中工具调用是否有效
                if not tool_calls:
                    logger.debug("模型返回了空的tool_calls列表")
                    return {"used_tools": False}

                tool_calls_str = ""
                for tool_call in tool_calls:
                    tool_calls_str += f"{tool_call['function']['name']}\n"
                logger.info(
                    f"根据:\n{prompt}\n\n内容：{content}\n\n模型请求调用{len(tool_calls)}个工具: {tool_calls_str}"
                )
                tool_results = []
                structured_info = {}  # 动态生成键

                # 执行所有工具调用
                for tool_call in tool_calls:
                # 将 chat_stream 传递给 _execute_tool_call
                    result = await self._execute_tool_call(tool_call, chat_stream=chat_stream) 
                    if result:
                        tool_results.append(result)
                        # 使用工具名称作为键
                        tool_name = result["name"]
                        if tool_name not in structured_info:
                            structured_info[tool_name] = []
                        structured_info[tool_name].append({"name": result["name"], "content": result["content"]})

                # 如果有工具结果，返回结构化的信息
                if structured_info:
                    logger.debug(f"工具调用收集到结构化信息: {json.dumps(structured_info, ensure_ascii=False)}")
                    return {"used_tools": True, "structured_info": structured_info}
            else:
                # 没有工具调用
                content, reasoning_content = response
                logger.debug("模型没有请求调用任何工具")

            # 如果没有工具调用或处理失败，直接返回原始思考
            return {
                "used_tools": False,
            }

        except Exception as e:
            logger.error(f"工具调用过程中出错: {str(e)}")
            logger.error(f"工具调用过程中出错: {traceback.format_exc()}")
            return {
                "used_tools": False,
                "error": str(e),
            }
