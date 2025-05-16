import asyncio
import json
import random  # 添加 random 模块导入
import re
from datetime import datetime
from typing import Tuple, Union, Dict, Any, Set  # 引入 Set

import aiohttp
from aiohttp.client import ClientResponse
from src.common.logger import get_module_logger
from ...common.database import db
from ...config.config import global_config


import base64
from PIL import Image
import io
import os

from rich.traceback import install

install(extra_lines=3)

logger = get_module_logger("model_utils")


class PayLoadTooLargeError(Exception):
    """自定义异常类，用于处理请求体过大错误"""

    # (代码不变)
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return "请求体过大，请尝试压缩图片或减少输入内容。"


class RequestAbortException(Exception):
    """自定义异常类，用于处理请求中断异常"""

    # (代码不变)
    def __init__(self, message: str, response: ClientResponse):
        super().__init__(message)
        self.message = message
        self.response = response

    def __str__(self):
        return self.message


class PermissionDeniedException(Exception):
    """自定义异常类，用于处理访问拒绝的异常"""

    # (代码不变)
    def __init__(self, message: str, key_identifier: str = None):  # 添加 key 标识符
        super().__init__(message)
        self.message = message
        self.key_identifier = key_identifier  # 存储导致 403 的 key

    def __str__(self):
        return self.message


# 新增：用于内部标记需要切换 Key 的异常
class _SwitchKeyException(Exception):
    """内部异常，用于标记需要切换Key并且跳过标准等待时间."""

    # (代码不变)
    pass


# 常见Error Code Mapping
error_code_mapping = {
    # (代码不变)
    400: "参数不正确",
    401: "API key 错误，认证失败，请检查/config/bot_config.toml和.env中的配置是否正确哦~",  # 401 也可能是 Key 无效
    402: "账号余额不足",
    403: "需要实名,或余额不足,或Key无权限",  # 扩展 403 的含义
    404: "Not Found",
    429: "请求过于频繁，请稍后再试",
    500: "服务器内部故障",
    503: "服务器负载过高",
}


async def _safely_record(request_content: Dict[str, Any], payload: Dict[str, Any]):
    """安全地记录请求内容，隐藏敏感信息"""
    # (代码不变)
    image_base64: str = request_content.get("image_base64")
    image_format: str = request_content.get("image_format")
    is_gemini_payload = payload and isinstance(payload, dict) and "contents" in payload
    safe_payload = json.loads(json.dumps(payload)) if payload else {}

    if image_base64 and safe_payload and isinstance(safe_payload, dict):
        if "messages" in safe_payload and len(safe_payload["messages"]) > 0:
            if isinstance(safe_payload["messages"][0], dict) and "content" in safe_payload["messages"][0]:
                content = safe_payload["messages"][0]["content"]
                if (
                    isinstance(content, list)
                    and len(content) > 1
                    and isinstance(content[1], dict)
                    and "image_url" in content[1]
                ):
                    safe_payload["messages"][0]["content"][1]["image_url"]["url"] = (
                        f"data:image/{image_format.lower() if image_format else 'jpeg'};base64,"
                        f"{image_base64[:10]}...{image_base64[-10:]}"
                    )
        elif is_gemini_payload and "contents" in safe_payload and len(safe_payload["contents"]) > 0:
            if isinstance(safe_payload["contents"][0], dict) and "parts" in safe_payload["contents"][0]:
                parts = safe_payload["contents"][0]["parts"]
                for i, part in enumerate(parts):
                    if isinstance(part, dict) and "inlineData" in part:
                        safe_payload["contents"][0]["parts"][i]["inlineData"]["data"] = (
                            f"{image_base64[:10]}...{image_base64[-10:]}"
                        )
                        break

    return safe_payload


class LLMRequest:
    # (代码不变)
    MODELS_NEEDING_TRANSFORMATION = [
        "o1",
        "o1-2024-12-17",
        "o1-mini",
        "o1-mini-2024-09-12",
        "o1-preview",
        "o1-preview-2024-09-12",
        "o1-pro",
        "o1-pro-2025-03-19",
        "o3",
        "o3-2025-04-16",
        "o3-mini",
        "o3-mini-2025-01-31o4-mini",
        "o4-mini-2025-04-16",
    ]
    _abandoned_keys_runtime: Set[str] = set()

    def __init__(self, model: dict, **kwargs):
        """初始化 LLMRequest 实例"""

        self.model_key_name = f"{model['provider']}_KEY"
        self.model_name: str = model["name"]
        self.params = kwargs
        self.stream = model.get("stream", False)
        self.pri_in = model.get("pri_in", 0)
        self.pri_out = model.get("pri_out", 0)
        self.request_type = model.get("request_type", "default")

        try:
            raw_api_key_config = os.environ[self.model_key_name]
            self.base_url = os.environ[f"{model['provider']}_BASE_URL"]
            self.is_gemini = "googleapis.com" in self.base_url.lower()
            if self.is_gemini:
                logger.debug(f"模型 {self.model_name}: 检测到为 Gemini API (Base URL: {self.base_url})")
                if self.stream:
                    logger.warning(f"模型 {self.model_name}: Gemini 流式输出处理与 OpenAI 不同，暂时强制禁用流式。")
                    self.stream = False

            # 解析和过滤 API Keys (代码不变)
            parsed_keys = []
            is_list_config = False
            try:
                loaded_keys = json.loads(raw_api_key_config)
                if isinstance(loaded_keys, list):
                    parsed_keys = [str(key) for key in loaded_keys if key]
                    is_list_config = True
                elif isinstance(loaded_keys, str) and loaded_keys:
                    parsed_keys = [loaded_keys]
                else:
                    raise ValueError(f"Parsed API key for {self.model_key_name} is not a valid list or string.")
            except (json.JSONDecodeError, TypeError) as e:
                if isinstance(raw_api_key_config, list):
                    parsed_keys = [str(key) for key in raw_api_key_config if key]
                    is_list_config = True
                elif isinstance(raw_api_key_config, str) and raw_api_key_config:
                    parsed_keys = [raw_api_key_config]
                else:
                    raise ValueError(
                        f"Invalid or empty API key config for {self.model_key_name}: {raw_api_key_config}"
                    ) from e

            if not parsed_keys:
                raise ValueError(f"No valid API keys found for {self.model_key_name}.")

            abandoned_key_name = f"abandon_{self.model_key_name}"
            abandoned_keys_set = set()
            raw_abandoned_keys = os.environ.get(abandoned_key_name)

            if raw_abandoned_keys:
                try:
                    loaded_abandoned = json.loads(raw_abandoned_keys)
                    if isinstance(loaded_abandoned, list):
                        abandoned_keys_set.update(str(key) for key in loaded_abandoned if key)
                    elif isinstance(loaded_abandoned, str) and loaded_abandoned:
                        abandoned_keys_set.add(loaded_abandoned)
                    logger.info(
                        f"模型 {model['name']}: 加载了 {len(abandoned_keys_set)} 个来自配置 '{abandoned_key_name}' 的废弃 Keys。"
                    )
                except (json.JSONDecodeError, TypeError):
                    if isinstance(raw_abandoned_keys, list):
                        abandoned_keys_set.update(str(key) for key in raw_abandoned_keys if key)
                        logger.info(
                            f"模型 {model['name']}: 加载了 {len(abandoned_keys_set)} 个来自配置 '{abandoned_key_name}' (直接列表) 的废弃 Keys。"
                        )
                    elif isinstance(raw_abandoned_keys, str) and raw_abandoned_keys:
                        abandoned_keys_set.add(raw_abandoned_keys)
                        logger.info(
                            f"模型 {model['name']}: 加载了 1 个来自配置 '{abandoned_key_name}' (字符串) 的废弃 Key。"
                        )
                    else:
                        logger.warning(f"无法解析环境变量 '{abandoned_key_name}' 的内容: {raw_abandoned_keys}")

            all_abandoned_keys = abandoned_keys_set.union(LLMRequest._abandoned_keys_runtime)
            active_keys = [key for key in parsed_keys if key not in all_abandoned_keys]

            if not active_keys:
                logger.error(f"模型 {model['name']}: 所有为 '{self.model_key_name}' 配置的 Keys 都已被废弃或无效。")
                raise ValueError(
                    f"No active API keys available for {self.model_key_name} after filtering abandoned keys."
                )

            if is_list_config and len(active_keys) > 1:
                self._api_key_config = active_keys
                logger.info(
                    f"模型 {model['name']}: 初始化完成，可用 Keys: {len(self._api_key_config)} (已排除 {len(all_abandoned_keys)} 个废弃 Keys)。"
                )
            elif active_keys:
                self._api_key_config = active_keys[0]
                logger.info(
                    f"模型 {model['name']}: 初始化完成，使用单个活动 Key (已排除 {len(all_abandoned_keys)} 个废弃 Keys)。"
                )
            else:
                raise ValueError(f"Unexpected state: No active keys for {self.model_key_name}.")

            # 加载代理配置 (代码不变)
            self.proxy_url = None
            self.proxy_models_set = set()
            proxy_host = os.environ.get("PROXY_HOST")
            proxy_port = os.environ.get("PROXY_PORT")
            proxy_models_str = os.environ.get("PROXY_MODELS", "")

            if proxy_host and proxy_port:
                try:
                    int(proxy_port)
                    self.proxy_url = f"http://{proxy_host}:{proxy_port}"
                    logger.debug(f"代理已配置: {self.proxy_url}")

                    if proxy_models_str:
                        try:
                            cleaned_str = proxy_models_str.strip("'\"")
                            self.proxy_models_set = {
                                model_name.strip() for model_name in cleaned_str.split(",") if model_name.strip()
                            }
                            logger.debug(f"以下模型将使用代理: {self.proxy_models_set}")
                        except Exception as e:
                            logger.error(
                                f"解析 PROXY_MODELS ('{proxy_models_str}') 出错: {e}. 代理将不会对特定模型生效。"
                            )
                            self.proxy_models_set = set()
                except ValueError:
                    logger.error(f"无效的代理端口号: {proxy_port}。代理将不被启用。")
                    self.proxy_url = None
                    self.proxy_models_set = set()
                except Exception as e:
                    logger.error(f"加载代理配置时发生错误: {e}")
                    self.proxy_url = None
                    self.proxy_models_set = set()
            else:
                logger.info("未配置代理服务器 (PROXY_HOST 或 PROXY_PORT 未设置)。")

        except KeyError as e:
            # (代码不变)
            missing_key = str(e).strip("'")
            if missing_key == self.model_key_name:
                logger.error(f"配置错误：找不到 API Key 环境变量 '{self.model_key_name}'")
                raise ValueError(f"配置错误：找不到 API Key 环境变量 '{self.model_key_name}'") from e
            elif missing_key == model["base_url"]:
                logger.error(f"配置错误：找不到 Base URL 环境变量 '{model['base_url']}'")
                raise ValueError(f"配置错误：找不到 Base URL 环境变量 '{model['base_url']}'") from e
            else:
                logger.error(f"配置错误：找不到环境变量 - {str(e)}")
                raise ValueError(f"配置错误：找不到环境变量 - {str(e)}") from e
        except AttributeError as e:
            # (代码不变)
            logger.error(f"原始 model dict 信息：{model}")
            logger.error(f"配置错误：找不到对应的配置项 - {str(e)}")
            raise ValueError(f"配置错误：找不到对应的配置项 - {str(e)}") from e
        except ValueError as e:
            # (代码不变)
            logger.error(f"API Key 或配置初始化错误 for {self.model_key_name}: {str(e)}")
            raise e

        self._init_database()

    @staticmethod
    def _init_database():
        """初始化数据库集合"""
        # (代码不变)
        try:
            db.llm_usage.create_index([("timestamp", 1)])
            db.llm_usage.create_index([("model_name", 1)])
            db.llm_usage.create_index([("user_id", 1)])
            db.llm_usage.create_index([("request_type", 1)])
        except Exception as e:
            logger.error(f"创建数据库索引失败: {str(e)}")

    def _record_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        user_id: str = "system",
        request_type: str = None,
        endpoint: str = "/chat/completions",
    ):
        """记录模型使用情况到数据库
        Args:
            prompt_tokens: 输入token数
            completion_tokens: 输出token数
            total_tokens: 总token数
            user_id: 用户ID，默认为system
            request_type: 请求类型
            endpoint: API端点
        """
        # 如果 request_type 为 None，则使用实例变量中的值
        if request_type is None:
            request_type = self.request_type

        actual_endpoint = endpoint
        if self.is_gemini:
            if endpoint == "/embeddings":
                actual_endpoint = ":embedContent"
            else:
                actual_endpoint = ":generateContent"

        try:
            usage_data = {
                "model_name": self.model_name,
                "user_id": user_id,
                "request_type": request_type or self.request_type,
                "endpoint": actual_endpoint,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": self._calculate_cost(prompt_tokens, completion_tokens),
                "status": "success",
                "timestamp": datetime.now(),
            }
            db.llm_usage.insert_one(usage_data)
            logger.trace(
                f"Token使用情况 - 模型: {self.model_name}, "
                f"用户: {user_id}, 类型: {request_type or self.request_type}, "
                f"提示词: {prompt_tokens}, 完成: {completion_tokens}, "
                f"总计: {total_tokens}"
            )
        except Exception as e:
            logger.error(f"记录token使用情况失败: {str(e)}")

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """计算API调用成本"""
        # (代码不变)
        input_cost = (prompt_tokens / 1000000) * self.pri_in
        output_cost = (completion_tokens / 1000000) * self.pri_out
        return round(input_cost + output_cost, 6)

    async def _prepare_request(
        self,
        endpoint: str,
        prompt: str = None,
        image_base64: str = None,
        image_format: str = None,
        payload: dict = None,
        retry_policy: dict = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """配置请求参数，合并实例参数和调用时参数"""
        default_retry = {
            "max_retries": global_config.experimental.api_polling_max_retries,
            "base_wait": 10,
            "retry_codes": [429, 413, 500, 503],
            "abort_codes": [400, 401, 402, 403],
        }
        policy = {**default_retry, **(retry_policy or {})}

        _actual_endpoint = endpoint
        if self.is_gemini:
            action = endpoint.lstrip("/")
            api_url = f"{self.base_url.rstrip('/')}/{self.model_name}{action}"
            stream_mode = False
        else:
            api_url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            stream_mode = self.stream

        call_params = {k: v for k, v in kwargs.items() if k != "request_type"}
        merged_params = {**self.params, **call_params}

        if payload is None:
            payload = await self._build_payload(prompt, image_base64, image_format, merged_params)
        else:
            logger.debug("使用外部提供的 payload，忽略单次调用参数合并。")

        if not self.is_gemini and stream_mode:
            payload["stream"] = merged_params.get("stream", stream_mode)

        return {
            "policy": policy,
            "payload": payload,
            "api_url": api_url,
            "stream_mode": payload.get("stream", False),
            "image_base64": image_base64,
            "image_format": image_format,
            "prompt": prompt,
        }

    async def _execute_request(
        self,
        endpoint: str,
        prompt: str = None,
        image_base64: str = None,
        image_format: str = None,
        payload: dict = None,
        retry_policy: dict = None,
        response_handler: callable = None,
        user_id: str = "system",
        request_type: str = None,
        **kwargs: Any,
    ):
        """统一请求执行入口, 支持列表 key 切换、代理和单次调用参数覆盖"""
        final_request_type = request_type or kwargs.get("request_type") or self.request_type
        api_kwargs = {k: v for k, v in kwargs.items() if k != "request_type"}

        request_content = await self._prepare_request(
            endpoint, prompt, image_base64, image_format, payload, retry_policy, **api_kwargs
        )
        policy = request_content["policy"]
        api_url = request_content["api_url"]
        actual_payload = request_content["payload"]
        stream_mode = request_content["stream_mode"]

        use_proxy = False
        current_proxy_url = None
        if self.proxy_url and self.model_name in self.proxy_models_set:
            use_proxy = True
            current_proxy_url = self.proxy_url
            logger.debug(f"模型 {self.model_name}: 将通过代理 {current_proxy_url} 发送请求。")
        elif self.proxy_url:
            logger.debug(f"模型 {self.model_name}: 配置了代理，但此模型不在 PROXY_MODELS 列表中，将不使用代理。")
        else:
            logger.debug(f"模型 {self.model_name}: 未配置或不为此模型使用代理。")

        current_key = None
        keys_failed_429 = set()
        keys_abandoned_runtime = set()
        key_switch_limit_429 = global_config.experimental.api_polling_max_retries
        key_switch_limit_403 = global_config.experimental.api_polling_max_retries

        available_keys_pool = []
        is_key_list = isinstance(self._api_key_config, list)

        if is_key_list:
            available_keys_pool = list(self._api_key_config)
            if not available_keys_pool:
                logger.error(f"模型 {self.model_name}: 初始化后无可用活动 Keys。")
                raise ValueError(f"模型 {self.model_name}: 无可用活动 Keys。")
            random.shuffle(available_keys_pool)
            key_switch_limit_429 = min(key_switch_limit_429, len(available_keys_pool))
            key_switch_limit_403 = min(key_switch_limit_403, len(available_keys_pool))
            logger.info(
                f"模型 {self.model_name}: Key 列表模式，启用 429/403 自动切换（429上限: {key_switch_limit_429}, 403上限: {key_switch_limit_403}）。"
            )
        elif isinstance(self._api_key_config, str):
            available_keys_pool = [self._api_key_config]
            key_switch_limit_429 = 1
            key_switch_limit_403 = 1
        else:
            logger.error(f"模型 {self.model_name}: 无效的 API Key 配置类型在执行时遇到: {type(self._api_key_config)}")
            raise TypeError(f"模型 {self.model_name}: 无效的 API Key 配置类型")

        last_exception = None

        for attempt in range(policy["max_retries"]):
            if available_keys_pool:
                current_key = available_keys_pool.pop(0)
            elif current_key:
                logger.debug(
                    f"模型 {self.model_name}: 无新 Key 可用或为单 Key 模式，将使用 Key ...{current_key[-4:]} 进行重试 (第 {attempt + 1} 次尝试)"
                )
            else:
                if (
                    not self._api_key_config
                    or all(
                        k in LLMRequest._abandoned_keys_runtime
                        for k in self._api_key_config
                        if isinstance(self._api_key_config, list)
                    )
                    or (
                        isinstance(self._api_key_config, str)
                        and self._api_key_config in LLMRequest._abandoned_keys_runtime
                    )
                ):
                    final_error_msg = f"模型 {self.model_name}: 所有可用 API Keys 均因 403 错误被禁用。"
                    logger.critical(final_error_msg)
                    raise PermissionDeniedException(final_error_msg)
                else:
                    raise RuntimeError(f"模型 {self.model_name}: 无法选择 API key (第 {attempt + 1} 次尝试)")

            logger.debug(f"模型 {self.model_name}: 尝试使用 Key: ...{current_key[-4:]} (总第 {attempt + 1} 次尝试)")

            try:
                headers = await self._build_headers(current_key)
                if not self.is_gemini and stream_mode:
                    headers["Accept"] = "text/event-stream"

                async with aiohttp.ClientSession() as session:
                    post_kwargs = {"headers": headers, "json": actual_payload, "timeout": 60}
                    if use_proxy:
                        post_kwargs["proxy"] = current_proxy_url

                    async with session.post(api_url, **post_kwargs) as response:
                        if response.status == 429 and is_key_list:
                            logger.warning(f"模型 {self.model_name}: Key ...{current_key[-4:]} 遇到 429 错误。")
                            response_text = await response.text()
                            logger.debug(
                                f"模型 {self.model_name}: Key ...{current_key[-4:]} response:\n{json.dumps(json.loads(response_text), indent=2, ensure_ascii=False)}\napi_url:\n{api_url}\nheader:\n{headers}\npayload:\n{actual_payload}"
                            )
                            if current_key not in keys_failed_429:
                                keys_failed_429.add(current_key)
                                logger.info(
                                    f"  (因 429 已失败 {len(keys_failed_429)}/{key_switch_limit_429} 个不同 Key)"
                                )
                                if available_keys_pool and len(keys_failed_429) < key_switch_limit_429:
                                    logger.info("  尝试因 429 切换到下一个可用 Key...")
                                    raise _SwitchKeyException()
                                else:
                                    logger.warning("  无更多 Key 可因 429 切换或已达上限。")
                            else:
                                logger.warning(f"  Key ...{current_key[-4:]} 再次遇到 429，按标准重试流程。")

                        elif response.status == 403 and is_key_list:
                            logger.error(
                                f"模型 {self.model_name}: Key ...{current_key[-4:]} 遇到 403 (权限拒绝) 错误。"
                            )
                            if current_key not in keys_abandoned_runtime:
                                keys_abandoned_runtime.add(current_key)
                                LLMRequest._abandoned_keys_runtime.add(current_key)
                                logger.critical(
                                    f"  !! Key ...{current_key[-4:]} 已添加到运行时废弃列表。请考虑将其移至配置中的 'abandon_{self.model_key_name}' !!"
                                )
                                if current_key in available_keys_pool:
                                    available_keys_pool.remove(current_key)
                                if available_keys_pool and len(keys_abandoned_runtime) < key_switch_limit_403:
                                    logger.info("  尝试因 403 切换到下一个可用 Key...")
                                    raise _SwitchKeyException()
                                else:
                                    logger.error("  无更多 Key 可因 403 切换或已达上限。将中止请求。")
                                    await response.read()
                                    raise PermissionDeniedException(
                                        f"Key ...{current_key[-4:]} 权限被拒，且无其他可用 Key 切换。",
                                        key_identifier=current_key,
                                    )
                            else:
                                logger.error(f"  Key ...{current_key[-4:]} 再次遇到 403，这不应发生。中止请求。")
                                await response.read()
                                raise PermissionDeniedException(
                                    f"Key ...{current_key[-4:]} 重复遇到 403。", key_identifier=current_key
                                )

                        elif response.status in policy["retry_codes"] or response.status in policy["abort_codes"]:
                            await self._handle_error_response(response, attempt, policy, current_key)

                        if response.status in policy["retry_codes"] and attempt < policy["max_retries"] - 1:
                            if response.status not in [429, 403]:
                                wait_time = policy["base_wait"] * (2**attempt)
                                logger.warning(
                                    f"模型 {self.model_name}: 遇到可重试错误 {response.status}, 等待 {wait_time} 秒后重试..."
                                )
                                await asyncio.sleep(wait_time)
                            last_exception = RuntimeError(f"重试错误 {response.status}")
                            continue

                        if response.status in policy["abort_codes"] or (
                            response.status in policy["retry_codes"] and attempt >= policy["max_retries"] - 1
                        ):
                            if attempt >= policy["max_retries"] - 1 and response.status in policy["retry_codes"]:
                                logger.error(
                                    f"模型 {self.model_name}: 达到最大重试次数，最后一次尝试仍为可重试错误 {response.status}。"
                                )
                            # await self._handle_error_response(response, attempt, policy, current_key)
                            # await response.read()
                            # final_error_msg = f"请求中止或达到最大重试次数，最终状态码: {response.status}"
                            # logger.error(final_error_msg)
                            # raise RequestAbortException(final_error_msg, response)

                        response.raise_for_status()
                        result = {}
                        if not self.is_gemini and stream_mode:
                            result = await self._handle_stream_output(response)
                        else:
                            result = await response.json()

                        return (
                            response_handler(result)
                            if response_handler
                            else self._default_response_handler(result, user_id, final_request_type, endpoint)
                        )

            except _SwitchKeyException:
                last_exception = _SwitchKeyException()
                logger.debug("捕获到 _SwitchKeyException，立即进行下一次尝试。")
                continue
            except PermissionDeniedException as e:
                logger.error(f"模型 {self.model_name}: 因权限拒绝 (403) 中止请求: {e}")
                if is_key_list and not available_keys_pool and e.key_identifier:
                    logger.critical(f"  中止原因是 Key ...{e.key_identifier[-4:]} 触发 403 后已无其他 Key 可用。")
                raise e
            except aiohttp.ClientProxyConnectionError as e:
                logger.error(f"代理连接错误: {e} (代理地址: {current_proxy_url})")
                last_exception = e
                if attempt >= policy["max_retries"] - 1:
                    raise RuntimeError(f"代理连接失败达到最大重试次数: {e}") from e
                wait_time = policy["base_wait"] * (2**attempt)
                logger.warning(f"模型 {self.model_name}: 代理连接错误，等待 {wait_time} 秒后重试...")
                await asyncio.sleep(wait_time)
                continue
            except aiohttp.ClientConnectorError as e:
                logger.error(f"网络连接错误: {e} (URL: {api_url}, 代理: {current_proxy_url})")
                last_exception = e
                if attempt >= policy["max_retries"] - 1:
                    raise RuntimeError(f"网络连接失败达到最大重试次数: {e}") from e
                wait_time = policy["base_wait"] * (2**attempt)
                logger.warning(f"模型 {self.model_name}: 网络连接错误，等待 {wait_time} 秒后重试...")
                await asyncio.sleep(wait_time)
                continue
            except (PayLoadTooLargeError, RequestAbortException) as e:
                # (代码不变)
                logger.error(f"模型 {self.model_name}: 请求处理中遇到关键错误，将中止: {e}")
                raise e
            except Exception as e:
                # (代码不变)
                last_exception = e
                logger.warning(
                    f"模型 {self.model_name}: 第 {attempt + 1} 次尝试中发生非 HTTP 错误: {str(e.__class__.__name__)} - {str(e)}"
                )

                if attempt >= policy["max_retries"] - 1:
                    logger.error(
                        f"模型 {self.model_name}: 达到最大重试次数 ({policy['max_retries']})，因非 HTTP 错误失败。"
                    )
                else:
                    try:
                        temp_request_content = {
                            "policy": policy,
                            "payload": actual_payload,
                            "api_url": api_url,
                            "stream_mode": stream_mode,
                            "image_base64": image_base64,
                            "image_format": image_format,
                            "prompt": prompt,
                        }
                        handled_payload, count_delta = await self._handle_exception(
                            e, attempt, temp_request_content, merged_params=api_kwargs
                        )
                        if handled_payload:
                            actual_payload = handled_payload
                            logger.info(f"模型 {self.model_name}: 异常处理更新了 payload，将使用当前 Key 重试。")

                        wait_time = policy["base_wait"] * (2**attempt)
                        logger.warning(f"模型 {self.model_name}: 等待 {wait_time} 秒后重试...")
                        await asyncio.sleep(wait_time)
                        continue

                    except (RequestAbortException, PermissionDeniedException) as abort_exception:
                        logger.error(f"模型 {self.model_name}: 异常处理判断需要中止请求: {abort_exception}")
                        raise abort_exception
                    except RuntimeError as rt_error:
                        logger.error(f"模型 {self.model_name}: 异常处理遇到运行时错误: {rt_error}")
                        raise rt_error

        # --- 循环结束 ---
        logger.error(f"模型 {self.model_name}: 所有重试尝试 ({policy['max_retries']} 次) 均失败。")
        if last_exception:
            if isinstance(last_exception, PermissionDeniedException):
                logger.error(f"最后遇到的错误是权限拒绝: {str(last_exception)}")
                raise last_exception
            logger.error(f"最后遇到的错误: {str(last_exception.__class__.__name__)} - {str(last_exception)}")
            raise RuntimeError(
                f"模型 {self.model_name} 达到最大重试次数，API 请求失败。最后错误: {str(last_exception)}"
            ) from last_exception
        else:
            if not available_keys_pool and keys_abandoned_runtime:
                final_error_msg = f"模型 {self.model_name}: 所有可用 API Keys 均因 403 错误被禁用。"
                logger.critical(final_error_msg)
                raise PermissionDeniedException(final_error_msg)
            else:
                raise RuntimeError(f"模型 {self.model_name} 达到最大重试次数，API 请求失败，原因未知。")

    async def _handle_stream_output(self, response: ClientResponse) -> Dict[str, Any]:
        """处理 OpenAI 兼容的流式输出"""
        # (代码不变)
        flag_delta_content_finished = False
        accumulated_content = ""
        usage = None
        reasoning_content = ""
        content = ""
        tool_calls = None

        async for line_bytes in response.content:
            try:
                line = line_bytes.decode("utf-8").strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if flag_delta_content_finished:
                            chunk_usage = chunk.get("usage", None)
                            if chunk_usage:
                                usage = chunk_usage
                        else:
                            delta = chunk["choices"][0]["delta"]
                            delta_content = delta.get("content")
                            if delta_content is None:
                                delta_content = ""
                            accumulated_content += delta_content

                            if "tool_calls" in delta:
                                if tool_calls is None:
                                    tool_calls = []
                                    for tc in delta["tool_calls"]:
                                        new_tc = dict(tc)
                                        if "function" in new_tc and "arguments" not in new_tc["function"]:
                                            new_tc["function"]["arguments"] = ""
                                        tool_calls.append(new_tc)
                                else:
                                    for i, tc_delta in enumerate(delta["tool_calls"]):
                                        if (
                                            i < len(tool_calls)
                                            and "function" in tc_delta
                                            and "arguments" in tc_delta["function"]
                                        ):
                                            if "arguments" in tool_calls[i]["function"]:
                                                tool_calls[i]["function"]["arguments"] += tc_delta["function"][
                                                    "arguments"
                                                ]
                                            else:
                                                tool_calls[i]["function"]["arguments"] = tc_delta["function"][
                                                    "arguments"
                                                ]
                                        elif i >= len(tool_calls):
                                            new_tc = dict(tc_delta)
                                            if "function" in new_tc and "arguments" not in new_tc["function"]:
                                                new_tc["function"]["arguments"] = ""
                                            tool_calls.append(new_tc)

                            finish_reason = chunk["choices"][0].get("finish_reason")
                            if delta.get("reasoning_content", None):
                                reasoning_content += delta["reasoning_content"]
                            if finish_reason == "stop" or finish_reason == "tool_calls":
                                chunk_usage = chunk.get("usage", None)
                                if chunk_usage:
                                    usage = chunk_usage
                                    break
                                flag_delta_content_finished = True
                    except json.JSONDecodeError as e:
                        logger.error(f"模型 {self.model_name} 解析流式 JSON 错误: {e} - data: '{data_str}'")
                    except Exception as e:
                        logger.exception(f"模型 {self.model_name} 解析流式输出块错误: {str(e)}")
            except UnicodeDecodeError as e:
                logger.warning(f"模型 {self.model_name} 流式输出解码错误: {e} - bytes: {line_bytes[:50]}...")
            except Exception as e:
                if isinstance(e, GeneratorExit):
                    log_content = f"模型 {self.model_name} 流式输出被中断，正在清理资源..."
                else:
                    log_content = f"模型 {self.model_name} 处理流式输出时发生错误: {str(e)}"
                logger.warning(log_content)
                try:
                    await response.release()
                except Exception as cleanup_error:
                    logger.error(f"清理资源时发生错误: {cleanup_error}")
                content = accumulated_content
                break
        if not content and accumulated_content:
            content = accumulated_content
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if think_match:
            reasoning_content = think_match.group(1).strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        message = {
            "content": content,
            "reasoning_content": reasoning_content,
        }

        if tool_calls:
            message["tool_calls"] = tool_calls

        result = {
            "choices": [{"message": message}],
            "usage": usage,
        }
        return result

    async def _handle_error_response(
        self, response: ClientResponse, retry_count: int, policy: Dict[str, Any], current_key: str = None
    ) -> None:
        """处理 HTTP 错误响应 (区分 403 和其他错误)"""
        # (代码不变)
        status = response.status
        try:
            error_text = await response.text()
        except Exception as e:
            error_text = f"(无法读取响应体: {e})"

        if status == 403:
            logger.error(
                f"模型 {self.model_name}: 遇到 403 (权限拒绝) 错误。Key: ...{current_key[-4:] if current_key else 'N/A'}. "
                f"响应: {error_text[:200]}"
            )
            raise PermissionDeniedException(f"模型禁止访问 ({status})", key_identifier=current_key)

        elif status in policy["retry_codes"] and status != 429:
            if status == 413:
                logger.warning(
                    f"模型 {self.model_name}: 错误码 413 (Payload Too Large)。Key: ...{current_key[-4:] if current_key else 'N/A'}. 尝试压缩..."
                )
                raise PayLoadTooLargeError("请求体过大")
            elif status in [500, 503]:
                logger.error(
                    f"模型 {self.model_name}: 服务器内部错误或过载 ({status})。Key: ...{current_key[-4:] if current_key else 'N/A'}. "
                    f"响应: {error_text[:200]}"
                )
                return
            else:
                logger.warning(
                    f"模型 {self.model_name}: 遇到可重试错误码: {status}. Key: ...{current_key[-4:] if current_key else 'N/A'}"
                )
                return

        elif status in policy["abort_codes"]:
            logger.error(
                f"模型 {self.model_name}: 遇到需要中止的错误码: {status} - {error_code_mapping.get(status, '未知错误')}. "
                f"Key: ...{current_key[-4:] if current_key else 'N/A'}. 响应: {error_text[:200]}"
            )
            raise RequestAbortException(f"请求出现错误 {status}，中止处理", response)
        else:
            logger.error(
                f"模型 {self.model_name}: 遇到未明确处理的错误码: {status}. Key: ...{current_key[-4:] if current_key else 'N/A'}. 响应: {error_text[:200]}"
            )
            try:
                response.raise_for_status()
                raise RequestAbortException(f"未处理的错误状态码 {status}", response)
            except aiohttp.ClientResponseError as e:
                raise RequestAbortException(f"未处理的错误状态码 {status}: {e.message}", response) from e

    async def _handle_exception(
        self, exception, retry_count: int, request_content: Dict[str, Any], merged_params: Dict[str, Any] = None
    ) -> Union[Tuple[Dict[str, Any], int], Tuple[None, int]]:
        """处理非 HTTP 错误，支持使用合并后的参数重建 payload"""
        policy = request_content["policy"]
        payload = request_content["payload"]
        _wait_time = policy["base_wait"] * (2**retry_count)
        keep_request = False
        if retry_count < policy["max_retries"] - 1:
            keep_request = True

        params_for_rebuild = merged_params if merged_params is not None else payload

        if isinstance(exception, PayLoadTooLargeError):
            if keep_request:
                logger.warning("请求体过大 (PayLoadTooLargeError)，尝试压缩图片...")
                image_base64 = request_content.get("image_base64")
                if image_base64:
                    compressed_image_base64 = compress_base64_image_by_scale(image_base64)
                    if compressed_image_base64 != image_base64:
                        new_payload = await self._build_payload(
                            request_content["prompt"],
                            compressed_image_base64,
                            request_content["image_format"],
                            params_for_rebuild,
                        )
                        logger.info("图片压缩成功，将使用压缩后的图片重试。")
                        return new_payload, 0
                    else:
                        logger.warning("图片压缩未改变大小或失败。")
                else:
                    logger.warning("请求体过大但请求中不包含图片，无法压缩。")
                return None, 0
            else:
                logger.error("达到最大重试次数，请求体仍然过大。")
                raise RuntimeError("请求体过大，压缩或重试后仍然失败。") from exception

        elif isinstance(exception, (aiohttp.ClientError, asyncio.TimeoutError)):
            if keep_request:
                logger.error(f"模型 {self.model_name} 网络错误: {str(exception)}")
                return None, 0
            else:
                logger.critical(f"模型 {self.model_name} 网络错误达到最大重试次数: {str(exception)}")
                raise RuntimeError(f"网络请求失败: {str(exception)}") from exception

        elif isinstance(exception, aiohttp.ClientResponseError):
            if keep_request:
                logger.error(
                    f"模型 {self.model_name} HTTP响应错误 (未被策略覆盖): 状态码: {exception.status}, 错误: {exception.message}"
                )
                try:
                    error_text = await exception.response.text() if hasattr(exception, "response") else str(exception)
                    logger.error(f"服务器错误响应详情: {error_text[:500]}")
                except Exception as parse_err:
                    logger.warning(f"无法解析服务器错误响应内容: {str(parse_err)}")
                return None, 0
            else:
                logger.critical(
                    f"模型 {self.model_name} HTTP响应错误达到最大重试次数: 状态码: {exception.status}, 错误: {exception.message}"
                )
                current_key_placeholder = request_content.get("current_key", "******")
                handled_payload = await _safely_record(request_content, payload)
                logger.critical(
                    f"请求头: {await self._build_headers(api_key=current_key_placeholder, no_key=True)} 请求体: {handled_payload}"
                )
                raise RuntimeError(
                    f"模型 {self.model_name} API请求失败: 状态码 {exception.status}, {exception.message}"
                ) from exception

        else:
            if keep_request:
                logger.error(
                    f"模型 {self.model_name} 遇到未知错误: {str(exception.__class__.__name__)} - {str(exception)}"
                )
                return None, 0
            else:
                logger.critical(
                    f"模型 {self.model_name} 请求因未知错误失败: {str(exception.__class__.__name__)} - {str(exception)}"
                )
                current_key_placeholder = request_content.get("current_key", "******")
                handled_payload = await _safely_record(request_content, payload)
                logger.critical(
                    f"请求头: {await self._build_headers(api_key=current_key_placeholder, no_key=True)} 请求体: {handled_payload}"
                )
                raise RuntimeError(f"模型 {self.model_name} API请求失败: {str(exception)}") from exception

    async def _transform_parameters(self, merged_params: dict) -> dict:
        """根据模型名称转换合并后的参数，并移除内部参数"""
        # (代码不变)
        new_params = dict(merged_params)
        new_params.pop("request_type", None)

        if not self.is_gemini and self.model_name.lower() in self.MODELS_NEEDING_TRANSFORMATION:
            new_params.pop("temperature", None)
            if "max_tokens" in new_params:
                new_params["max_completion_tokens"] = new_params.pop("max_tokens")
        elif self.is_gemini:
            gen_config = new_params.get("generationConfig", {})
            if "temperature" in new_params:
                gen_config["temperature"] = new_params.pop("temperature")
            if "max_tokens" in new_params:
                gen_config["maxOutputTokens"] = new_params.pop("max_tokens")
            if "top_p" in new_params:
                gen_config["topP"] = new_params.pop("top_p")
            if "top_k" in new_params:
                gen_config["topK"] = new_params.pop("top_k")

            if gen_config:
                new_params["generationConfig"] = gen_config

            new_params.pop("frequency_penalty", None)
            new_params.pop("presence_penalty", None)
            new_params.pop("max_completion_tokens", None)

        return new_params

    async def _build_payload(
        self, prompt: str, image_base64: str = None, image_format: str = None, merged_params: dict = None
    ) -> dict:
        """构建请求体 (区分 Gemini 和 OpenAI)，使用合并和转换后的参数"""
        # (代码不变)
        if merged_params is None:
            merged_params = self.params

        params_copy = await self._transform_parameters(merged_params)

        if self.is_gemini:
            parts = []
            if prompt:
                parts.append({"text": prompt})
            if image_base64:
                mime_type = f"image/{image_format.lower() if image_format else 'jpeg'}"
                parts.append({"inlineData": {"mimeType": mime_type, "data": image_base64}})
            payload = {"contents": [{"parts": parts}], **params_copy}
            payload.pop("model", None)
            # --- 添加 Gemini 安全设置 ---
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
            ]
            payload["safetySettings"] = safety_settings
            logger.debug(f"模型 {self.model_name}: 已为 Gemini 函数调用请求添加 safetySettings (BLOCK_NONE)。")
            # --- 结束添加安全设置 ---

        else:
            if image_base64:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_format.lower() if image_format else 'jpeg'};base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ]
            else:
                messages = [{"role": "user", "content": prompt}]

            payload = {
                "model": self.model_name,
                "messages": messages,
                **params_copy,
            }
            if "max_tokens" not in payload and "max_completion_tokens" not in payload:
                if "max_tokens" not in params_copy and "max_completion_tokens" not in params_copy:
                    payload["max_tokens"] = global_config.model.model_max_output_length
            if "max_completion_tokens" in payload:
                payload["max_tokens"] = payload.pop("max_completion_tokens")

        return payload

    def _default_response_handler(
        self, result: dict, user_id: str = "system", request_type: str = None, endpoint: str = "/chat/completions"
    ) -> Tuple:
        """默认响应解析 (区分 Gemini 和 OpenAI)，并处理函数/工具调用"""
        content = "没有返回结果"
        reasoning_content = ""
        tool_calls = None  # OpenAI 格式
        function_call = None  # Gemini 格式
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        if self.is_gemini:
            # --- 解析 Gemini 响应 ---
            try:
                if "candidates" in result and result["candidates"]:
                    candidate = result["candidates"][0]
                    # 检查是否有 content 和 parts
                    if "content" in candidate and "parts" in candidate["content"] and candidate["content"]["parts"]:
                        # 查找 functionCall 或 text 部分
                        final_text_parts = []
                        for part in candidate["content"]["parts"]:
                            if "functionCall" in part:
                                function_call = part["functionCall"]  # 获取 Gemini 的 functionCall
                                # Gemini functionCall 通常不与 text 一起返回，这里假设只处理 functionCall
                                break  # 找到 functionCall 就停止处理 parts
                            elif "text" in part:
                                final_text_parts.append(part.get("text", ""))

                        if not function_call:  # 如果没有 functionCall，处理 text
                            raw_content = "".join(final_text_parts).strip()
                            content, reasoning = self._extract_reasoning(raw_content)
                            reasoning_content = reasoning
                        # else: function_call 已获取，content 留空或设为特定值

                    else:
                        content = "Gemini响应中缺少 content 或 parts"
                        logger.warning(f"模型 {self.model_name}: Gemini 响应格式不完整 (缺少 content/parts): {result}")

                    finish_reason = candidate.get("finishReason")
                    if finish_reason == "SAFETY":
                        logger.warning(f"模型 {self.model_name}: Gemini 响应因安全设置被阻止。")
                        content = "响应内容因安全原因被过滤。"
                    elif finish_reason == "RECITATION":
                        logger.warning(f"模型 {self.model_name}: Gemini 响应因引用限制被阻止。")
                        content = "响应内容因引用限制被过滤。"
                    elif finish_reason == "OTHER":
                        logger.warning(f"模型 {self.model_name}: Gemini 响应因未知原因停止。")
                    # finishReason == "TOOL_CODE" or "FUNCTION_CALL" 是正常情况

                usage = result.get("usageMetadata", {})
                if usage:
                    prompt_tokens = usage.get("promptTokenCount", 0)
                    completion_tokens = usage.get("candidatesTokenCount", 0)
                    total_tokens = usage.get("totalTokenCount", 0)
                    if completion_tokens == 0 and total_tokens > 0:
                        completion_tokens = total_tokens - prompt_tokens
                else:
                    logger.warning(f"模型 {self.model_name} (Gemini) 的响应中缺少 'usageMetadata' 信息。")

            except Exception as e:
                logger.error(f"解析 Gemini 响应出错: {e} - 响应: {result}")
                content = "解析 Gemini 响应时出错"

        else:
            # --- 解析 OpenAI 兼容响应 ---
            # (代码不变)
            if "choices" in result and result["choices"]:
                message = result["choices"][0].get("message", {})
                raw_content = message.get("content", "")
                content, reasoning = self._extract_reasoning(raw_content if raw_content else "")

                explicit_reasoning = message.get("model_extra", {}).get("reasoning_content", "")
                if not explicit_reasoning:
                    explicit_reasoning = message.get("reasoning_content", "")
                reasoning_content = explicit_reasoning if explicit_reasoning else reasoning

                tool_calls = message.get("tool_calls", None)  # 获取 OpenAI 的 tool_calls

                usage = result.get("usage", {})
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                else:
                    logger.warning(f"模型 {self.model_name} (OpenAI) 的响应中缺少 'usage' 信息。")
            else:
                logger.warning(f"模型 {self.model_name} (OpenAI) 的响应格式不符合预期: {result}")

        # --- 记录 Token 使用情况 ---
        # (代码不变)
        if prompt_tokens > 0 or completion_tokens > 0 or total_tokens > 0:
            self._record_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                user_id=user_id,
                request_type=request_type,
                endpoint=endpoint,
            )
        else:
            logger.warning(f"模型 {self.model_name}: 未能从响应中提取有效的 token 使用信息。")

        # --- 返回结果 (统一格式) ---
        final_tool_calls = None
        if tool_calls:  # 来自 OpenAI
            final_tool_calls = tool_calls
            logger.debug(f"检测到 OpenAI 工具调用: {final_tool_calls}")
        elif function_call:  # 来自 Gemini
            logger.debug(f"检测到 Gemini 函数调用: {function_call}")
            # 将 Gemini functionCall 转换为 OpenAI tool_calls 格式
            # 注意: Gemini 的 functionCall 没有显式的 id 和 type，需要模拟
            final_tool_calls = [
                {
                    "id": f"call_{random.randint(1000, 9999)}",  # 生成一个随机 ID
                    "type": "function",
                    "function": {
                        "name": function_call.get("name"),
                        # Gemini 的参数在 'args' 中，OpenAI 在 'arguments' (通常是 JSON 字符串)
                        # 需要将 Gemini 的 dict 参数转换为 JSON 字符串
                        "arguments": json.dumps(function_call.get("args", {})),
                    },
                }
            ]
            logger.debug(f"转换为 OpenAI tool_calls 格式: {final_tool_calls}")

        if final_tool_calls:
            # 如果有工具/函数调用，通常 content 为空或包含思考过程，这里返回转换后的调用信息
            return content, reasoning_content, final_tool_calls
        else:
            # 没有工具/函数调用，返回普通文本响应
            return content, reasoning_content

    @staticmethod
    def _extract_reasoning(content: str) -> Tuple[str, str]:
        """CoT思维链提取"""
        # (代码不变)
        if not content:
            return "", ""
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL, count=1).strip()
        if match:
            reasoning = match.group(1).strip()
        else:
            reasoning = ""
        return cleaned_content, reasoning

    async def _build_headers(self, api_key: str, no_key: bool = False) -> dict:
        """构建请求头 (区分 Gemini 和 OpenAI)"""
        # (代码不变)
        if no_key:
            if self.is_gemini:
                return {"x-goog-api-key": "**********", "Content-Type": "application/json"}
            else:
                return {"Authorization": "Bearer **********", "Content-Type": "application/json"}
        else:
            if not api_key:
                logger.error(f"尝试使用无效 (空) 的 API key 为模型 {self.model_name} 构建请求头。")
                raise ValueError("无效的 API key 提供给 _build_headers。")

            if self.is_gemini:
                return {"x-goog-api-key": api_key, "Content-Type": "application/json"}
            else:
                return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async def generate_response(self, prompt: str, user_id: str = "system", **kwargs) -> Tuple:
        """根据输入的提示生成模型的异步响应，支持覆盖参数"""
        endpoint = ":generateContent" if self.is_gemini else "/chat/completions"
        response = await self._execute_request(
            endpoint=endpoint, prompt=prompt, user_id=user_id, request_type="chat", **kwargs
        )
        if len(response) == 3:
            content, reasoning_content, tool_calls = response
            return content, reasoning_content, self.model_name, tool_calls
        else:
            content, reasoning_content = response
            return content, reasoning_content, self.model_name

    async def generate_response_for_image(
        self, prompt: str, image_base64: str, image_format: str, user_id: str = "system", **kwargs
    ) -> Tuple:
        """根据输入的提示和图片生成模型的异步响应，支持覆盖参数"""
        endpoint = ":generateContent" if self.is_gemini else "/chat/completions"
        response = await self._execute_request(
            endpoint=endpoint,
            prompt=prompt,
            image_base64=image_base64,
            image_format=image_format,
            user_id=user_id,
            request_type="vision",
            **kwargs,
        )
        # _default_response_handler 现在总是返回至少2个值
        if len(response) == 3:
            return response  # content, reasoning, tool_calls (tool_calls 可能为 None)
        elif len(response) == 2:
            content, reasoning = response
            return content, reasoning  # 对于 vision 请求，通常没有 tool_calls
        else:
            logger.error(f"来自 _default_response_handler 的意外响应格式: {response}")
            return "处理响应出错", ""

    async def generate_response_async(
        self, prompt: str, user_id: str = "system", request_type: str = "chat", **kwargs
    ) -> Union[str, Tuple]:
        """异步方式根据输入的提示生成模型的响应 (通用)，支持覆盖参数"""
        # (代码不变)
        endpoint = ":generateContent" if self.is_gemini else "/chat/completions"
        response = await self._execute_request(
            endpoint=endpoint,
            prompt=prompt,
            payload=None,
            retry_policy=None,
            response_handler=None,
            user_id=user_id,
            request_type=request_type,
            **kwargs,
        )
        return response

    # 修改：实现 Gemini Function Calling 的 Payload 构建
    async def generate_response_tool_async(
        self, prompt: str, tools: list, user_id: str = "system", **kwargs
    ) -> tuple[str, str, list | None]:
        """异步方式根据输入的提示和工具生成模型的响应，支持覆盖参数和 Gemini 函数调用"""

        endpoint = ":generateContent" if self.is_gemini else "/chat/completions"
        merged_params = {**self.params, **kwargs}
        transformed_params = await self._transform_parameters(merged_params)  # 清理 request_type 等

        payload = None

        if self.is_gemini:
            # --- 构建 Gemini Function Calling Payload ---
            logger.debug(f"为 Gemini ({self.model_name}) 构建函数调用请求。")
            # 1. 转换工具定义 (OpenAI -> Gemini)
            # OpenAI tool format: [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]
            # Gemini tool format: [{"functionDeclarations": [{"name": ..., "description": ..., "parameters": ...}]}]
            function_declarations = []
            if tools:
                for tool in tools:
                    if tool.get("type") == "function" and "function" in tool:
                        func_def = tool["function"]
                        # Gemini parameters 使用 OpenAPI Schema，与 OpenAI 基本兼容
                        function_declarations.append(
                            {
                                "name": func_def.get("name"),
                                "description": func_def.get("description", ""),  # Description is required for Gemini
                                "parameters": func_def.get(
                                    "parameters", {"type": "object", "properties": {}}
                                ),  # Ensure parameters exist
                            }
                        )
                    else:
                        logger.warning(f"跳过不支持的工具类型或格式: {tool}")

            if not function_declarations:
                logger.error("没有有效的函数声明可用于 Gemini 请求。")
                return "没有提供有效的函数定义", "", None

            gemini_tools = [{"functionDeclarations": function_declarations}]

            # 2. 构建 Gemini Payload
            # parts = [{"text": prompt}] # 初始 parts
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],  # 包含用户提示
                "tools": gemini_tools,
                # toolConfig 默认是 AUTO，可以根据需要从 kwargs 获取或硬编码
                # "toolConfig": {"functionCallingConfig": {"mode": "ANY"}}, # 例如强制调用
                **transformed_params,  # 合并其他转换后的参数 (如 generationConfig)
            }
            payload.pop("model", None)  # Gemini 不在顶层传 model
            payload.pop("messages", None)  # 移除 OpenAI 特有的 messages
            payload.pop("tool_choice", None)  # 移除 OpenAI 特有的 tool_choice

            logger.trace(f"构建的 Gemini 函数调用 Payload: {json.dumps(payload, indent=2)}")

        else:
            # --- 构建 OpenAI Tool Calling Payload ---
            logger.debug(f"为 OpenAI 兼容模型 ({self.model_name}) 构建工具调用请求。")
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                **transformed_params,
                "tools": tools,
                "tool_choice": transformed_params.get("tool_choice", "auto"),
            }
            if "max_completion_tokens" in payload:
                payload["max_tokens"] = payload.pop("max_completion_tokens")
            if "max_tokens" not in payload:
                payload["max_tokens"] = global_config.model.model_max_output_length

        # --- 执行请求 ---
        if payload is None:
            logger.error("未能构建有效的 API 请求 payload。")
            return "内部错误：无法构建请求", "", None

        response = await self._execute_request(
            endpoint=endpoint,
            payload=payload,
            prompt=prompt,  # prompt 仍然需要，用于可能的重试
            user_id=user_id,
            request_type="tool_call",
            **kwargs,  # 传递原始 kwargs 以便在重试时重新合并
        )

        # _default_response_handler 现在会处理 Gemini functionCall 并统一格式
        logger.debug(f"模型 {self.model_name} 工具/函数调用返回结果: {response}")

        if isinstance(response, tuple) and len(response) == 3:
            content, reasoning_content, final_tool_calls = response
            # final_tool_calls 已经是统一的 OpenAI 格式
            return content, reasoning_content, final_tool_calls
        elif isinstance(response, tuple) and len(response) == 2:
            content, reasoning_content = response
            logger.debug("收到普通响应，无工具/函数调用")
            return content, reasoning_content, None
        else:
            logger.error(f"收到来自 _execute_request/_default_response_handler 的意外响应格式: {response}")
            return "处理响应时出错", "", None

    async def get_embedding(self, text: str, user_id: str = "system", **kwargs) -> Union[list, None]:
        """异步方法：获取文本的embedding向量，支持覆盖参数 (Gemini Embedding 需注意模型名称)"""
        if len(text) < 1:
            logger.debug("该消息没有长度，不再发送获取embedding向量的请求")
            return None

        api_kwargs = {k: v for k, v in kwargs.items() if k != "request_type"}

        if self.is_gemini:
            endpoint = ":embedContent"
            payload = {"model": f"models/{self.model_name}", "content": {"parts": [{"text": text}]}, **api_kwargs}
            payload.pop("encoding_format", None)
            payload.pop("input", None)

        else:
            endpoint = "/embeddings"
            payload = {"model": self.model_name, "input": text, "encoding_format": "float", **api_kwargs}
            payload.pop("content", None)
            payload.pop("taskType", None)

        def embedding_handler(result):
            # (代码不变)
            embedding_value = None
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

            if self.is_gemini:
                if "embedding" in result and "value" in result["embedding"]:
                    embedding_value = result["embedding"]["value"]
                logger.warning(f"模型 {self.model_name} (Gemini Embedding): 响应中未找到明确的 token 使用信息。")
            else:
                if "data" in result and len(result["data"]) > 0:
                    embedding_value = result["data"][0].get("embedding", None)
                usage = result.get("usage", {})
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                else:
                    logger.warning(f"模型 {self.model_name} (OpenAI Embedding) 的响应中缺少 'usage' 信息。")

            self._record_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                user_id=user_id,
                request_type="embedding",
                endpoint=endpoint,
            )
            return embedding_value

        embedding = await self._execute_request(
            endpoint=endpoint,
            payload=payload,
            prompt=text,
            retry_policy={"max_retries": 2, "base_wait": 6},
            response_handler=embedding_handler,
            user_id=user_id,
            request_type="embedding",
            **api_kwargs,
        )
        return embedding


def compress_base64_image_by_scale(base64_data: str, target_size: int = 0.8 * 1024 * 1024) -> str:
    """压缩base64格式的图片到指定大小"""
    # (代码不变)
    try:
        image_data = base64.b64decode(base64_data)
        if len(image_data) <= target_size * 1.05:
            logger.info(f"图片大小 {len(image_data) / 1024:.1f}KB 已足够小，无需压缩。")
            return base64_data
        img = Image.open(io.BytesIO(image_data))
        img_format = img.format
        original_width, original_height = img.size
        scale = max(0.2, min(1.0, (target_size / len(image_data)) ** 0.5))
        new_width = max(1, int(original_width * scale))
        new_height = max(1, int(original_height * scale))
        output_buffer = io.BytesIO()
        save_format = img_format  # Default to original format

        if getattr(img, "is_animated", False) and img.n_frames > 1:
            frames = []
            durations = []
            loop = img.info.get("loop", 0)
            disposal = img.info.get("disposal", 2)
            logger.info(f"检测到 GIF 动图 ({img.n_frames} 帧)，尝试按比例压缩...")
            for frame_idx in range(img.n_frames):
                img.seek(frame_idx)
                current_duration = img.info.get("duration", 100)
                durations.append(current_duration)
                new_frame = img.convert("RGBA").copy()
                resized_frame = new_frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
                frames.append(resized_frame)
            if frames:
                frames[0].save(
                    output_buffer,
                    format="GIF",
                    save_all=True,
                    append_images=frames[1:],
                    optimize=False,
                    duration=durations,
                    loop=loop,
                    disposal=disposal,
                    transparency=img.info.get("transparency", None),
                    background=img.info.get("background", None),
                )
                save_format = "GIF"
            else:
                logger.warning("未能处理 GIF 帧。")
                return base64_data
        else:
            if img.mode in ("RGBA", "LA") or "transparency" in img.info:
                resized_img = img.convert("RGBA").resize((new_width, new_height), Image.Resampling.LANCZOS)
                save_format = "PNG"
                save_params = {"optimize": True}
            else:
                resized_img = img.convert("RGB").resize((new_width, new_height), Image.Resampling.LANCZOS)
                if img_format and img_format.upper() == "JPEG":
                    save_format = "JPEG"
                    save_params = {"quality": 85, "optimize": True}
                else:
                    save_format = "PNG"
                    save_params = {"optimize": True}
            resized_img.save(output_buffer, format=save_format, **save_params)

        compressed_data = output_buffer.getvalue()
        logger.success(
            f"压缩图片: {original_width}x{original_height} -> {new_width}x{new_height} ({img.format} -> {save_format})"
        )
        logger.info(
            f"压缩前大小: {len(image_data) / 1024:.1f}KB, 压缩后大小: {len(compressed_data) / 1024:.1f}KB (目标: {target_size / 1024:.1f}KB)"
        )
        if len(compressed_data) < len(image_data) * 0.95:
            return base64.b64encode(compressed_data).decode("utf-8")
        else:
            logger.info("压缩效果不明显或反而增大，返回原始图片。")
            return base64_data
    except Exception as e:
        logger.error(f"压缩图片失败: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return base64_data
