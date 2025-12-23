from config.logger import setup_logging
import json
import time
import jwt
import requests
import threading
from typing import Any, Dict, Optional
from core.providers.llm.base import LLMProviderBase

# official coze sdk for Python [cozepy](https://github.com/coze-dev/coze-py)
from cozepy import COZE_CN_BASE_URL
from cozepy import (
    Coze,
    TokenAuth,
    Message,
    ChatEventType,
)  # noqa
from core.providers.llm.system_prompt import get_system_prompt_for_function
from core.utils.util import check_model_key

TAG = __name__
logger = setup_logging()


class JWTOAuth:
    """Coze JWT OAuth 授权管理器
    
    用于服务端应用的 JWT Grant 授权方式，支持自动刷新 access_token。
    无需用户参与，适合商业场景长期使用。
    
    参考文档：https://www.coze.cn/open/docs/developer_guides/oauth_jwt
    """
    
    # Token 缓存（全局共享，避免频繁请求）
    _token_cache: Dict[str, Dict[str, Any]] = {}
    _cache_lock = threading.Lock()
    
    # Token 刷新阈值（在过期前 1 小时刷新）
    REFRESH_THRESHOLD_SECONDS = 3600
    
    def __init__(self, client_id: str, private_key: str, 
                 base_url: str = COZE_CN_BASE_URL,
                 token_ttl: int = 86400,
                 kid: str = None):
        """初始化 JWT OAuth 管理器
        
        Args:
            client_id: Coze OAuth 应用的 client_id
            private_key: RSA 私钥内容（PEM 格式）
            base_url: Coze API 基础 URL
            token_ttl: access_token 有效期（秒），最大 86400（24小时）
            kid: 公钥指纹 (key id)，在 Coze OAuth 应用中创建密钥后获取
        """
        self.client_id = client_id
        self.private_key = private_key
        self.base_url = base_url.rstrip('/')
        self.token_ttl = min(token_ttl, 86400) if token_ttl else 86400  # 最大 24 小时
        self.kid = kid  # 公钥指纹
        
    def _generate_jwt(self) -> str:
        """生成用于获取 access_token 的 JWT"""
        import uuid
        now = int(time.time())
        payload = {
            "iss": self.client_id,  # 应用 client_id
            "aud": "api.coze.cn",   # Coze API
            "iat": now,             # 签发时间
            "exp": now + 3600,      # JWT 有效期（1小时，用于换取 access_token）
            "jti": str(uuid.uuid4()),  # 唯一标识，使用 UUID
        }
        
        # 构建 JWT headers，包含 kid（公钥指纹）
        jwt_headers = {
            "alg": "RS256",
            "typ": "JWT",
        }
        if self.kid:
            jwt_headers["kid"] = self.kid
        
        return jwt.encode(payload, self.private_key, algorithm="RS256", headers=jwt_headers)
    
    def _request_access_token(self) -> Dict[str, Any]:
        """从 Coze 获取 access_token"""
        jwt_token = self._generate_jwt()
        
        token_url = f"{self.base_url}/api/permission/oauth2/token"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {jwt_token}"
        }
        data = {
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "duration_seconds": self.token_ttl
        }
        
        try:
            response = requests.post(token_url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # 检查是否有错误（只有错误响应才有 code 字段）
            if "code" in result and result.get("code") != 0:
                raise Exception(f"Coze OAuth 错误: {result.get('msg', '未知错误')}")
            
            # 检查是否获取到 access_token
            if "access_token" not in result:
                raise Exception(f"Coze OAuth 响应缺少 access_token: {result}")
            
            return {
                "access_token": result["access_token"],
                "expires_at": time.time() + result.get("expires_in", self.token_ttl),
                "obtained_at": time.time()
            }
        except requests.RequestException as e:
            logger.bind(tag=TAG).error(f"获取 Coze access_token 失败: {e}")
            raise
    
    def get_access_token(self) -> str:
        """获取有效的 access_token（自动刷新）
        
        Returns:
            有效的 access_token
        """
        cache_key = self.client_id
        
        with self._cache_lock:
            cached = self._token_cache.get(cache_key)
            
            # 检查缓存是否存在且未过期
            if cached:
                remaining = cached["expires_at"] - time.time()
                if remaining > self.REFRESH_THRESHOLD_SECONDS:
                    return cached["access_token"]
                else:
                    logger.bind(tag=TAG).info(
                        f"Coze JWT token 即将过期（剩余 {remaining:.0f}s），正在刷新..."
                    )
            
            # 获取新的 token
            logger.bind(tag=TAG).info("正在获取 Coze JWT access_token...")
            token_info = self._request_access_token()
            self._token_cache[cache_key] = token_info
            logger.bind(tag=TAG).info(
                f"Coze JWT access_token 获取成功，有效期 {self.token_ttl}s"
            )
            return token_info["access_token"]


class LLMProvider(LLMProviderBase):
    """Coze LLM 提供者
    
    支持两种授权模式：
    1. 个人访问令牌 (PAT)：有效期 30 天，适合个人开发
    2. JWT OAuth：可自动刷新，适合商业场景
    
    当配置了 client_id 和 private_key 时使用 JWT 模式，
    否则使用 personal_access_token 模式。
    """
    
    def __init__(self, config):
        self.bot_id = str(config.get("bot_id"))
        self.user_id = str(config.get("user_id"))
        self.parameters = self._parse_parameters(config.get("parameters"))
        self.session_conversation_map = {}  # 存储session_id和conversation_id的映射
        
        # JWT OAuth 配置
        self.client_id = config.get("client_id")
        self.private_key = config.get("private_key")
        self.kid = config.get("kid")  # 公钥指纹
        token_ttl = config.get("token_ttl")
        self.token_ttl = int(token_ttl) if token_ttl else 86400  # 默认 24 小时
        
        # 个人访问令牌配置
        self.personal_access_token = config.get("personal_access_token")
        
        # 确定授权模式
        self.use_jwt_oauth = bool(self.client_id and self.private_key)
        
        if self.use_jwt_oauth:
            logger.bind(tag=TAG).info("Coze 使用 JWT OAuth 授权模式")
            if not self.kid:
                logger.bind(tag=TAG).warning("Coze JWT OAuth 未配置 kid（公钥指纹），可能导致认证失败")
            self.jwt_oauth = JWTOAuth(
                client_id=self.client_id,
                private_key=self.private_key,
                token_ttl=self.token_ttl,
                kid=self.kid
            )
        else:
            logger.bind(tag=TAG).info("Coze 使用个人访问令牌 (PAT) 授权模式")
            model_key_msg = check_model_key("CozeLLM", self.personal_access_token)
            if model_key_msg:
                logger.bind(tag=TAG).error(model_key_msg)
            self.jwt_oauth = None

    def _get_access_token(self) -> str:
        """获取当前有效的 access_token"""
        if self.use_jwt_oauth:
            return self.jwt_oauth.get_access_token()
        return self.personal_access_token

    def _parse_parameters(self, params: Optional[Any]) -> Optional[Dict[str, Any]]:
        """解析 parameters 配置，支持 dict 或 JSON 字符串"""
        if params is None:
            return None
        if isinstance(params, dict):
            return {k: v for k, v in params.items() if v is not None}
        if isinstance(params, str):
            try:
                parsed = json.loads(params)
                if isinstance(parsed, dict):
                    return {k: v for k, v in parsed.items() if v is not None}
            except json.JSONDecodeError:
                logger.bind(tag=TAG).warning("Coze parameters JSON 解析失败")
        return None

    def response(self, session_id, dialogue, **kwargs):
        coze_api_token = self._get_access_token()
        coze_api_base = COZE_CN_BASE_URL

        last_msg = next(m for m in reversed(dialogue) if m["role"] == "user")

        coze = Coze(auth=TokenAuth(token=coze_api_token), base_url=coze_api_base)
        conversation_id = self.session_conversation_map.get(session_id)

        # 如果没有找到conversation_id，则创建新的对话
        if not conversation_id:
            conversation = coze.conversations.create(messages=[])
            conversation_id = conversation.id
            self.session_conversation_map[session_id] = conversation_id  # 更新映射

        # 构建请求参数
        stream_kwargs: Dict[str, Any] = dict(
            bot_id=self.bot_id,
            user_id=self.user_id,
            additional_messages=[
                Message.build_user_question_text(last_msg["content"]),
            ],
            conversation_id=conversation_id,
        )
        
        # 如果配置了 parameters，添加到请求中
        if self.parameters:
            stream_kwargs["parameters"] = self.parameters

        for event in coze.chat.stream(**stream_kwargs):
            if event.event == ChatEventType.CONVERSATION_MESSAGE_DELTA:
                print(event.message.content, end="", flush=True)
                yield event.message.content

    def response_with_functions(self, session_id, dialogue, functions=None):
        if len(dialogue) == 2 and functions is not None and len(functions) > 0:
            # 第一次调用llm， 取最后一条用户消息，附加tool提示词
            last_msg = dialogue[-1]["content"]
            function_str = json.dumps(functions, ensure_ascii=False)
            modify_msg = get_system_prompt_for_function(function_str) + last_msg
            dialogue[-1]["content"] = modify_msg

        # 如果最后一个是 role="tool"，附加到user上
        if len(dialogue) > 1 and dialogue[-1]["role"] == "tool":
            assistant_msg = "\ntool call result: " + dialogue[-1]["content"] + "\n\n"
            while len(dialogue) > 1:
                if dialogue[-1]["role"] == "user":
                    dialogue[-1]["content"] = assistant_msg + dialogue[-1]["content"]
                    break
                dialogue.pop()

        for token in self.response(session_id, dialogue):
            yield token, None

