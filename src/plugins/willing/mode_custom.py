from .willing_manager import BaseWillingManager


class CustomWillingManager(BaseWillingManager):
    async def async_task_starter(self) -> None:
        pass

    async def before_generate_reply_handle(self, message_id: str):
        pass

    async def after_generate_reply_handle(self, message_id: str):
        pass

    async def not_reply_handle(self, message_id: str):
        pass

    async def get_reply_probability(self, message_id: str):
        pass

    async def bombing_buffer_message_handle(self, message_id: str):
        pass

    def __init__(self):
        super().__init__()
