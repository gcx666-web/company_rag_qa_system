from typing  import List,Dict,Any

import openai

class LLM:
    def __init__(self, 
                api_key: str,
                model: str = "deepseek-ai/DeepSeek-V4.1-Flash",
                base_url: str = "https://api.deepseek.ai/v1/",
                ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.client=openai.OpenAI(api_key=api_key, base_url=base_url)

    def generate(self,message: List[Dict[str,str]],temperature: float = 0.7,max_tokens: int = 2000) -> Dict[str,Any]:
        """生成内容

        Args:
            message (List[Dict[str,str]]): 聊天内容
            temperature (float, optional): _description_. Defaults to 0.7.
            max_tokens (int, optional): _description_. Defaults to 2000.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            'content': response.choices[0].message.content,
            'finish_reason': response.choices[0].finish_reason,
            'token_usage': response.usage
        }