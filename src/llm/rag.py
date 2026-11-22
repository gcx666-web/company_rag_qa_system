import re
from typing import Any,List,Dict

from click import prompt
from numpy.random import f

from src.llm.llm import LLM


#格式化上下文函数
def format_context(result:List[Dict[str,Any]],max_token:int=2000) -> str:
    '''
     格式化上下文
     参数：
      result:搜索结果
      返回：
      格式化后的上下文
    '''
    seen=set()#存储已经处理的文件名,去重
    context=[]#存储格式化后的上下文
    total=0
    for item in result:
        content=item.get("content","").strip() #获取内容
        if not content or content in seen: #如果内容为空或者已经处理过了，则跳过
            continue
        #长度
        next_len=total+len(content)+(2 if len(context)>0 else 0)
        if next_len>max_token: #如果长度超过最大限制，则跳过
            continue
        seen.add(content)#添加到已处理集合中
        context.append(content)#添加内容
        total=next_len
    return "\n\n".join(context)
        
    
#构建提示词函数
def build_prompt(question: str, context: str) -> str:
    """
    构建提示词

    Args:
        question (str): 问题
        context (str): 上下文

    Returns:
        str: 提示词
    """
    #提示词
    instruction="你是一个知识库问答助手，请根据提供的上下文回答问题。如果上下文缺少相关信息，请返回\"我不知道\"。"
    return (
        f"{instruction}\n\n"
        f"【问题】：\n{question}\n\n"
        f"【上下文】：\n{context}\n\n"
        f"【答案】：\n"
)  
        
    


def generate_answer(llm:LLM,question: str, result:List[Dict[str,Any]],temperature:float=0.7,max_token:int=2000) -> Dict[str,Any]:
    """
    生成答案

    Args:
        llm (LLM): LLM模型
        question (str): 问题
        result (List[Dict[str,Any]]): 搜索结果, 格式为[{"id": "xxx", "content": "xxx"}, ...]
        _description_

    Returns:
        Dict[str,Any]: _description_
        {
        "answer": response.get('content','').strip(),
        "question": question,
        "context": context,
        "tokens_used": response.get('token_usage',{}),
        "model": llm.model,
    }
    """
    if not result:
        return {
            "answer": "没有找到答案",
            "question": question,
            "context": "",
            "tokens_used": {},
            "model": llm.model,
        }
    
    #格式化上下文
    context=format_context(result,max_token)
    #构建提示词
    prompt=build_prompt(question,context)
    #调用大模型
    message=[{"role":"user","content":prompt}]
    response=llm.generate(message,temperature=temperature,max_tokens=max_token)
    #返回结果
    return {
        "answer": response.get('content','').strip(),
        "question": question,
        "context": context,
        "tokens_used": response.get('token_usage',{}),
        "model": llm.model,
    }