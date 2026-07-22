from typing import Annotated,TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel,Field
import os
from dotenv import load_dotenv
from transformers.testing_utils import backend_reset_max_memory_allocated

load_dotenv()

from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from tools.hr_tools import get_employee_profile,get_leave_balance,generate_employment_cerificate
from agent.rag_pipeline2 import search_hr_policy

# 1.定义全局共享状态(state)
class AgentState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    current_uid:str
    loop_state:int

# 2.初始化LLM语工具绑定
llm=ChatOpenAI(
    model=os.getenv('DEEPSEEK_MODEL_NAME'),
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL'),
    temperature=0.0
)

tools=[get_employee_profile,get_leave_balance,generate_employment_cerificate,search_hr_policy]
llm_with_tools=llm.bind_tools(tools)
tools_node=ToolNode(tools)
# 3.定义执行节点
def chatbot_node(state:AgentState):
    """[执行者节点]意图理解、工具调用与内容生成"""
    messages=state.get('messages',[])
    # 首轮对话注入System Prompt
    if len(messages)==1:
        system_msg=SystemMessage(
            content=f'你是飞羽科技的高级HR智能助理。\n'
                    f'当前提问员工 UID为{state.get("current_uid")}。\n'
                    f'请务必先调用get_employee_profile获取该员工的工作属性，再回答具体问题。\n'
                    f'必须基于工具返回的事实，绝对不能编造数字或条件'
        )
        messages=[system_msg]+messages
    response=llm_with_tools.invoke(messages)
    return {'messages':[response],'loop_state':state.get('loop_state',0) + 1}
class FactCheckResult(BaseModel):
    is_pass:bool=Field(description='如果AI的回答完全忠于知识库原文输出True,捏造了数字或者政策则输出False')
    feedback:str=Field(description='False,指出造假点;如果True,输"PASS"')

def fact_check_node(state:AgentState):
    """【审计节点】后置事实检验(Self-Refection)"""
    messages=state['messages']
    last_message=messages[-1]
    # 逆向查找RAG召回的原文
    rag_context=""
    for msg in reversed(messages):
        if getattr(msg,'name','')=='search_hr_policy':
            rag_context=msg.content
            break
    # 若未调用知识库，直接放行
    if not rag_context:
        return {'messages':[]}
    print('\n【审计介入】正在喝茶生成内容是否包含幻觉=========')
    check_llm = ChatOpenAI(
        model=os.getenv('DEEPSEEK_MODEL_NAME'),
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url=os.getenv('DEEPSEEK_BASE_URL'),
        temperature=0.0
    )
    parser=JsonOutputParser(pydantic_object=FactCheckResult)

    check_prompt =(
        f'你是一个冷酷的合规审计员。对比以下「知识库原文」和「AI生成的恢复」。\n'
        f'「知识库原文」：\n{rag_context}\n'
        f'「AI生成的恢复」：\n{last_message.content}\n'
        f'严查金额、职级门槛、天数！发现捏造请判False并给出修改意见。\n\n'
        f'{parser.get_format_instructions()}'
    )
    response=check_llm.invoke(check_prompt)
    # 手动解析JSON
    try:
        result=parser.invoke(response)
        is_pass=result.get('is_pass',True)
        feedback=result.get('feedback','PASS')
    except Exception as e:
        print(f'[审计异常]JSON解析失败，默认放行。原因：{e}')
        is_pass=True
        feedback='PASS'
    if is_pass:
        print('【审计通过】回答安全，无幻觉')
        return {'messages':[]}
    else:
        print(f'发现幻觉：拦截生成，审计意见：{feedback}')
        currection_msg=HumanMessage(
            content=f'[SYSTEM AUDIT FAILED]事实错误反馈:{feedback}。请根据知识库原文重写，绝不可包含虚假数据。',
        )
        return {'messages':[currection_msg]}

# 4.定义路由逻辑
def router_after_chatbot(state:AgentState):
    """Chatbot输出后的路由判断"""
    last_message=state['messages'][-1]
    if last_message.tool_calls:
        return 'tools'
    else:
        return 'fact_checker'

def router_after_fact_check(state:AgentState):
    """审计完成后的路由判断"""
    last_message=state['messages'][-1]
    if isinstance(last_message,HumanMessage):
        if state.get('loop_state',0)>4:
            print('强制熔断反思次数达到上限，放弃纠错')
            return 'end'
        print('[打回重写]图路由指针倒流回chatbot节点')
        return 'chatbot'
    return 'end'

# 5.构建状态图
workflow=StateGraph(AgentState)

workflow.add_node('chatbot',chatbot_node)
workflow.add_node('tools',tools_node)
workflow.add_node('fact_checker',fact_check_node)

workflow.add_edge(START,'chatbot')
workflow.add_conditional_edges(
    'chatbot',router_after_chatbot,{'tools':'tools',
                                    'fact_checker':'fact_checker',}
)
workflow.add_edge('tools','chatbot')
workflow.add_conditional_edges('fact_checker',router_after_fact_check,{
    'chatbot':'chatbot',
    'end':END
})
hr_agent_app=workflow.compile()