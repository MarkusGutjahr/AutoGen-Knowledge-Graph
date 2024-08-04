from autogen.agentchat.assistant_agent import AssistantAgent
from autogen.agentchat.user_proxy_agent import UserProxyAgent
from autogen.agentchat.groupchat import GroupChat, GroupChatManager
from autogen.graph_utils import visualize_speaker_transitions_dict

config_list_gpt4 = [{
    "model": "gpt-4",
    "api_key": "OPENAI_API_KEY"
}]

gpt4_config = {
    "seed": 42,
    "temperature": 0,
    "config_list": config_list_gpt4,
}

config_list_gemini = [{
    "model": "gpt-4",
    "api_key": "GEMINI_API_KEY"
}]

gemini_config = {
    "seed": 42,
    "temperature": 0,
    "config_list": config_list_gpt4,
}

user_proxy = UserProxyAgent(
    name="Admin",
    system_message="A human admin. interact with the planner to discuss the plan, the plan execution needs to be "
                   "approved by this admin.",
    code_execution_config=False,
)

fun_engineer = AssistantAgent(
    name="Fun Engineer",
    system_message="""Fun Manager, You maximize the fun when Admin is at a location - Optimize for unique memorable 
    experieences & fun stories.""",
)

gym_trainer = AssistantAgent(
    name="Gym_Trainer",
    llm_config=gpt4_config,
    system_message="""Gym Trainer. You make sure admin is getting the right tranining (lifting 4-5 times a week) and 
    eating the right to get to a 6-pack.""",
)

executive_assistant = AssistantAgent(
    name="Executive Assistant",
    llm_config=gpt4_config,
    system_message="""Executive Assistant. You make sure the daily work (like project deadline & daily habits like 
    design and copywriting practice) required by the admin is done before any of the fun activities."""
)

planner = AssistantAgent(
    name="planner",
    llm_config=gpt4_config,
    system_message="""planner, suggest a plan. Revise the plan based on feedback from admin, Executive Assistant, 
    fun Manager, until admin approval. Explain the plan first. Be clear which sleep is performed by an engineer, 
    and wich step is performed by a scientist."""
)

critic = AssistantAgent(
    name="Critic",
    llm_config=gpt4_config,
    system_message="""Critic. Double check plan, make sure all objectives from fun manager, executive assistant, 
    and gym trainer are met. provide feedback."""
)

agents = [user_proxy, fun_engineer, gym_trainer, executive_assistant, critic]
allowed_speaker_transitions_dict = {agents: [other_agent for other_agent in agents] for agent in agents}
visualize_speaker_transitions_dict(allowed_speaker_transitions_dict, agents)

group_chat = GroupChat(
    agents=agents,
    messages=[],
    max_round=10,
    allowed_or_disallowed_speaker_transitions=allowed_speaker_transitions_dict,
    speaker_transitions_type="allowed"
)

manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=gpt4_config,
    code_execution_config=False,
)

chatbot = user_proxy.initiate_chat(
    manager,
    message="""plan  a long trip to bangkok. include a table of dates and activity. i will give you a list of tasks 
    that need to be done in a particular way."""
)

agents = [user_proxy, critic, planner, executive_assistant, gym_trainer, fun_engineer]
allowed_speaker_transitions_dict_hub_and_spoke = {
    agents[0]: [agents[1], agents[2], agents[3]],
    agents[1]: agents[0],
    agents[2]: agents[0],
    agents[3]: agents[0],
}

value = visualize_speaker_transitions_dict(allowed_speaker_transitions_dict_hub_and_spoke)

print(value)
