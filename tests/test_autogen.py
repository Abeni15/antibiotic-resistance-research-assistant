import asyncio,json,unittest
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import FunctionCall
from autogen_core.models import CreateResult,RequestUsage
from autogen_ext.models.replay import ReplayChatCompletionClient
async def fixture_search(query:str)->str:
    """Search a deterministic test fixture."""
    return "fixture:" + query
class AutoGenTests(unittest.TestCase):
    def test_agent_calls_registered_python_tool_with_mock_model(self):
        async def run():
            mocked_call=CreateResult(finish_reason="function_calls",content=[FunctionCall(id="call1",name="fixture_search",arguments=json.dumps({"query":"efflux"}))],usage=RequestUsage(prompt_tokens=1,completion_tokens=1),cached=False)
            client=ReplayChatCompletionClient([mocked_call],model_info={"vision":False,"function_calling":True,"json_output":True,"family":"unknown","structured_output":True})
            agent=AssistantAgent(name="test_researcher",model_client=client,tools=[fixture_search],system_message="Use the fixture tool.")
            self.assertEqual(agent.name,"test_researcher");self.assertEqual(len(agent._tools),1)
            result=await agent.run(task="Search for efflux.")
            await client.close()
            return result.messages[-1].content
        self.assertEqual(asyncio.run(run()),"fixture:efflux")
if __name__=="__main__":unittest.main()
