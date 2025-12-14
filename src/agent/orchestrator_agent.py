import re
from src.agent.base import Agent
from src.tools.python_tool import PythonTool
from langchain_core.messages import HumanMessage
from src.agent.risk_portfolio_agent import RiskPortfolioAgent
from src.agent.market_intelligence_analyst import MarketAnalystAgent
from src.agent.forecasting_analyst import ForecastingTechnicalAnalystAgent
from src.agent.synthesis_reccomendation_agent import SynthesisReccomendationAgent
from src.agent.prompts.orchestrator_agent_prompt import SYSTEM_PROMPT

NAME = "Supervisor Orchestrator Agent"
SUB_AGENT_CLASSES = [
    MarketAnalystAgent,
    ForecastingTechnicalAnalystAgent,
    RiskPortfolioAgent,
    SynthesisReccomendationAgent,
]

class OrchestratorAgent(Agent):
    def __init__(self, llm, sub_agent_shared_llm, name=NAME, system_prompt=SYSTEM_PROMPT):
        self.sub_agents = [AgentClass(llm=sub_agent_shared_llm) for AgentClass in SUB_AGENT_CLASSES]
        tools = [PythonTool(self._make_executor(agent)) for agent in self.sub_agents] # Create tools from sub-agents
        super().__init__(name, llm, tools, system_prompt)
    
    def _make_executor(self, agent):
        """Create a tool executor for the given sub-agent"""
        def execute(request: str) -> str:
            self._log_agent_start(agent.name)
            result = agent.invoke([HumanMessage(content=request)])
            self._log_agent_complete()
            return result["messages"][-1].content
        
        # set subagent function name
        execute.__name__ = f"execute_{self._sanitize_function_name(agent.name)}_tasks"
        # Use agent's DESCRIPTION if available, otherwise use generic description
        subagent_description = getattr(agent.__class__, 'DESCRIPTION', f"Execute tasks using {agent.name}")
        execute.__doc__ = f"""
        {subagent_description}
        
        Args:
            request (str): The task or request to delegate to the {agent.name}

        Returns:
            str: The agent's analysis and response
        """
        
        return execute
    
    def _sanitize_function_name(self, name: str) -> str:
        """Convert agent name to valid function name (alphanumeric, underscore, hyphen only)"""
        # Replace invalid characters (including &) with underscore
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
        # Replace multiple underscores with single underscore
        safe_name = re.sub(r'_+', '_', safe_name)
        # Remove leading/trailing underscores
        return safe_name.strip('_')
    
    def _log_agent_start(self, agent_name):
        """Log sub-agent execution start"""
        print(f"\n{'='*60}")
        print(f"🔍 SUB-AGENT: {agent_name}")
        print(f"{'='*60}")
    
    def _log_agent_complete(self):
        """Log sub-agent execution completion"""
        print(f"\n{'='*60}")
        print(f"✓ SUB-AGENT COMPLETE")
        print(f"{'='*60}\n")