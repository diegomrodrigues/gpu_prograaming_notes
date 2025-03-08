import argparse
from ceviche.core.context import Context
import importlib
import pkgutil
from pathlib import Path
from typing import Dict, Any
import json

def main():
    parser = argparse.ArgumentParser(description="Run Ceviche agents.")
    parser.add_argument('--agent', type=str, required=True, help='Name of the agent to run')
    parser.add_argument('--api_key', type=str, required=True, help='API Key for the LLM')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    # Add a generic --args argument to pass any other agent-specific arguments as a dictionary
    parser.add_argument('--args', type=str, help='Path to JSON file containing additional agent-specific arguments')
    # Add the --mock_api flag
    parser.add_argument('--mock_api', action='store_true', help='Use mock API for testing')

    args = parser.parse_args()

    # Dynamically load the agent class
    try:
        # Correct way to dynamically load agents
        agents_package = 'ceviche.agents'  # Assuming agents are in a package
        agent_module = importlib.import_module(f"{agents_package}.{args.agent}")
        agent_class_name = "".join(part.title() for part in args.agent.split("_")) + "Agent"
        agent_class = getattr(agent_module, agent_class_name)

    except (ImportError, AttributeError) as e:
        print(f"Error: Agent '{args.agent}' not found. {e}")
        return

    # Load agent args from JSON file if provided
    agent_args = {}
    if args.args:
        try:
            with open(args.args, 'r') as f:
                agent_args = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error: Could not load args from '{args.args}': {e}")
            return

    # Add common arguments to agent_args, ensuring no overwriting
    for key, value in vars(args).items():
        if key not in agent_args:  # Prioritize --args content
            agent_args[key] = value

    # Create a context with all necessary values
    ctx = Context()
    ctx["api_key"] = args.api_key
    ctx["mock_api"] = args.mock_api
    ctx["debug"] = args.debug
    
    # Run the agent with context and args
    agent_instance = agent_class(debug=args.debug)
    agent_instance.pre_execution(ctx, agent_args)
    result = agent_instance.execute(ctx, agent_args)
    agent_instance.post_execution(ctx, agent_args, result)

if __name__ == "__main__":
    main()