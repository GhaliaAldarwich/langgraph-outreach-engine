import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

load_dotenv()

@tool
def scrape_company_insights(url: str) -> str:
    """Useful for researching a company URL to find their mission, recent news, or blog topics."""
    print(f"\n[Tool Executing] Simulating deep-scrape on: {url}...")
    
    # Mocking data based on URL for stable testing. 
    # You can easily swap this with real requests/BeautifulSoup logic later!
    url_lower = url.lower()
    if "tech" in url_lower:
        return """
        Company Name: NexaTech Solutions
        Mission: Accelerating enterprise cloud migration safely with AI-driven operations.
        Recent Blog Post: 'Why Legacy Database Systems are Bottlenecking AI Scale in 2026'
        Company Tone: Professional, cutting-edge, engineering-focused.
        """
    elif "health" in url_lower:
        return """
        Company Name: CarePulse Health
        Mission: Bringing patient-first telehealth infrastructure to rural communities.
        Recent News: Partnered with regional clinics to deploy remote cardiac monitoring tech.
        Company Tone: Empathetic, community-driven, security-conscious.
        """
    else:
        return f"""
        Target Domain: {url}
        General Insight: Expanding operations, heavily focused on integrating machine learning automation into their core software workflows.
        Tone: Modern startup, agile, growth-oriented.
        """

@tool
def save_outreach_draft(company_name: str, email_content: str) -> str:
    """Saves the final generated outreach email to a local text file."""
    filename = f"outreach_{company_name.lower().replace(' ', '_')}.txt"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(email_content.strip())
        return f"Successfully saved outreach draft locally to '{filename}'."
    except Exception as e:
        return f"Failed to save file due to error: {e}"


def run_outreach_generator():
    # Initialize the model with slightly higher temperature for better writing creativity
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    tools = [scrape_company_insights, save_outreach_draft]
    agent_executor = create_react_agent(model, tools)

    print("=== Hyper-Personalized Outreach Engine ===")
    
    # Collect inputs upfront
    target_url = input("Enter target company URL (e.g., nexatech.com): ").strip()
    my_skills = input("Enter your core project/skills (e.g., Python backend engineer building LangGraph tools): ").strip()
    
    if not target_url or not my_skills:
        print("Error: Both a target URL and your skills description are required.")
        return

    # Master prompt telling the agent exactly what tools to chain together in one shot
    master_prompt = f"""
    You are an expert executive outreach assistant. Your goal is to write a highly customized cold email.
    
    Follow these steps precisely:
    1. Use the 'scrape_company_insights' tool to research the company at URL: {target_url}
    2. Based on the insights found, identify a core problem or focus area they have.
    3. Tie the user's skills seamlessly into that company focus. User skills: {my_skills}
    4. Write a compelling, concise cold email pitch (Max 200 words). Avoid generic buzzwords. Make it clear, value-driven, and end with a soft call-to-action.
    5. Use the 'save_outreach_draft' tool to save the email draft permanently. Extract a clean company name to pass to the tool.
    
    Provide a brief final summary to the user showing what you did.
    """

    print("\nProcessing... Researched insights will be compiled directly into your draft.")
    
    try:
        # Stream the execution so you can watch the tool invocation happen live
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=master_prompt)]}
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")
        print("\n\n=== Process Complete ===")
    except Exception as e:
        print(f"\nExecution Error: {e}")

if __name__ == "__main__":
    run_outreach_generator()