# Import necessary libraries
import streamlit as st
import sys
import re
from datetime import datetime
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
import os
from crewai_tools import SerperDevTool

from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import SeleniumScrapingTool

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def generate_pdf_report(output: dict):
    """
    Generate a PDF report from the task output.
    """
    filename = f"task_report_{output['description'].replace(' ', '_')}.pdf"
    
    # Create a canvas for PDF generation
    c = canvas.Canvas(filename, pagesize=letter)
    
    # Add content to the PDF report
    c.drawString(100, 750, f"Task Description: {output['description']}")
    c.drawString(100, 730, f"Expected Output: {output['expected_output']}")
    
    # Add more content as needed
    
    # Save the PDF report
    c.save()
    
    print(f"Task completed! Report saved to '{filename}'")


class StreamToExpander:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []
        self.colors = ['red', 'green', 'blue', 'orange']
        self.color_index = 0

    def write(self, data):
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)

        task_match_object = re.search(r'\"task\"\s*:\s*\"(.*?)\"', cleaned_data, re.IGNORECASE)
        task_match_input = re.search(r'task\s*:\s*([^\n]*)', cleaned_data, re.IGNORECASE)
        task_value = None
        if task_match_object:
            task_value = task_match_object.group(1)
        elif task_match_input:
            task_value = task_match_input.group(1).strip()

        if task_value:
            st.toast(":robot_face: " + task_value)

        if "Entering new CrewAgentExecutor chain" in cleaned_data:
            self.color_index = (self.color_index + 1) % len(self.colors)

            cleaned_data = cleaned_data.replace("Entering new CrewAgentExecutor chain",
                                                f":{self.colors[self.color_index]}[Entering new CrewAgentExecutor chain]")

        if "Market Research Analyst" in cleaned_data:
            cleaned_data = cleaned_data.replace("Market Research Analyst",
                                                f":{self.colors[self.color_index]}[Market Research Analyst]")
        if "Business Development Consultant" in cleaned_data:
            cleaned_data = cleaned_data.replace("Business Development Consultant",
                                                f":{self.colors[self.color_index]}[Business Development Consultant]")
        if "Technology Expert" in cleaned_data:
            cleaned_data = cleaned_data.replace("Technology Expert",
                                                f":{self.colors[self.color_index]}[Technology Expert]")
        if "Finished chain." in cleaned_data:
            cleaned_data = cleaned_data.replace("Finished chain.", f":{self.colors[self.color_index]}[Finished chain.]")

        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []
  

class CrewAIApp:
    def __init__(self):
        self.llm_options = ['OpenAI GPT-4', 'Claude-3', 'Groq']

    def run(self):
        st.title("Fela - Crewai Agents Team for AI Language Learning Startups Research")

        llm_option = st.selectbox("Choose LLM for Agents:", self.llm_options)
        api_key = st.text_input("Enter API Key for chosen LLM:", type="password")
        serper_api_key = st.text_input("Enter Serper API Key:", type="password")
        os.environ["SERPER_API_KEY"] = serper_api_key if serper_api_key else ""

        if st.button("Run Startup Research"):
            llm = self.setup_llm(llm_option, api_key)
            self.execute_startup_research(llm)

    def setup_llm(self, llm_option, api_key):
        if llm_option == 'OpenAI GPT-4':
            return ChatOpenAI(model="gpt-4-0125-preview", api_key=api_key)
        elif llm_option == 'Claude-3':
            return ChatAnthropic(model="claude-3-haiku-20240307", api_key=api_key)
        else:  
            return ChatGroq(api_key=api_key, model_name="llama3-8b-8192")

    def execute_startup_research(self, llm):
        
        market_research_agent = Agent(
            role='Market Research Analyst',
            goal='Identify potential investors for AI language learning startups',
            backstory='As a market research analyst, your primary responsibility is to conduct comprehensive research to identify potential investors interested specifically in AI language learning startups. Your expertise lies in gathering detailed information about investors who have previously invested in similar startups, as well as those who show interest in investing in the AI education sector, with a focus on language learning applications.',
            llm=llm,
            verbose=True
        )

        business_development_agent = Agent(
            role='Business Development Consultant',
            goal='Provide best approaches to establish connections with potential investors in AI language learning startups',
            backstory='As a business development consultant, your primary responsibility is to facilitate investments in AI language learning startups by establishing connections with potential investors. Your expertise lies in identifying the most effective approaches to reach out to investors specifically interested in AI language learning startups, craft compelling pitches tailored to this sector, and foster meaningful relationships to secure funding.',
            llm=llm,
            verbose=True    
        )

        research_analyst_agent = Agent(
            role='Research Analyst',
            goal='Analyze results and provide a comprehensive report on AI language learning startups',
            backstory='As a research analyst, your primary responsibility is to analyze all collected data related to AI language learning startups, identify potential investors, analyze trends in the AI education space specifically focusing on language learning applications, and determine key areas to focus on when applying for fundraising for such startups. Your expertise lies in conducting thorough data analysis and providing actionable insights to guide the fundraising strategy for AI language learning startups.',
            llm=llm,
            verbose=True
        )

        search_investors_task = Task(
            description='Search for potential investors specifically interested in AI language learning startups. Find all the investors around the world who are interested in, have invested in, or are willing to invest in similar startups, with a focus on language learning applications.',
            expected_output='List of potential investors specifically interested in AI language learning startups and their details',
            agent=market_research_agent,
            tools=[SerperDevTool(), SeleniumScrapingTool()],
            async_execution=True,
        )

        specific_platform_task = Task(
            description='Search for investors interested in AI language learning startups on specific platforms (e.g., LinkedIn, startup incubators, Angel investors in New York)',
            expected_output='Data scraped from specific platforms related to AI language learning startups',
            agent=market_research_agent,  
            tools=[SerperDevTool(), SeleniumScrapingTool()],
            async_execution=True,
        )

        scrape_investors_task = Task(
            description='Scrape investor information specifically for AI language learning startups',
            expected_output='Scraped data on potential investors interested in AI language learning startups',
            agent=market_research_agent,
            tools=[SeleniumScrapingTool()],
            context=[search_investors_task, specific_platform_task],
        )

        connect_with_investors_task = Task(
            description='Provide best approaches to reach investors interested in AI language learning startups and assist in fundraising',
            expected_output='Guidance on effective outreach strategies, compelling pitches, and fundraising techniques specifically tailored to AI language learning startups',
            agent=business_development_agent,
            tools=[DuckDuckGoSearchRun()]  
        )

        analyze_results_task = Task(
            description='Analyze results and provide a comprehensive report on AI language learning startups',
            expected_output='Comprehensive report on potential investors interested in AI language learning startups, trends in the AI education space focusing on language learning applications, and key fundraising areas specifically tailored to such startups',
            agent=research_analyst_agent, 
            context=[search_investors_task, specific_platform_task, scrape_investors_task, connect_with_investors_task],
            callback=lambda output: generate_pdf_report(output.raw_output) 
        )

        # Run the Crew
        startup_research_crew = Crew(
            agents=[market_research_agent, business_development_agent, research_analyst_agent],
            tasks=[search_investors_task, specific_platform_task, scrape_investors_task, connect_with_investors_task, analyze_results_task],
            verbose=True
        )

        log_expander = st.expander("Execution Logs", expanded=False)
        sys.stdout = StreamToExpander(log_expander)

        crew_results = startup_research_crew.kickoff()
        print(crew_results)

        # Save results to text file
        with open("startup_research_results.txt", "a") as text_file:
            text_file.write(f"----------------------\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            text_file.write(f"{crew_results}\n")

if __name__ == "__main__":
    app = CrewAIApp()
    app.run()
