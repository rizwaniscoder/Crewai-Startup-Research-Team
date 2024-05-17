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
            role='Web Research Agent',
            goal='Search for a diverse range of investors worldwide who are interested in language-learning apps, mobile applications, and AI technology.',
            backstory='As a Web Research Agent, your primary responsibility is to search the web for investors interested in language-learning apps, mobile applications, and AI technology. The startup is launching an AI-driven conversation module in the language-learning space, based in the USA, and is in the pre-seed stage without prior investments. Focus on identifying both high-profile and emerging investors who are looking for new opportunities.',
            llm=llm,
            verbose=True
        )

        platform_research_agent = Agent(
            role='Platform Research Agent',
            goal='Identify potential investors from various platforms like LinkedIn, startup incubators, and angel networks.',
            backstory='As a Platform Research Agent, your task is to explore various platforms to identify potential investors interested in AI technology and language learning apps. This includes both well-known and lesser-known investors with a history of funding early-stage startups or looking to diversify into new areas like AI-driven applications.',
            llm=llm,
            verbose=True    
        )

        research_analyst_agent = Agent(
            role='Analysis & Reporting Agent',
            goal='Organize the data collected into a comprehensive Excel sheet for easy access and prospecting.',
            backstory='As an Analysis & Reporting Agent, your primary responsibility is to organize the collected data into a comprehensive Excel sheet that facilitates easy access to both high-profile and lesser-known investor information. This will help in prioritizing potential outreach efforts.',
            llm=llm,
            verbose=True
        )

        search_investors_task = Task(
            description='###Instruction###\nYour task is to search the web for a diverse range of investors worldwide who are interested in language-learning apps, mobile applications, and AI technology. Our startup is launching an AI-driven conversation module in the language-learning space, based in the USA, and is in the pre-seed stage without prior investments. Focus on identifying both high-profile and emerging investors who are looking for new opportunities.',
            expected_output='###Output###\nCompile a comprehensive report listing investors with varying levels of fame and investment sizes:\n•\tInvestor name\n•\tInvestment focus\n•\tLocation\n•\tNotable investments\n•\tContact details (if available)\n•\tIndication of their typical investment stage and size',
            agent=market_research_agent,
            tools=[SerperDevTool(), DuckDuckGoSearchRun()],
        )

        specific_platform_task = Task(
            description='###Instruction###\nExplore LinkedIn, startup incubators, angel networks, and other investment-related platforms to identify potential investors. Target both well-known and lesser-known investors interested in AI technology and language learning apps. Ensure to include details from investors who have a history of funding early-stage startups, as well as those looking to diversify into new areas like AI-driven applications.',
            expected_output='###Output###\nProvide detailed profiles of each investor, including:\n•\tName\n•\tInvestment focus\n•\tProfessional background\n•\tAffiliated organizations\n•\tContact details (if available)\n•\tPrevious investment stages and sizes',
            agent=platform_research_agent,  
            tools=[SerperDevTool(), DuckDuckGoSearchRun()],
        )

        # scrape_investors_task = Task(
        #     description='Scrape investor information specifically for AI language learning startups',
        #     expected_output='Scraped data on potential investors interested in AI language learning startups',
        #     agent=market_research_agent,
        #     tools=[SeleniumScrapingTool()],
        #     context=[search_investors_task, specific_platform_task],
        # )

        # connect_with_investors_task = Task(
        #     description='Provide best approaches to reach investors interested in AI language learning startups and assist in fundraising',
        #     expected_output='Guidance on effective outreach strategies, compelling pitches, and fundraising techniques specifically tailored to AI language learning startups',
        #     agent=platform_research_agent,
        #     tools=[DuckDuckGoSearchRun()]  
        # )

        analyze_results_task = Task(
            description='###Instruction###\nOrganize the data collected into a comprehensive Excel sheet, suitable for prospecting a broad spectrum of investors, including emerging ones. The sheet should facilitate easy access to both high-profile and lesser-known investor information.',
            expected_output='###Output###\nCreate an Excel database with columns for:\n•\tInvestor name\n•\tInvestment focus\n•\tLocation\n•\tNotable investments\n•\tContact details\n•\tInvestment stage and size\n•\tAdditional notes on their investment trends and interests',
            agent=research_analyst_agent, 
            context=[search_investors_task, specific_platform_task],
            callback=lambda output: generate_pdf_report(output.raw_output) 
        )

        # Run the Crew
        startup_research_crew = Crew(
            agents=[market_research_agent, platform_research_agent, research_analyst_agent],
            tasks=[search_investors_task, specific_platform_task, analyze_results_task],
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
