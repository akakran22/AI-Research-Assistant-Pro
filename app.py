import streamlit as st
import os
from dotenv import load_dotenv
import json
import re
from datetime import datetime
from io import BytesIO
import time

# Third-party imports
from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Research Assistant Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .feature-container {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        flex: 1;
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        text-align: center;
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
    }
    
    .report-content {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
    
    .report-content h1 {
        color: #1e40af !important;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5rem;
    }
    
    .report-content h2 {
        color: #1e40af !important;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.3rem;
    }
    
    .report-content h3 {
        color: #374151 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'research_result' not in st.session_state:
    st.session_state.research_result = None
if 'research_query' not in st.session_state:
    st.session_state.research_query = ""

class TavilyRetrievalSystem:
    def __init__(self, api_key):
        self.tavily = TavilyClient(api_key=api_key)
    
    def advanced_search(self, query):
        try:
            # Primary search
            response = self.tavily.search(
                query=query,
                search_depth="advanced",
                max_results=15,
                include_answer=True,
                include_raw_content=True,
                include_domains=[
                    "yourstory.com", "economictimes.indiatimes.com", 
                    "techcrunch.com", "inc42.com", "business-standard.com",
                    "livemint.com", "forbesindia.com", "moneycontrol.com"
                ]
            )
            
            # Additional targeted searches
            targeted_searches = [
                f"{query} funding investment 2024 2025",
                f"{query} latest news recent developments"
            ]
            
            all_results = [self._format_response(response)]
            
            for targeted_query in targeted_searches:
                try:
                    additional_response = self.tavily.search(
                        query=targeted_query,
                        search_depth="basic",
                        max_results=5,
                        include_answer=True
                    )
                    all_results.append(self._format_response(additional_response))
                except:
                    continue
            
            return "\n\n" + "="*50 + "\n\n".join(all_results)
        
        except Exception as e:
            raise Exception(f"Search failed: {str(e)}")
    
    def _format_response(self, response):
        formatted = []
        
        if response.get('answer'):
            formatted.append(f"**INSIGHT:** {response['answer']}\n")
        
        if response.get('results'):
            for i, result in enumerate(response['results'], 1):
                formatted.append(f"**SOURCE {i}:** {result.get('title', 'No title')}")
                formatted.append(f"**URL:** {result.get('url', 'No URL')}")
                formatted.append(f"**CONTENT:** {result.get('content', 'No content')}")
                formatted.append("-" * 30)
        
        return "\n".join(formatted)

class ResearchAgents:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        if not self.groq_api_key or not self.tavily_api_key:
            raise ValueError("API keys not found in environment variables")
        
        self.llm = ChatGroq(
            api_key=self.groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=4000
        )
    
    def research_agent(self, query, search_results):
        prompt = PromptTemplate(
            input_variables=["query", "search_results"],
            template="""You are an Expert Research Agent. Analyze this search data about: {query}

SEARCH RESULTS:
{search_results}

Extract:
1. Specific company names, funding amounts, recent developments
2. Key market players and leaders  
3. Concrete statistics and growth data
4. Expert quotes and industry insights
5. Recent news and technological breakthroughs

Focus on factual, recent information. Prioritize Indian companies and current developments."""
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query, "search_results": search_results})
    
    def summarizer_agent(self, research_content):
        prompt = PromptTemplate(
            input_variables=["research_content"],
            template="""Process this research content:
{research_content}

Create structured summary:

## Company Profiles
- Company name, founding year, headquarters
- Core AI technology and healthcare focus
- Key products/services and recent funding

## Market Intelligence  
- Market size, growth statistics
- Investment trends, key partnerships
- Technology applications and innovations

## Recent Developments
- Latest news, product launches
- Awards, recognitions, expansions

Include specific numbers, dates, and company names."""
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"research_content": research_content})
    
    def critic_agent(self, summary_content):
        prompt = PromptTemplate(
            input_variables=["summary_content"],
            template="""Evaluate this content for accuracy:
{summary_content}

Analyze:
- Information consistency and conflicts
- Data completeness and currency  
- Source credibility and reliability
- Missing critical information

Provide reliability score (1-10) and improvement recommendations."""
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"summary_content": summary_content})
    
    def writer_agent(self, research_data, summary, critique):
        prompt = PromptTemplate(
            input_variables=["research_data", "summary", "critique"],
            template="""Create a comprehensive research report using:

RESEARCH: {research_data}
SUMMARY: {summary}  
CRITIQUE: {critique}

Structure:

# Executive Summary
Brief overview of Indian AI healthcare startup ecosystem, key insights, and top performers.

# Top AI Healthcare Startups in India
For each company:
## [Company Name]
- **Founded:** Year, Location
- **Focus Area:** Healthcare AI application
- **Technology:** Core AI technologies
- **Funding:** Latest rounds, total raised, valuation
- **Products:** Main offerings
- **Recent News:** Latest developments

# Market Analysis
- **Market Size:** Current and projected figures
- **Growth Trends:** Investment patterns and statistics  
- **Key Technologies:** Popular AI applications
- **Challenges:** Market obstacles and opportunities

# Investment Landscape
- **Funding Trends:** Recent investment patterns
- **Key Investors:** Major VCs and sources
- **Success Stories:** Notable achievements

# Future Outlook
- **Emerging Trends:** Next-gen technologies
- **Predictions:** Market forecasts
- **Opportunities:** Development areas

# References and Sources
List sources with clickable URLs and publication dates.

Use professional tone with specific data, figures, and company details."""
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "research_data": research_data,
            "summary": summary,
            "critique": critique
        })

def process_markdown_to_html(content):
    """Convert markdown-style content to HTML"""
    if not content:
        return ""
    
    # Process line by line
    lines = content.split('\n')
    processed_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Process headings
        if line.startswith('### '):
            processed_lines.append(f'<h3 style="color: #374151; margin-top: 1.5rem; margin-bottom: 0.8rem;">{line[4:]}</h3>')
        elif line.startswith('## '):
            processed_lines.append(f'<h2 style="color: #1e40af; margin-top: 2rem; margin-bottom: 1rem; padding-bottom: 0.3rem; border-bottom: 2px solid #e2e8f0;">{line[3:]}</h2>')
        elif line.startswith('# '):
            processed_lines.append(f'<h1 style="color: #1e40af; margin-top: 2.5rem; margin-bottom: 1.5rem; padding-bottom: 0.5rem; border-bottom: 3px solid #3b82f6;">{line[2:]}</h1>')
        # Process bullet points
        elif line.startswith('- ') or line.startswith('* '):
            bullet_text = line[2:].strip()
            bullet_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', bullet_text)
            bullet_text = re.sub(r'(?<!\*)\*(.*?)\*(?!\*)', r'<em>\1</em>', bullet_text)
            processed_lines.append(f'<li style="margin-bottom: 0.5rem;">{bullet_text}</li>')
        # Process regular paragraphs
        else:
            line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
            line = re.sub(r'(?<!\*)\*(.*?)\*(?!\*)', r'<em>\1</em>', line)
            line = re.sub(r'(https?://[^\s<>"{}|\\^[\]]+)', r'<a href="\1" target="_blank" style="color: #3b82f6; text-decoration: underline;">\1</a>', line)
            processed_lines.append(f'<p style="margin-bottom: 1rem; line-height: 1.6;">{line}</p>')
    
    html = '\n'.join(processed_lines)
    
    # Wrap consecutive <li> elements in <ul>
    html = re.sub(r'(<li[^>]*>.*?</li>(?:\s*<li[^>]*>.*?</li>)*)', r'<ul style="margin-bottom: 1.5rem; padding-left: 1.5rem;">\1</ul>', html, flags=re.DOTALL)
    
    return html

def generate_pdf(report_content, query):
    """Generate PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=1*inch)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=20,
        spaceAfter=30,
        alignment=1
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=12
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=16,
        spaceAfter=10
    )
    
    content = []
    content.append(Paragraph("🔬 AI Research Report", title_style))
    content.append(Paragraph(f"<b>Query:</b> {query}", styles['Normal']))
    content.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    content.append(Spacer(1, 0.3*inch))
    
    # Process content with proper markdown conversion
    lines = report_content.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            content.append(Spacer(1, 0.1*inch))
            continue
            
        # Convert markdown to ReportLab HTML
        processed_line = line
        
        # Convert **bold** to <b>bold</b>
        processed_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', processed_line)
        
        # Convert *italic* to <i>italic</i>
        processed_line = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<i>\1</i>', processed_line)
        
        # Convert URLs to clickable links
        processed_line = re.sub(r'(https?://[^\s<>"{}|\\^[\]]+)', r'<link href="\1">\1</link>', processed_line)
        
        # Handle headings
        if line.startswith('# '):
            content.append(Paragraph(processed_line[2:], heading1_style))
        elif line.startswith('## '):
            content.append(Paragraph(processed_line[3:], heading2_style))
        elif line.startswith('### '):
            content.append(Paragraph(processed_line[4:], styles['Heading3']))
        elif line.startswith('- ') or line.startswith('* '):
            bullet_text = processed_line[2:]
            content.append(Paragraph(f"• {bullet_text}", styles['Normal']))
        else:
            if processed_line:
                content.append(Paragraph(processed_line, styles['Normal']))
    
    doc.build(content)
    buffer.seek(0)
    return buffer

def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 AI Research Assistant Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Multi-Agent Research System powered by LLaMA-3.3-70B and Tavily AI Search</p>', unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2rem; color: red; margin-bottom: 0.5rem;">🤖</div>
            <h5>LLaMA-3.3-70B</h5>
            <p>Advanced language model for intelligent analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2rem; color: #3b82f6; margin-bottom: 0.5rem;">🔍</div>
            <h5>Tavily AI Search</h5>
            <p>AI-optimized search engine for comprehensive data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2rem; color: #3b82f6; margin-bottom: 0.5rem;">📊</div>
            <h5>Multi-Agent Analysis</h5>
            <p>Research, summarize, critique, and write with AI agents</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2rem; color: #3b82f6; margin-bottom: 0.5rem;">📄</div>
            <h5>Professional Export</h5>
            <p>Download reports in PDF and Markdown formats</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Research Form
    st.markdown("### 🎯 Start Your Research")
    
    query = st.text_area(
        "Research Query",
        height=150,
        placeholder="Enter your research question here...\n\n• Top 5 AI startups in Indian Healthcare\n• Latest fintech trends in Southeast Asia\n• Renewable energy investments in Europe 2025",
        help="Be specific about your research topic. Include industry, region, or timeframe for better results."
    )
    
    if st.button("🚀 Start Advanced Research", type="primary", use_container_width=True):
        if not query.strip():
            st.error("Please enter a research query")
            return
        
        # Check API keys
        if not os.getenv("GROQ_API_KEY") or not os.getenv("TAVILY_API_KEY"):
            st.error("API keys not found. Please check your .env file")
            return
        
        try:
            # Initialize progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Phase 1: Initialize
            status_text.text("🔧 Initializing AI agents...")
            progress_bar.progress(10)
            agents = ResearchAgents()
            retrieval = TavilyRetrievalSystem(agents.tavily_api_key)
            
            # Phase 2: Search
            status_text.text("🔍 Conducting advanced web search...")
            progress_bar.progress(25)
            search_results = retrieval.advanced_search(query)
            
            # Phase 3: Research
            status_text.text("📊 Analyzing search results...")
            progress_bar.progress(40)
            research_data = agents.research_agent(query, search_results)
            
            # Phase 4: Summarize
            status_text.text("📝 Processing and summarizing information...")
            progress_bar.progress(60)
            summary = agents.summarizer_agent(research_data)
            
            # Phase 5: Critique
            status_text.text("🔍 Fact-checking and verification...")
            progress_bar.progress(80)
            critique = agents.critic_agent(summary)
            
            # Phase 6: Write Report
            status_text.text("✍️ Generating final report...")
            progress_bar.progress(95)
            final_report = agents.writer_agent(research_data, summary, critique)
            
            # Complete
            status_text.text("✅ Research completed successfully!")
            progress_bar.progress(100)
            
            # Store results
            st.session_state.research_result = final_report
            st.session_state.research_query = query
            
            st.success("Research completed! Results are displayed below.")
            
        except Exception as e:
            st.error(f"Research failed: {str(e)}")
            return
    
    # Display Results
    if st.session_state.research_result:
        st.markdown("---")
        st.markdown("## 📋 Research Report")
        st.markdown(f"**Query:** {st.session_state.research_query}")
        
        # Process and display report
        html_content = process_markdown_to_html(st.session_state.research_result)
        st.markdown(f'<div class="report-content">{html_content}</div>', unsafe_allow_html=True)
        
        # Download options
        st.markdown("### 💾 Export Your Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Markdown download
            markdown_content = f"""# 🔬 AI Research Report

**Research Query:** {st.session_state.research_query}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Powered by:** LLaMA-3.3-70B + Tavily Advanced Search

---

{st.session_state.research_result}

---

## Report Metadata
- **Search Engine:** Tavily AI Search (Advanced)
- **AI Model:** LLaMA-3.3-70B-Versatile via Groq
- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

*Generated by AI Research Assistant with comprehensive web search capabilities.*
"""
            st.download_button(
                label="📝 Download Markdown",
                data=markdown_content,
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        
        with col2:
            # PDF download
            try:
                pdf_buffer = generate_pdf(st.session_state.research_result, st.session_state.research_query)
                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_buffer.getvalue(),
                    file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"PDF generation failed: {str(e)}")
        
        with col3:
            # JSON download
            json_data = {
                "query": st.session_state.research_query,
                "report": st.session_state.research_result,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "ai_model": "llama-3.3-70b-versatile",
                    "search_engine": "tavily_advanced"
                }
            }
            st.download_button(
                label="💻 Download JSON",
                data=json.dumps(json_data, indent=2),
                file_name=f"research_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()




