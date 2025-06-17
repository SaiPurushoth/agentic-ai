# Simplified Strategic Roadmap Agent

import requests
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from strands import Agent, tool
from strands.models.ollama import OllamaModel

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"

@dataclass
class Employee:
    """Simple employee data structure"""
    id: int
    name: str
    role: str
    skills: List[str]
    hourly_rate: float
    availability: str = "available"
    location: str = ""

@dataclass
class ProjectRequirement:
    """Simple project requirement structure"""
    description: str
    priority: str = "medium"  # high, medium, low
    category: str = "technical"  # technical, compliance, business

class AnalysisEngine:
    """Simplified analysis engine for LLM interactions"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model_name = model_name
        self.base_url = base_url
    
    def analyze(self, prompt: str) -> Dict[str, Any]:
        """Generic analysis method"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "content": response.json()["response"],
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {"success": False, "error": f"API Error: {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}

class ResourceMatcher:
    """Simplified resource matching logic"""
    
    def __init__(self, employees: List[Employee]):
        self.employees = employees
    
    def find_matches(self, requirements: List[str], max_budget: Optional[float] = None) -> List[Dict]:
        """Find employees matching requirements"""
        matches = []
        
        for emp in self.employees:
            if max_budget and emp.hourly_rate > max_budget:
                continue
                
            # Simple skill matching
            matching_skills = self._get_matching_skills(emp.skills, requirements)
            if matching_skills:
                match_score = len(matching_skills) / len(requirements) * 100
                matches.append({
                    "employee": emp,
                    "matching_skills": matching_skills,
                    "match_score": round(match_score, 1)
                })
        
        # Sort by availability and match score
        matches.sort(key=lambda x: (x["employee"].availability == "available", x["match_score"]), reverse=True)
        return matches
    
    def _get_matching_skills(self, emp_skills: List[str], requirements: List[str]) -> List[str]:
        """Simple skill matching logic"""
        matches = []
        for req in requirements:
            for skill in emp_skills:
                if req.lower() in skill.lower() or skill.lower() in req.lower():
                    matches.append(skill)
        return list(set(matches))

class RoadmapGenerator:
    """Simplified roadmap generation"""
    
    def __init__(self, analysis_engine: AnalysisEngine):
        self.engine = analysis_engine
    
    def create_phases(self, complexity: str = "medium") -> Dict[str, int]:
        """Generate standard project phases"""
        phase_templates = {
            "simple": {"Planning": 2, "Implementation": 4, "Testing": 2, "Deployment": 1},
            "medium": {"Planning": 3, "Design": 2, "Implementation": 6, "Testing": 3, "Deployment": 2},
            "complex": {"Planning": 4, "Design": 3, "Implementation": 8, "Testing": 4, "Deployment": 3, "Optimization": 2}
        }
        return phase_templates.get(complexity, phase_templates["medium"])

# Simplified Tools
@tool
def analyze_requirements(project_data: Dict[str, Any], model_name: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    Analyze project requirements and extract key technical information for roadmap planning.
    
    This tool analyzes a project description and extracts structured information including
    technical requirements, priorities, complexity assessment, and risk factors.
    
    Args:
        project_data (Dict[str, Any]): Project information containing:
            - client_name (str): Name of the client/company
            - description (str): Project description 
            - requirements (List[str]): List of technical requirements
            - budget (str, optional): Budget information
            - timeline (str, optional): Expected timeline
            - compliance_needs (str, optional): Compliance requirements
        model_name (str, optional): Ollama model name to use. Defaults to 'llama3.2'
    
    Returns:
        Dict[str, Any]: Analysis results containing:
            - analysis (str): Detailed analysis text
            - timestamp (str): When analysis was performed
            - project_name (str): Extracted project name
            - error (str, optional): Error message if analysis failed
    
    Example:
        project = {
            "client_name": "TechCorp", 
            "description": "Cloud migration project",
            "requirements": ["AWS migration", "Security compliance"]
        }
        result = analyze_requirements(project)
    """
    
    engine = AnalysisEngine(model_name)
    
    prompt = f"""
    Analyze this project and extract:
    1. Key technical requirements (list 3-5 main items)
    2. Priority level (high/medium/low for each requirement)
    3. Estimated complexity (simple/medium/complex)
    4. Main risk factors (list top 3)
    
    Project: {json.dumps(project_data, indent=2)}
    
    Respond in JSON format with: technical_requirements, priorities, complexity, risks
    """
    
    result = engine.analyze(prompt)
    if result["success"]:
        try:
            # Simple parsing - in production, you'd use more robust JSON parsing
            return {
                "analysis": result["content"],
                "timestamp": result["timestamp"],
                "project_name": project_data.get("client_name", "Unknown")
            }
        except:
            return {"error": "Failed to parse analysis", "raw": result["content"]}
    else:
        return result

@tool
def match_team(requirements: List[str], budget_limit: Optional[float] = None, max_team_size: int = 4) -> Dict[str, Any]:
    """
    Find and recommend the best team members for a project based on technical requirements.
    
    This tool matches project requirements with available team members based on their
    skills, availability, and hourly rates. It returns a recommended team with cost analysis.
    
    Args:
        requirements (List[str]): List of technical skills/requirements needed for the project.
            Examples: ["AWS", "Security", "Database migration", "CI/CD", "React"]
        budget_limit (Optional[float]): Maximum hourly rate budget per team member. 
            If specified, excludes team members above this rate. Defaults to None (no limit)
        max_team_size (int): Maximum number of team members to recommend. Defaults to 4
    
    Returns:
        Dict[str, Any]: Team matching results containing:
            - recommended_team (List[Dict]): List of recommended team members with:
                - name (str): Team member name
                - role (str): Their primary role
                - hourly_rate (float): Cost per hour
                - matching_skills (List[str]): Skills that match requirements
                - match_score (float): Percentage match with requirements
                - availability (str): Current availability status
            - team_cost (Dict): Cost analysis with weekly_cost, team_size, average_rate
            - coverage (Dict): Skill coverage analysis and total candidates found
    
    Example:
        requirements = ["AWS", "Security", "Database migration"]
        team = match_team(requirements, budget_limit=150.0, max_team_size=3)
    """
    
    # Sample employee database - in practice, this would come from a database
    employees = [
        Employee(1, "Sarah Chen", "Cloud Architect", ["AWS", "Azure", "Kubernetes", "Security"], 150),
        Employee(2, "Marcus Rodriguez", "DevOps Engineer", ["CI/CD", "Docker", "AWS", "Monitoring"], 120),
        Employee(3, "Emily Watson", "Security Specialist", ["Security", "Compliance", "HIPAA", "SOC2"], 180, "limited"),
        Employee(4, "Alex Kim", "Database Engineer", ["PostgreSQL", "MongoDB", "AWS RDS", "Migration"], 130),
        Employee(5, "Priya Patel", "Full Stack Developer", ["React", "Node.js", "Python", "APIs"], 110),
    ]
    
    matcher = ResourceMatcher(employees)
    matches = matcher.find_matches(requirements, budget_limit)
    
    recommended_team = matches[:max_team_size]  # Use parameter instead of hardcoded value
    total_cost = sum(match["employee"].hourly_rate for match in recommended_team)
    
    return {
        "recommended_team": [
            {
                "name": match["employee"].name,
                "role": match["employee"].role,
                "hourly_rate": match["employee"].hourly_rate,
                "matching_skills": match["matching_skills"],
                "match_score": match["match_score"],
                "availability": match["employee"].availability
            }
            for match in recommended_team
        ],
        "team_cost": {
            "weekly_cost": total_cost * 40,
            "team_size": len(recommended_team),
            "average_rate": round(total_cost / len(recommended_team), 2) if recommended_team else 0
        },
        "coverage": {
            "covered_skills": list(set(skill for match in matches for skill in match["matching_skills"])),
            "total_candidates": len(matches)
        }
    }

@tool
def generate_project_roadmap(analysis_result: Dict[str, Any], team_result: Dict[str, Any], 
                           project_complexity: str = "medium", model_name: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    Generate a comprehensive project roadmap with phases, timeline, and team assignments.
    
    This tool creates a detailed project roadmap combining the requirements analysis and
    team matching results. It generates phases, timelines, cost estimates, and actionable next steps.
    
    Args:
        analysis_result (Dict[str, Any]): Results from analyze_requirements tool containing:
            - analysis (str): Project analysis text
            - project_name (str): Project name
            - timestamp (str): Analysis timestamp
        team_result (Dict[str, Any]): Results from match_team tool containing:
            - recommended_team (List): Team member recommendations
            - team_cost (Dict): Cost analysis
            - coverage (Dict): Skill coverage information
        project_complexity (str, optional): Project complexity level. 
            Options: "simple", "medium", "complex". Affects phase structure and duration.
            Defaults to "medium"
        model_name (str, optional): Ollama model name for roadmap generation. 
            Defaults to 'llama3.2'
    
    Returns:
        Dict[str, Any]: Complete project roadmap containing:
            - project_name (str): Name of the project
            - generated_at (str): Roadmap generation timestamp
            - timeline (Dict): Phase-by-phase timeline with start/end dates
            - team_assignment (List): Assigned team members with details
            - budget_summary (Dict): Total cost, duration, and weekly cost breakdown
            - roadmap_details (str): AI-generated detailed roadmap description
            - next_steps (List[str]): Recommended immediate actions
    
    Example:
        analysis = analyze_requirements(project_data)
        team = match_team(["AWS", "Security"])
        roadmap = generate_project_roadmap(analysis, team, "complex")
    """
    
    engine = AnalysisEngine(model_name)
    generator = RoadmapGenerator(engine)
    
    # Determine project complexity from parameter
    phases = generator.create_phases(project_complexity)
    
    # Generate timeline
    start_date = datetime.now()
    timeline = {}
    current_date = start_date
    
    for phase_name, duration_weeks in phases.items():
        end_date = current_date + timedelta(weeks=duration_weeks)
        timeline[phase_name] = {
            "start": current_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
            "duration_weeks": duration_weeks
        }
        current_date = end_date
    
    # Generate detailed roadmap with AI
    prompt = f"""
    Create a project roadmap summary for:
    
    Analysis: {analysis_result.get('analysis', '')}
    Team: {len(team_result.get('recommended_team', []))} members
    Budget: ${team_result.get('team_cost', {}).get('weekly_cost', 0)}/week
    Duration: {sum(phases.values())} weeks
    
    Provide:
    1. Executive summary (2-3 sentences)
    2. Key deliverables for each phase
    3. Success metrics
    4. Main risks and mitigation
    
    Keep it concise and actionable.
    """
    
    roadmap_analysis = engine.analyze(prompt)
    
    return {
        "project_name": analysis_result.get("project_name", "Unknown"),
        "generated_at": datetime.now().isoformat(),
        "timeline": timeline,
        "team_assignment": team_result.get("recommended_team", []),
        "budget_summary": {
            "total_duration_weeks": sum(phases.values()),
            "estimated_total_cost": team_result.get('team_cost', {}).get('weekly_cost', 0) * sum(phases.values()),
            "weekly_cost": team_result.get('team_cost', {}).get('weekly_cost', 0)
        },
        "roadmap_details": roadmap_analysis.get("content", "") if roadmap_analysis.get("success") else "Failed to generate details",
        "next_steps": [
            "Review and approve roadmap",
            "Finalize team assignments",
            "Set up project infrastructure",
            "Begin Phase 1 execution"
        ]
    }

# Main Agent Setup
def create_strategic_agent():
    """Create the main strategic planning agent"""
    
    model = OllamaModel(model_id=DEFAULT_MODEL, host=OLLAMA_BASE_URL)
    
    return Agent(
        system_prompt="""
        You are a Strategic Project Planning Assistant specialized in creating comprehensive project roadmaps.
        
        Your process is always:
        1. FIRST: Use analyze_requirements() to analyze the project data and extract key technical requirements
        2. SECOND: Use match_team() with the extracted requirements to find the best team members
        3. THIRD: Use generate_project_roadmap() with both previous results to create the final roadmap
        
        IMPORTANT TOOL USAGE:
        - analyze_requirements(project_data, model_name="llama3.2")
          - project_data must be a dictionary with client info and requirements
        
        - match_team(requirements, budget_limit=None, max_team_size=4) 
          - requirements must be a list of technical skills like ["AWS", "Security", "Database"]
          - budget_limit is optional hourly rate limit
        
        - generate_project_roadmap(analysis_result, team_result, project_complexity="medium")
          - Use the exact output from the first two tools
          - project_complexity can be "simple", "medium", or "complex"
        
        Always use ALL THREE tools in this exact sequence and provide a clear executive summary of the final roadmap.
        """,
        model=model,
        tools=[analyze_requirements, match_team, generate_project_roadmap]
    )

# Example Usage
if __name__ == "__main__":
    # Comprehensive project data example
    project = {
        "client_name": "TechCorp Industries",
        "description": "Cloud migration project with security compliance requirements",
        "requirements": [
            "AWS migration", 
            "Security compliance", 
            "Auto-scaling implementation", 
            "Database migration",
            "CI/CD pipeline setup"
        ],
        "budget": "$500K total budget",
        "timeline": "6 months",
        "compliance_needs": "SOC2, HIPAA compliance required"
    }
    
    # Create and run agent
    agent = create_strategic_agent()
    
    result = agent(f"""
    Please create a comprehensive project roadmap for the following project:
    
    PROJECT DETAILS:
    {json.dumps(project, indent=2)}
    
    INSTRUCTIONS:
    1. First analyze the requirements and extract key technical needs
    2. Then find the best team members for these requirements (budget limit: $160/hour)
    3. Finally generate a complete roadmap with phases and timeline
    
    Please use all three tools in sequence and provide an executive summary of the recommended approach.
    """)
    
    print("=" * 80)
    print("STRATEGIC ROADMAP GENERATION COMPLETE")
    print("=" * 80)
    print(result)