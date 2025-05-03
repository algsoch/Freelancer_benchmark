import os
import uuid
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime, date
import time
from fastapi import FastAPI, Depends, HTTPException, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, UUID4, Field
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATA_FILE = "freelancer_notes.json"

# Custom JSON encoder to handle dates and UUIDs
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return super().default(obj)

# Data access functions
def initialize_data_file():
    """Create the data file if it doesn't exist"""
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w') as f:
            json.dump([], f)

def read_data():
    """Read all notes from the JSON file"""
    initialize_data_file()
    try:
        with open(DATA_FILE, 'r') as f:
            content = f.read().strip()
            if not content:  # Handle empty file case
                return []
            return json.loads(content)
    except json.JSONDecodeError:
        # If the file is corrupted, start with empty data
        return []

def write_data(data):
    """Write notes to the JSON file with atomic operation"""
    # Write to a temporary file first to ensure atomic operation
    temp_file = f"{DATA_FILE}.tmp"
    with open(temp_file, 'w') as f:
        json.dump(data, f, cls=CustomJSONEncoder, indent=2)
    
    # Rename temp file to the actual file (atomic operation)
    os.replace(temp_file, DATA_FILE)

# Pydantic Models
class FreelancerNoteBase(BaseModel):
    name: str
    skills: Optional[str] = None
    projects: Optional[str] = None
    clients: Optional[str] = None
    work_summary: Optional[str] = None

class FreelancerNoteCreate(FreelancerNoteBase):
    pass

class FreelancerNoteUpdate(FreelancerNoteBase):
    name: Optional[str] = None

class FreelancerNote(FreelancerNoteBase):
    id: UUID4
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# CRUD Operations
def get_notes(skip: int = 0, limit: int = 100):
    data = read_data()
    # Convert string dates back to datetime objects
    for note in data:
        if isinstance(note["created_at"], str):
            note["created_at"] = datetime.fromisoformat(note["created_at"])
        if isinstance(note["updated_at"], str):
            note["updated_at"] = datetime.fromisoformat(note["updated_at"])
    
    return data[skip:skip+limit]

def get_note(note_id: uuid.UUID):
    notes = read_data()
    for note in notes:
        if note["id"] == str(note_id):
            # Convert string dates back to datetime objects
            if isinstance(note["created_at"], str):
                note["created_at"] = datetime.fromisoformat(note["created_at"])
            if isinstance(note["updated_at"], str):
                note["updated_at"] = datetime.fromisoformat(note["updated_at"])
            return note
    
    raise HTTPException(status_code=404, detail="Note not found")

def create_note(note: FreelancerNoteCreate):
    notes = read_data()
    now = datetime.now()
    
    new_note = note.model_dump()
    new_note["id"] = str(uuid.uuid4())
    new_note["created_at"] = now
    new_note["updated_at"] = now
    
    notes.append(new_note)
    write_data(notes)
    
    # Convert string dates back to datetime objects for return value
    new_note["created_at"] = now
    new_note["updated_at"] = now
    return new_note

def update_note(note_id: uuid.UUID, note: FreelancerNoteUpdate):
    notes = read_data()
    now = datetime.now()
    
    for i, existing_note in enumerate(notes):
        if existing_note["id"] == str(note_id):
            # Keep original created_at, but update updated_at
            created_at = existing_note["created_at"]
            
            # Update with new data, excluding unset fields
            update_data = note.model_dump(exclude_unset=True)
            existing_note.update(update_data)
            existing_note["updated_at"] = now
            
            notes[i] = existing_note
            write_data(notes)
            
            # Convert string dates back to datetime objects for return value
            if isinstance(existing_note["created_at"], str):
                existing_note["created_at"] = datetime.fromisoformat(created_at)
            existing_note["updated_at"] = now
            
            return existing_note
    
    raise HTTPException(status_code=404, detail="Note not found")

def delete_note(note_id: uuid.UUID):
    notes = read_data()
    deleted_note = None
    
    for i, note in enumerate(notes):
        if note["id"] == str(note_id):
            deleted_note = note
            # Convert string dates back to datetime objects for return value
            if isinstance(note["created_at"], str):
                deleted_note["created_at"] = datetime.fromisoformat(note["created_at"])
            if isinstance(note["updated_at"], str):
                deleted_note["updated_at"] = datetime.fromisoformat(note["updated_at"])
            
            notes.pop(i)
            write_data(notes)
            return deleted_note
    
    raise HTTPException(status_code=404, detail="Note not found")

# AI Integration
def get_insights(note, api_key=None):
    if api_key:
        genai.configure(api_key=api_key)
    elif GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    else:
        raise HTTPException(status_code=400, detail="Gemini API key is required")
    
    try:
        model = genai.GenerativeModel('gemini-flash-2.0')
        
        prompt = f"""
        Analyze this freelancer's information and provide insights:
        
        Name: {note["name"]}
        Skills: {note["skills"] or "Not specified"}
        Projects: {note["projects"] or "Not specified"}
        Clients: {note["clients"] or "Not specified"}
        Work Summary: {note["work_summary"] or "Not specified"}
        
        Please provide:
        1. Suggestions for improving their profile or skills
        2. Patterns or trends you notice in their work history
        3. Recommended client types or platforms based on their skills
        4. Other insights or opportunities they might explore
        """
        
        response = model.generate_content(prompt)
        return {
            "suggestions": response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")

# FastAPI Application
app = FastAPI(title="Freelancer Research Notebook")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create essential directories to ensure they exist
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Ensure directories exist
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
(STATIC_DIR / "css").mkdir(exist_ok=True)
(STATIC_DIR / "js").mkdir(exist_ok=True)

# Setup templates and static files with absolute paths
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize data file on startup
@app.on_event("startup")
async def startup_event():
    initialize_data_file()
    print(f"Base directory: {BASE_DIR}")
    print(f"Templates directory: {TEMPLATES_DIR}")
    print(f"Templates exist: {TEMPLATES_DIR.exists()}")
    print(f"Template files: {[f.name for f in TEMPLATES_DIR.glob('*.html') if f.is_file()]}")
    print(f"Static directory: {STATIC_DIR}")
    print(f"Static exists: {STATIC_DIR.exists()}")
    
    # Create template files if they don't exist
    if not (TEMPLATES_DIR / "index.html").exists():
        create_template_files()

def create_template_files():
    """Create essential template files if they don't exist"""
    # Create base.html template
    base_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Freelancer Research Notebook{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-100 min-h-screen">
    <nav class="bg-blue-600 text-white p-4">
        <div class="container mx-auto flex justify-between items-center">
            <h1 class="text-2xl font-bold">Freelancer Research Notebook</h1>
            <div>
                <a href="/" class="px-4 py-2 hover:underline">Dashboard</a>
                <a href="/add" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
                    Add New Note
                </a>
            </div>
        </div>
    </nav>
    
    <main class="container mx-auto my-8 px-4">
        {% block content %}{% endblock %}
    </main>
    
    <footer class="bg-gray-800 text-white p-4 text-center">
        <p>Freelancer Research Notebook &copy; 2025</p>
    </footer>
</body>
</html>"""

    # Create index.html template
    index_html = """{% extends "base.html" %}

{% block content %}
<div class="mb-8">
    <h2 class="text-3xl font-bold mb-4">Freelancer Notes</h2>
    
    <div class="mb-4">
        <input type="text" id="searchInput" placeholder="Search notes..." 
               class="w-full p-2 border rounded shadow">
    </div>
    
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {% for note in notes %}
        <div class="bg-white rounded-lg shadow-md p-6 note-card">
            <h3 class="text-xl font-bold mb-2">{{ note.name }}</h3>
            
            <div class="mb-2">
                <p class="font-semibold">Skills:</p>
                <p class="text-gray-700">{{ note.skills or "Not specified" }}</p>
            </div>
            
            <div class="mb-2">
                <p class="font-semibold">Projects:</p>
                <p class="text-gray-700">{{ note.projects or "Not specified" }}</p>
            </div>
            
            <div class="mb-2">
                <p class="font-semibold">Clients:</p>
                <p class="text-gray-700">{{ note.clients or "Not specified" }}</p>
            </div>
            
            <div class="mb-4">
                <p class="font-semibold">Work Summary:</p>
                <p class="text-gray-700">{{ note.work_summary or "Not specified" }}</p>
            </div>
            
            <div class="text-sm text-gray-500 mb-4">
                <p>Created: {{ note.created_at.strftime('%Y-%m-%d %H:%M') }}</p>
                <p>Updated: {{ note.updated_at.strftime('%Y-%m-%d %H:%M') }}</p>
            </div>
            
            <div class="flex space-x-2">
                <a href="/edit/{{ note.id }}" 
                   class="bg-yellow-500 hover:bg-yellow-600 text-white font-bold py-1 px-3 rounded">
                    Edit
                </a>
                <a href="/delete/{{ note.id }}" 
                   class="bg-red-500 hover:bg-red-600 text-white font-bold py-1 px-3 rounded"
                   onclick="return confirm('Are you sure you want to delete this note?')">
                    Delete
                </a>
                <a href="/insight/{{ note.id }}" 
                   class="bg-purple-500 hover:bg-purple-600 text-white font-bold py-1 px-3 rounded">
                    AI Insights
                </a>
            </div>
        </div>
        {% else %}
        <div class="col-span-full text-center p-8 bg-white rounded-lg shadow">
            <p class="text-xl">No notes found. Create your first freelancer note!</p>
            <a href="/add" class="mt-4 inline-block bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
                Add New Note
            </a>
        </div>
        {% endfor %}
    </div>
</div>

<script>
    document.addEventListener('DOMContentLoaded', function() {
        const searchInput = document.getElementById('searchInput');
        const noteCards = document.querySelectorAll('.note-card');
        
        searchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            
            noteCards.forEach(card => {
                const content = card.textContent.toLowerCase();
                if (content.includes(searchTerm)) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        });
    });
</script>
{% endblock %}"""

    # Create add_note.html
    add_html = """{% extends "base.html" %}

{% block content %}
<div class="max-w-2xl mx-auto bg-white p-8 rounded-lg shadow">
    <h2 class="text-2xl font-bold mb-6">Add New Freelancer Note</h2>
    
    <form action="/add" method="post">
        <div class="mb-4">
            <label for="name" class="block text-gray-700 font-bold mb-2">Name</label>
            <input type="text" id="name" name="name" required
                   class="w-full px-3 py-2 border rounded focus:outline-none focus:ring focus:border-blue-300">
        </div>
        
        <div class="mb-4">
            <label for="skills" class="block text-gray-700 font-bold mb-2">Skills</label>
            <textarea id="skills" name="skills" rows="3"
                      class="w-full px-3 py-2 border rounded focus:outline-none focus:ring focus:border-blue-300"></textarea>
        </div>
        
        <div class="mb-4">
            <label for="projects" class="block text-gray-700 font-bold mb-2">Projects</label>
            <textarea id="projects" name="projects" rows="3"
                      class="w-full px-3 py-2 border rounded focus:outline-none focus:ring focus:border-blue-300"></textarea>
        </div>
        
        <div class="mb-4">
            <label for="clients" class="block text-gray-700 font-bold mb-2">Clients</label>
            <textarea id="clients" name="clients" rows="3"
                      class="w-full px-3 py-2 border rounded focus:outline-none focus:ring focus:border-blue-300"></textarea>
        </div>
        
        <div class="mb-6">
            <label for="work_summary" class="block text-gray-700 font-bold mb-2">Work Summary</label>
            <textarea id="work_summary" name="work_summary" rows="5"
                      class="w-full px-3 py-2 border rounded focus:outline-none focus:ring focus:border-blue-300"></textarea>
        </div>
        
        <div class="flex justify-between">
            <a href="/" class="bg-gray-500 hover:bg-gray-600 text-white font-bold py-2 px-4 rounded">
                Cancel
            </a>
            <button type="submit" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
                Save Note
            </button>
        </div>
    </form>
</div>
{% endblock %}"""

    # Create edit_note.html
    edit_html = """{% extends "base.html" %}

{% block content %}
<div class="max-w-2xl mx-auto bg-white p-8 rounded-lg shadow">
    <h2 class="text-2xl font-bold mb-6">Edit Freelancer Note</h2>
    
    <form action="/edit/{{ note.id }}" method="post">
        <div class="mb-4">
            <label for="name" class="block text-gray-700 font-bold mb-2">Name</label>
            <input type="text" id="name" name="name" value="{{ note.name }}" required
                   class="w-full px-3 py-2 border rounded focus:outline-none focus:ring focus:border-blue-300">
        </div>
        
        <div class="mb-4">
            <label for="skills" class="block text-gray-700 font-bold mb-2">Skills</label>
            <textarea id="skills" name="skills" rows="3"
                      class="w-full px-3 py-2 border rounded focus:outline-none focus:ring focus:border-blue-300">{{ note.skills }}</textarea>
        </div>
        
        <div class="mb-4">
            <label for="projects" class="block text-gray-700 font-bold mb-2">Projects</label>
            <textarea id="projects" name="projects" rows="3"
                      class="w-full px-3 py-2 border rounded focus:outline-none focus:ring focus:border-blue-300">{{ note.projects }}</textarea>
        </div>
        
        <div class="mb-4">
            <label for="clients" class="block text-gray-700 font-bold mb-2">Clients</label>
            <textarea id="clients" name="clients" rows="3"
                      class="w-full px-3 py-2 border rounded focus:outline-none focus:ring focus:border-blue-300">{{ note.clients }}</textarea>
        </div>
        
        <div class="mb-6">
            <label for="work_summary" class="block text-gray-700 font-bold mb-2">Work Summary</label>
            <textarea id="work_summary" name="work_summary" rows="5"
                      class="w-full px-3 py-2 border rounded focus:outline-none focus:ring focus:border-blue-300">{{ note.work_summary }}</textarea>
        </div>
        
        <div class="flex justify-between">
            <a href="/" class="bg-gray-500 hover:bg-gray-600 text-white font-bold py-2 px-4 rounded">
                Cancel
            </a>
            <button type="submit" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
                Update Note
            </button>
        </div>
    </form>
</div>
{% endblock %}"""

    # Create insight.html
    insight_html = """{% extends "base.html" %}

{% block content %}
<div class="max-w-4xl mx-auto">
    <div class="bg-white p-6 rounded-lg shadow mb-6">
        <h2 class="text-2xl font-bold mb-4">Freelancer: {{ note.name }}</h2>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div>
                <h3 class="font-bold text-lg">Skills</h3>
                <p class="text-gray-700">{{ note.skills or "Not specified" }}</p>
            </div>
            
            <div>
                <h3 class="font-bold text-lg">Projects</h3>
                <p class="text-gray-700">{{ note.projects or "Not specified" }}</p>
            </div>
            
            <div>
                <h3 class="font-bold text-lg">Clients</h3>
                <p class="text-gray-700">{{ note.clients or "Not specified" }}</p>
            </div>
            
            <div>
                <h3 class="font-bold text-lg">Work Summary</h3>
                <p class="text-gray-700">{{ note.work_summary or "Not specified" }}</p>
            </div>
        </div>
    </div>
    
    <div class="bg-white p-6 rounded-lg shadow">
        <h2 class="text-2xl font-bold mb-4">AI Insights</h2>
        
        {% if error %}
            <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
                <p>{{ error }}</p>
                
                <div class="mt-4">
                    <form action="/ai/insights/{{ note.id }}" method="post" class="flex flex-col space-y-2">
                        <label for="api_key" class="font-bold">Enter your Gemini API Key:</label>
                        <input type="text" id="api_key" name="api_key" 
                               class="px-3 py-2 border rounded focus:outline-none focus:ring focus:border-blue-300"
                               placeholder="Your Gemini API Key">
                        <button type="submit" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
                            Get Insights
                        </button>
                    </form>
                </div>
            </div>
        {% elif insights %}
            <div class="prose max-w-none">
                {{ insights.suggestions | safe }}
            </div>
        {% else %}
            <div class="text-center py-8">
                <p class="text-lg mb-4">Loading insights...</p>
                <div class="loader mx-auto"></div>
            </div>
            
            <script>
                // Auto-fetch insights when page loads
                document.addEventListener('DOMContentLoaded', function() {
                    fetch('/ai/insights/{{ note.id }}')
                        .then(response => response.json())
                        .then(data => {
                            if (data.suggestions) {
                                const insightsDiv = document.querySelector('.prose');
                                insightsDiv.innerHTML = data.suggestions;
                                document.querySelector('.text-center').style.display = 'none';
                            }
                        })
                        .catch(error => {
                            console.error('Error fetching insights:', error);
                            const insightsDiv = document.querySelector('.prose');
                            insightsDiv.innerHTML = '<div class="text-red-500">Error loading insights. Please try again.</div>';
                            document.querySelector('.text-center').style.display = 'none';
                        });
                });
            </script>
            
            <style>
                .loader {
                    border: 5px solid #f3f3f3;
                    border-top: 5px solid #3498db;
                    border-radius: 50%;
                    width: 50px;
                    height: 50px;
                    animation: spin 2s linear infinite;
                }
                
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        {% endif %}
    </div>
    
    <div class="mt-6">
        <a href="/" class="bg-gray-500 hover:bg-gray-600 text-white font-bold py-2 px-4 rounded">
            Back to Dashboard
        </a>
    </div>
</div>
{% endblock %}"""

    # Write template files
    (TEMPLATES_DIR / "base.html").write_text(base_html)
    (TEMPLATES_DIR / "index.html").write_text(index_html)
    (TEMPLATES_DIR / "add_note.html").write_text(add_html)
    (TEMPLATES_DIR / "edit_note.html").write_text(edit_html)
    (TEMPLATES_DIR / "insight.html").write_text(insight_html)
    
    print("Created template files successfully")

@app.get("/debug")
def debug_data():
    data = read_data()
    return {"count": len(data), "data": data}

# API routes
@app.post("/notes/", response_model=FreelancerNote)
def create_note_endpoint(note: FreelancerNoteCreate):
    return create_note(note=note)

@app.get("/notes/", response_model=List[FreelancerNote])
def read_notes_endpoint(skip: int = 0, limit: int = 100):
    notes = get_notes(skip=skip, limit=limit)
    return notes

@app.get("/notes/{note_id}", response_model=FreelancerNote)
def read_note_endpoint(note_id: uuid.UUID):
    return get_note(note_id=note_id)

@app.put("/notes/{note_id}", response_model=FreelancerNote)
def update_note_endpoint(note_id: uuid.UUID, note: FreelancerNoteUpdate):
    return update_note(note_id=note_id, note=note)

@app.delete("/notes/{note_id}", response_model=FreelancerNote)
def delete_note_endpoint(note_id: uuid.UUID):
    return delete_note(note_id=note_id)

@app.post("/ai/insights/{note_id}")
def get_ai_insights_endpoint(note_id: uuid.UUID, api_key: Optional[str] = None):
    note = get_note(note_id=note_id)
    return get_insights(note, api_key)

# UI routes
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    notes = get_notes()
    return templates.TemplateResponse("index.html", {"request": request, "notes": notes})

@app.get("/add", response_class=HTMLResponse)
def add_note_form(request: Request):
    return templates.TemplateResponse("add_note.html", {"request": request})

@app.post("/add")
def add_note_endpoint(
    request: Request,
    name: str = Form(...),
    skills: str = Form(""),
    projects: str = Form(""),
    clients: str = Form(""),
    work_summary: str = Form("")
):
    note = FreelancerNoteCreate(
        name=name,
        skills=skills,
        projects=projects,
        clients=clients,
        work_summary=work_summary
    )
    create_note(note=note)
    return RedirectResponse(url="/", status_code=303)

@app.get("/edit/{note_id}", response_class=HTMLResponse)
def edit_note_form(request: Request, note_id: uuid.UUID):
    note = get_note(note_id=note_id)
    return templates.TemplateResponse("edit_note.html", {"request": request, "note": note})

@app.post("/edit/{note_id}")
def edit_note_endpoint(
    request: Request,
    note_id: uuid.UUID,
    name: str = Form(...),
    skills: str = Form(""),
    projects: str = Form(""),
    clients: str = Form(""),
    work_summary: str = Form("")
):
    note = FreelancerNoteUpdate(
        name=name,
        skills=skills,
        projects=projects,
        clients=clients,
        work_summary=work_summary
    )
    update_note(note_id=note_id, note=note)
    return RedirectResponse(url="/", status_code=303)

@app.get("/delete/{note_id}")
def delete_note_ui(note_id: uuid.UUID):
    delete_note(note_id=note_id)
    return RedirectResponse(url="/", status_code=303)

@app.get("/insight/{note_id}", response_class=HTMLResponse)
def get_insight_ui(request: Request, note_id: uuid.UUID):
    note = get_note(note_id=note_id)
    try:
        insights = get_insights(note)
        return templates.TemplateResponse(
            "insight.html", 
            {"request": request, "note": note, "insights": insights}
        )
    except HTTPException as e:
        return templates.TemplateResponse(
            "insight.html", 
            {"request": request, "note": note, "error": e.detail}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)