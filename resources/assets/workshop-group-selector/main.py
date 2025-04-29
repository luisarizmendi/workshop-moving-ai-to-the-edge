### group_selector_service
# A minimal FastAPI + HTML UI service for group selection, configurable via Kubernetes ConfigMap

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
import uvicorn

app = FastAPI()

# In-memory state (no persistence)
user_groups = {}

# Load group config from mounted ConfigMap (as file)
CONFIG_PATH = Path("./config/groups.json")
if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as f:
        GROUP_CONFIG = json.load(f)
else:
    GROUP_CONFIG = {f"Group {i:02}": {"url": "#", "info": "No info"} for i in range(1, 51)}
    GROUP_CONFIG["Group 99"] = {"url": "#", "info": "Reserved"}

# Serve static files (for dark-mode CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    group_html = ""
    for group, data in GROUP_CONFIG.items():
        members = [u for u, g in user_groups.items() if g == group]
        member_list = "<br>".join(members)
        group_html += (
            f"<div class='group'>"
            f"<h3>{group}</h3>"
            f"<p><strong>Workshop:</strong> <a href='{data['url']}' target='_blank'>{data['url']}</a></p>"
            f"<div class='button-row'>"
            f"<form method='post' action='/assign' class='inline-form'>"
            f"<input type='hidden' name='group' value='{group}' />"
            f"<button>Join</button>"
            f"</form>"
            f"<form method='post' action='/remove' class='inline-form'>"
            f"<button>Leave</button>"
            f"</form>"
            f"</div>"
            f"<p><strong>Members:</strong><br>{member_list if member_list else 'No members yet'}</p>"
            f"<p class='info'>{data['info'].replace(chr(10), '<br>')}</p>"
            f"</div>"
        )

    return (
        "<html>"
        "<head>"
        "<link href='/static/style.css' rel='stylesheet'>"
        "<title>Workshop group selector</title>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "</head>"
        "<body>"
        "<h1>Choose your workshop group</h1>"
        "<p>Please enter your full name or email before choosing a group:</p>"
        "<form method='get' action='/' onsubmit='return false;'>"
        "<input type='text' id='username' name='username' placeholder='Enter your name' required />"
        "</form>"
        "<div class='group-container'>"
        f"{group_html}"
        "</div>"
        "<script>"
        "document.querySelectorAll(\"form\").forEach(form => {"
        "    form.addEventListener(\"submit\", function(e) {"
        "        const usernameInput = document.getElementById(\"username\");"
        "        if (!usernameInput || !usernameInput.value) {"
        "            alert(\"Please enter your name at the top.\");"
        "            e.preventDefault();"
        "            return;"
        "        }"
        "        let hiddenInput = document.createElement(\"input\");"
        "        hiddenInput.type = \"hidden\";"
        "        hiddenInput.name = \"username\";"
        "        hiddenInput.value = usernameInput.value;"
        "        form.appendChild(hiddenInput);"
        "    });"
        "});"
        "</script>"
        "</body>"
        "</html>"
    )

@app.post("/assign")
async def assign(username: str = Form(...), group: str = Form(...)):
    old_group = user_groups.get(username)
    user_groups[username] = group
    return RedirectResponse(url="/", status_code=303)

@app.post("/remove")
async def remove(username: str = Form(...)):
    if username in user_groups:
        user_groups.pop(username)
    return RedirectResponse(url="/", status_code=303)

# Entry point
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
