# GitHub Copilot Starter

This repository is used for hands-on practice with **three modes of interacting with AI**.

> 이 리포지토리는 AI와 인터랙션하는 세 가지 모드를 직접 실습하기 위한 스타터 파일입니다.

---

## Getting Started / 시작하기

1. Click **"Use this template"** → **"Create a new repository"** to copy this repo to your own GitHub account.
2. In your new repo, click **`<> Code`** → **`Codespaces`** → **`Create codespace on main`**.
3. Wait a moment for the environment to load — Python (Anaconda) and GitHub Copilot are pre-installed.

> "Use this template" 버튼을 눌러 자신의 계정으로 복사한 뒤 Codespace를 여세요.

---

## The Three Modes / 세 가지 모드

| | Mode 1 | Mode 2 | Mode 3 |
|---|---|---|---|
| **Interface** | Web Browser | IDE (Editor) | CLI (Terminal) |
| **Tool** | Google Gemini | GitHub Copilot Chat | GitHub Copilot CLI |
| **Best for** | Conversation, images, ideas | Coding, data analysis | Agentic coding, automation |
| **File access** | ❌ No | ✅ Yes | ✅ Yes |
| **Runs code** | ❌ No | ✅ Yes | ✅ Yes (autonomously) |
| **Barrier** | 🟢 Low | 🟡 Medium | 🔴 High |

---

## The Task / 오늘의 과제

You have one data file: **`sample.csv`** — a simple table of daily activities and hours.

Your goal: create a bar chart from this data using **each of the three modes**, saving the result as `mode1.png`, `mode2.png`, and `mode3.png`.

> `sample.csv` 데이터를 가지고 막대 차트를 세 가지 모드로 각각 만들어보세요.

---

### Mode 1 — Web Interface (Gemini) → `mode1.py`

> Gemini has **no access to your files**. You must copy-paste the data in, and copy-paste the code back out.

1. Go to [gemini.google.com](https://gemini.google.com)
2. Paste the contents of `sample.csv` into the chat and ask:
   > *"Here is my CSV data: [paste contents]. Write Python code using matplotlib to create a bar chart. Title: 'My Day'. Save it as mode1.png"*
3. Copy the code Gemini gives you
4. Paste it into `mode1.py` in this Codespace
5. Open the terminal and run:
   ```bash
   python mode1.py
   ```
6. Right-click `mode1.png` in the file explorer → **Download**

---

### Mode 2 — IDE / Copilot Chat → `mode2.py`

> Copilot Chat **sees all your files** in the Codespace. No copy-paste needed.

1. Open Copilot Chat (left sidebar icon, or `Ctrl+Shift+I` / `Cmd+Shift+I`)
2. Type:
   > *"Read sample.csv and create a bar chart using matplotlib. Title: 'My Day'. Save the figure as mode2.png. Write the code into mode2.py"*
3. Copilot will generate and insert the code directly into `mode2.py`
4. Run in terminal:
   ```bash
   python mode2.py
   ```
5. Right-click `mode2.png` → **Download**

---

### Mode 3 — CLI / GitHub Copilot CLI → `mode3.py`

> Copilot CLI is a **terminal-native AI agent** — similar to Claude Code, but built by GitHub.
> It reads your files, writes code, and runs it, all from the terminal without any GUI.

#### Setup (one-time, requires GitHub Copilot subscription)

```bash
# Install — requires Node.js 22+
npm install -g @github/copilot

# Launch
copilot
```

#### Usage inside a Copilot CLI session

```
> Read sample.csv and create a bar chart saved as mode3.png
```

Copilot CLI will read the file, write the code, run it, and save `mode3.png` — all autonomously.

#### Useful slash commands

| Command | Description |
|---|---|
| `/plan` | Plan before writing code |
| `/model` | Switch between Claude, GPT, etc. |
| `/help` | See all available commands |

> **Note:** Mode 3 requires an active GitHub Copilot subscription (Free, Pro, Business, or Enterprise).
> After your **GitHub Education** approval, you will have access. Try this on your own after approval!

---

## Saving Your Work / 작업 저장

**Option 1 — Git (recommended):**
```bash
git add .
git commit -m "my first visualization"
git push
```
Your work is now saved permanently on GitHub.

**Option 2 — Direct download:**
Right-click any file in the file explorer → **Download**

> Your Codespace stays alive for 30 days after your last use.
> Reopen anytime at [github.com/codespaces](https://github.com/codespaces).
